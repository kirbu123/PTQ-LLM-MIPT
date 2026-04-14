import contextlib
from typing import Dict, List, Optional, Tuple, Union

from itertools import islice

import os
import datetime

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler

from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStrategy,
)
from llmcompressor.modifiers.utils.kernels import apply_conv
from compressed_tensors.quantization.quant_args import ActivationOrdering
from compressed_tensors.utils import (
    align_module_device,
    get_execution_device,
    getattr_chain,
    match_named_modules,
    update_offload_parameter,
)
from loguru import logger
from pydantic import PrivateAttr

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.quantization.gptq.gptq_quantize import (
    accumulate_hessian,
    accumulate_hessian_next_reg,
    make_empty_hessian,
    quantize_weight,
)
from llmcompressor.modifiers.quantization.quantization import QuantizationMixin
from llmcompressor.sentinel import Sentinel
from llmcompressor.utils.metric_logging import CompressionLogger
from llmcompressor.utils.loss import LOSS_DICT
from llmcompressor.utils.next_strats import NEXT_STRATS_DICT
from llmcompressor.utils.metric_logging import (
    compute_hessian_metrics,
    plot_eigenvalue_list,
    compute_quantized_hessian_metrics,
    pushed_l2_hessian_eigentrace_after_gptq
)

from torch.utils.tensorboard import SummaryWriter

__all__ = ["GPTQModifier"]


class GPTQModifier(Modifier, QuantizationMixin):
    """
    Implements the GPTQ algorithm from https://arxiv.org/abs/2210.17323. This modifier
    uses activations to calibrate a hessian matrix, which is then used to determine
    optimal quantizion values and orderings for the model weights.

    | Sample yaml:
    | test_stage:
    |    obcq_modifiers:
    |      GPTQModifier:
    |          block_size: 128
    |          dampening_frac: 0.001
    |          offload_hessians: False
    |          actorder: static
    |          config_groups:
    |            group_0:
    |                targets:
    |                  - "Linear"
    |                input_activations: null
    |                output_activations: null
    |                weights:
    |                    num_bits: 8
    |                    type: "int"
    |                    symmetric: true
    |                    strategy: group
    |                    group_size: 128

    Lifecycle:
        - on_initialize
            - apply config to model
        - on_start
            - add activation calibration hooks
            - add gptq weight calibration hooks
        - on_sequential_epoch_end
            - quantize_weight
        - on_finalize
            - remove_hooks()
            - model.apply(freeze_module_quantization)

    :param sequential_targets: list of layer names to compress during GPTQ, or
        '__ALL__' to compress every layer in the model
    :param block_size: Used to determine number of columns to compress in one pass
    :param dampening_frac: Amount of dampening to apply to H, as a fraction of the
        diagonal norm
    :param actorder: order in which weight columns are quantized. Defaults to "static"
        activation ordering, which achieves best accuracy recovery with no runtime cost.
        For more information, see https://github.com/vllm-project/vllm/pull/8135
    :param offload_hessians: Set to True for decreased memory usage but increased
        runtime.

    :param config_groups: dictionary specifying quantization schemes to apply to target
        modules. Modules not matching a scheme target will NOT be quantized.
    :param targets: list of layer names to quantize if a scheme is provided. Defaults
        to Linear layers
    :param ignore: optional list of module class names or submodule names to not
        quantize even if they match a target in config_groups. Defaults to empty list.
    :param scheme: a single quantization scheme to apply to the model. This is a
        dictionary that supports all keys from QuantizationScheme except targets, which
        will be set to the targets parameter set at the modifier level. Can also be set
        to a dictionary of the format `preset_scheme_name: targets` for example:
        `W8A8: ['Linear']` for weight and activation 8-bit.
    :param kv_cache_scheme: optional QuantizationArgs, that specify the
        quantization of the kv cache. If None, kv cache is not quantized.
        When applying kv cache quantization to transformer AutoModelForCausalLM,
        the kv_cache_scheme gets converted into a QuantizationScheme that:
            - targets the `q_proj` and `k_proj` modules of the model. The outputs
              of those modules are the keys and values that might be cached
            - quantizes the outputs of the aformentioned layers, so that
              keys and values are compressed before storing them in the cache
        There is an explicit assumption that the model contains modules with
        `k_proj` and `v_proj` in their names. If this is not the case
        and kv_cache_scheme != None, the quantization of kv cache will fail
    :param next_reg_lam: regularization parameter for next layer influence during
        quantization. Defaults to 0.0.
    """

    # gptq modifier arguments
    sequential_targets: Union[str, List[str], None] = None
    block_size: int = 128
    dampening_frac: Optional[float] = 0.01
    # TODO: this does not serialize / will be incorrectly written
    actorder: Optional[Union[ActivationOrdering, Sentinel]] = Sentinel("static")
    offload_hessians: bool = False

    next_reg_lam: float = 0.0
    next_loss_lam: float = 0.0
    kernel_mode: str = 'default'
    lam_lr: float = 3e-4
    lam_loss_name: str = 'HessianLossNormed'
    next_strat_name: str = 'BasicStrat'
    opt_steps_num: int = 10
    k_next: int = 2
    do_hessian_plot: bool = False
    reinitialize_lam: bool = False
    lam_optimize_method: str = 'multistep'
    lam_ls_ridge: float = 1e-4
    lam_pl_target_scale: float = 1.0

    # Lam optimize params
    lam_optimize: bool = False
    log_dir: str = './log'
    _log_writer: SummaryWriter = PrivateAttr()
    _hessian_log_dir: str = PrivateAttr()

    # Add these as PrivateAttr since they're not serializable/model fields
    _with_eigens: bool = PrivateAttr()
    _lam_tensor: torch.nn.Parameter = PrivateAttr()
    _lam_optimizer: torch.optim.Optimizer = PrivateAttr()
    _lam_scheduler: Optional[LRScheduler] = PrivateAttr()
    _lam_loss = PrivateAttr()
    _next_strat = PrivateAttr()
    _step_num: int = PrivateAttr()
    _step_num_no_optimize: int = PrivateAttr()

    # private variables
    _module_names: Dict[torch.nn.Module, str] = PrivateAttr(default_factory=dict)
    _hessians: Dict[torch.nn.Module, torch.Tensor] = PrivateAttr(default_factory=dict)
    _num_samples: Dict[torch.nn.Module, int] = PrivateAttr(default_factory=dict)
    # _eigenvals: Dict[torch.nn.Module, torch.Tensor] = PrivateAttr(default_factory=dict)
    # _eigenvects: Dict[torch.nn.Module, torch.Tensor] = PrivateAttr(default_factory=dict)
    _eigens: Dict = PrivateAttr(default_factory=dict)

    def resolve_quantization_config(self) -> QuantizationConfig:
        config = super().resolve_quantization_config()

        def resolve_actorder(existing):
            # sentinel default only overrides if existing is None
            if self.actorder == Sentinel("static"):
                return ActivationOrdering.STATIC if existing is None else existing

            # user-provided value always attempts to override
            if existing is None or self.actorder == existing:
                return self.actorder

            # if existing provided and conflicts
            raise ValueError(
                "Cannot resolve activation ordering when both "
                "`GPTQModifier.actorder` and `QuantizationScheme.actorder` "
                f"are provided and differ ({self.actorder}, {existing}). "
                "Either unset `GPTQModifier.actorder` or "
                "remove `actorder` from config groups."
            )

        for scheme in config.config_groups.values():
            assert isinstance(scheme, QuantizationScheme)
            if (
                getattr_chain(scheme, "weights.strategy", None)
                == QuantizationStrategy.GROUP
            ):
                scheme.weights.actorder = resolve_actorder(scheme.weights.actorder)
        return config

    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Initialize and run the GPTQ algorithm on the current state

        :param state: session state storing input model and calibration data
        """
        # apply config to model and prepare calibration hooks
        if QuantizationMixin.has_config(self):
            QuantizationMixin.initialize_quantization(self, state.model)

        # prepare module names
        self._module_names = {
            m: name
            for name, m in match_named_modules(
                state.model, self.resolved_targets, self.ignore
            )
        }

        # Tensorboard init
        log_path = os.path.join(self.log_dir, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        self._log_writer = SummaryWriter(log_dir=log_path)

        self._hessian_log_dir = os.path.join(self.log_dir, 'hessians_info')

        self._step_num_no_optimize = 0

        self._lam_tensor = torch.nn.Parameter(
            torch.full((self.k_next,), self.next_reg_lam, dtype=torch.float32),
            requires_grad=True
        )
        self._lam_optimizer = torch.optim.Adam([self._lam_tensor], lr=self.lam_lr)
        self._lam_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self._lam_optimizer, 
                T_max=self.opt_steps_num,
            )
        self._lam_loss = LOSS_DICT[self.lam_loss_name]()
        self._next_strat = NEXT_STRATS_DICT[self.next_strat_name]
        self._with_eigens = self._lam_loss.get_with_eigens()
        self._step_num = 0

        self._with_eigens = self.lam_optimize and self.lam_optimize_method == 'multistep'

        return True

    def on_start(self, state: State, event: Event, **kwargs):
        self.started_ = True

        # register quantization calibration hooks
        # assume quantization has been initialized by this modifier or one before it
        QuantizationMixin.start_calibration(self, state.model)

        # register gptq hooks
        added_hook = False
        for _, module in match_named_modules(
            state.model, self.resolved_targets, self.ignore
        ):
            if getattr_chain(module, "quantization_scheme.weights", None) is not None:
                # HACK: previously, embeddings were not quantized because they were not
                # accessible by the layer compressor. For now, we manually ignore it,
                # but in the FUTURE this should be ignored by the user
                if not isinstance(module, torch.nn.Embedding):
                    self.register_hook(module, self.calibrate_module, "forward")
                    added_hook = True

        if not added_hook:
            raise ValueError(
                "GPTQModifier requires a weight quantization config be specified by "
                "this modifier or a modifier preceding it"
            )

    def on_event(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state, None)

        if event.type_ == EventType.SEQUENTIAL_EPOCH_END:
            self.compress_modules()

        if event.type_ == EventType.CALIBRATION_EPOCH_END:
            self.compress_modules()

            if not self.ended_:
                self.on_end(state, None)

    def _get_next_module(self, current_module: torch.nn.Module) -> Optional[torch.nn.Module]:
        """
        Find the next module in the sequential order of targets
        
        :param current_module: current module to find next for
        :return: next module in sequence or None if not found
        """
        if current_module not in self._module_names:
            return None
        
        current_name = self._module_names[current_module]
        
        # Get all module names in order
        all_modules = list(self._module_names.items())
        
        # Find current module index
        current_idx = -1
        for idx, (mod, name) in enumerate(all_modules):
            if mod == current_module:
                current_idx = idx
                break
        
        if current_idx == -1 or current_idx + 1 >= len(all_modules):
            return None
        
        # Return next module
        next_module, next_name = all_modules[current_idx + 1]
        return next_module

    def calibrate_module(
        self,
        module: torch.nn.Module,
        args: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ):
        """
        Calibration hook used to accumulate the hessian of the input to the module

        :param module: module being calibrated
        :param args: inputs to the module, the first element of which is the
            cannonical input
        :param output: uncompressed module output, unused
        """
        # Assume that first argument is the input
        inp = args[0]

        # Initialize hessian if not present
        if module not in self._num_samples:
            init_device = (
                "cpu" if self.offload_hessians else get_execution_device(module)
            )
            self._hessians[module] = make_empty_hessian(module, device=init_device)
            self._num_samples[module] = 0

        # Get next module and its input (current module's output)
        module_next = None
        inp_next = None

        if self.next_loss_lam != 0.:
            # Find the next module in the sequential targets
            module_next = self._get_next_module(module)
            if module_next is not None:
                inp_next = output  # Use current module's output as next module's input

        # Accumulate hessian with input with optional offloading
        with self._maybe_onload_hessian(module):
            self._hessians[module], self._num_samples[module], eigens = accumulate_hessian(
                inp,
                module,
                self._hessians[module],
                self._num_samples[module],
                self._with_eigens
            )

            # self._eigenvals[module], self._eigenvects[module] = eigens['eigenvalues'], eigens['eigenvectors']
            self._eigens[module] = eigens

    def _update_lam_param_multistep(self, lam_loss, module, next_modules):
        if next_modules is not None:

            device = self._hessians[module].device
            module_name = self._module_names[module]

            if self._lam_tensor.device != device:
                self._lam_tensor = self._lam_tensor.to(device)

            if self.reinitialize_lam:
                with torch.no_grad():
                    self._lam_tensor.fill_(float(self.next_reg_lam))

            self._lam_optimizer = torch.optim.Adam([self._lam_tensor], lr=self.lam_lr)
            self._lam_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self._lam_optimizer, 
                T_max=self.opt_steps_num,
            )

            if not self._lam_tensor.requires_grad:
                self._lam_tensor.requires_grad_(True)

            init_sorted_eigens = None
            last_sorted_eigens = None

            with torch.enable_grad():
                for i in range(self.opt_steps_num):
                    self._lam_optimizer.zero_grad()

                    loss_reg, sorted_eigens = lam_loss(
                        lam=self._lam_tensor,
                        module=module,
                        next_modules=next_modules,
                        hessians=self._hessians,
                        eigens=self._eigens,
                        kernel_mode=self.kernel_mode
                    )

                    if i == 0: init_sorted_eigens = sorted_eigens
                    if sorted_eigens is not None: last_sorted_eigens = sorted_eigens

                    loss_reg.backward(retain_graph=True)

                    self._lam_optimizer.step()
                    self._lam_scheduler.step()

                    current_lr = self._lam_optimizer.param_groups[0]['lr']
                    self._log_writer.add_scalar('lam-lr', current_lr, self._step_num*self.opt_steps_num + i)
                    self._log_writer.add_scalar('loss-param', loss_reg.item(), self._step_num*self.opt_steps_num + i)

            self._log_writer.add_scalar('mean-lam-param', torch.mean(self._lam_tensor).item(), self._step_num)
            for i in range(len(self._lam_tensor)):
                self._log_writer.add_scalar(f'lam-param-dim-{i}', self._lam_tensor[i].item(), self._step_num)

            if init_sorted_eigens is not None and last_sorted_eigens is not None:

                estimated_eigens = None
                init_estimated_eigens, last_estimated_eigens = None, None
                if isinstance(init_sorted_eigens, tuple):
                    init_sorted_eigens, init_estimated_eigens = init_sorted_eigens
                if isinstance(last_sorted_eigens, tuple):
                    last_sorted_eigens, last_estimated_eigens = last_sorted_eigens

                max_plot_values = min(min(10000, len(init_sorted_eigens)), len(last_sorted_eigens))
                for idx in range(max_plot_values):
                    self._log_writer.add_scalar(
                        f'eigenvalue-spectr/module={module_name}:step={self._step_num}/init',
                        init_sorted_eigens[idx].item(),
                        idx
                    )
                    self._log_writer.add_scalar(
                        f'eigenvalue-spectr/module={module_name}:step={self._step_num}/last',
                        last_sorted_eigens[idx].item(),
                        idx
                    )
                    if init_estimated_eigens is not None:
                        self._log_writer.add_scalar(
                            f'eigenvalue-spectr/module={module_name}:step={self._step_num}/init-estimated',
                            init_estimated_eigens[idx].item(),
                            idx
                        )
                    if last_estimated_eigens is not None:
                        self._log_writer.add_scalar(
                            f'eigenvalue-spectr/module={module_name}:step={self._step_num}/last-estimated',
                            last_estimated_eigens[idx].item(),
                            idx
                        )

            self._step_num += 1

        return 0

    @staticmethod
    def _lam_trace_ridge_alpha(
        c: torch.Tensor,
        b: float,
        mu: float,
        alpha0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Minimize (c^T α - b)² + μ‖α - α₀‖²  ⇔  (c c^T + μ I) α = b c + μ α₀.
        """
        m = c.numel()
        G = c.unsqueeze(1) @ c.unsqueeze(0) + mu * torch.eye(
            m, device=c.device, dtype=c.dtype
        )
        rhs = b * c + mu * alpha0
        return torch.linalg.solve(G, rhs.unsqueeze(1)).squeeze(1)

    def _update_lam_param_onestep(self, module, next_modules):
        """
        Ridge closed-form update using only Hessian traces (no eigenvalues).

        Same combined matrix as quantize_weight:
        H(α) = H₀ + Σᵢ αᵢ apply_conv(Hᵢ). Then (1/d) tr H(α) = tr(H₀)/d + Σᵢ αᵢ tr(Kᵢ)/d,
        Kᵢ = apply_conv(Hᵢ). Target mean trace is lam_pl_target_scale · tr(H₀)/d.
        """
        if next_modules is None:
            return 0

        device = self._hessians[module].device
        module_name = self._module_names[module]

        if self._lam_tensor.device != device:
            self._lam_tensor = self._lam_tensor.to(device)
        
        if self.reinitialize_lam:
            with torch.no_grad():
                self._lam_tensor.fill_(float(self.next_reg_lam))

        H0 = self._hessians[module].to(device=device, dtype=torch.float32)
        d = H0.shape[0]
        k_next = self.k_next

        t0 = (torch.trace(H0) / d).item()
        c = torch.zeros(k_next, device=device, dtype=torch.float32)
        for i in range(min(len(next_modules), k_next)):
            mn = next_modules[i]
            if mn is None or mn not in self._hessians:
                continue
            Hn = self._hessians[mn].to(device=device, dtype=torch.float32)
            try:
                K = apply_conv(Hn, mode=self.kernel_mode)
            except RuntimeError:
                continue
            if K.shape != H0.shape:
                continue
            c[i] = torch.trace(K) / d

        if torch.all(c == 0):
            return 0

        mu1_ref = float(self.lam_pl_target_scale) * t0
        b = mu1_ref - t0
        lam_old = self._lam_tensor.detach().clone()
        mu = float(self.lam_ls_ridge)
        with torch.no_grad():
            alpha0 = torch.full(
                (k_next,), float(self.next_reg_lam), device=device, dtype=torch.float32
            )
            alpha = self._lam_trace_ridge_alpha(c, b, mu, alpha0)
            self._lam_tensor.data.copy_(alpha)

        with torch.no_grad():
            trace_combined = (torch.trace(H0) + d * torch.dot(c, alpha)) / d
            self._log_writer.add_scalar(
                "lam-trace-mean-combined", trace_combined.item(), self._step_num
            )
            self._log_writer.add_scalar(
                "lam-trace-mean-target", mu1_ref, self._step_num
            )

        return 0

    def compress_modules(self):
        """
        Quantize modules which have been calibrated
        """
        keys_list = list(self._num_samples.keys())

        mapped_lists = {
            'names': {},
            'modules': {}
        }
        for module in keys_list:
            name = self._module_names[module]

            postfixes = []
            postfix = name.split('.')[-1]
            postfixes = self._next_strat(postfix)

            if len(postfixes) == 0:
                mapped_lists['names'][postfix] = []
                mapped_lists['modules'][postfix] = []

            for postfix in postfixes:
                if postfix not in mapped_lists['names']:  # Check for postfix key, not name
                    mapped_lists['names'][postfix] = []
                    mapped_lists['modules'][postfix] = []
                mapped_lists['names'][postfix].append(name)
                mapped_lists['modules'][postfix].append(module)

        prev_loss = None

        if self.lam_optimize:
            self._lam_optimizer.zero_grad()

        for i, module in enumerate(list(self._num_samples.keys())):
            name = self._module_names[module]

            num_samples = self._num_samples[module]
            quant_args = getattr_chain(module, "quantization_scheme.weights")

            # Get k modules starting from position i+1
            next_modules = []

            postfix = name.split('.')[-1]

            name_list = mapped_lists['names'][postfix]
            module_list = mapped_lists['modules'][postfix]

            if len(name_list) > 0:
                start_idx = name_list.index(name) + 1
                end_idx = start_idx + self.k_next

                for j in range(start_idx, end_idx):
                    if j < len(module_list):
                        next_modules.append(module_list[j])
                    else:
                        next_modules.append(None)

            # log out-of-optimization params
            hessian_trace = self._eigens[module]['hessian_trace']
            eigenvalues_max = self._eigens[module]['eigenvalues_max']
            if hessian_trace is not None:
                self._log_writer.add_scalar(f'hessian_trace/module={name}', hessian_trace.item(), self._step_num_no_optimize)
            if eigenvalues_max is not None:
                self._log_writer.add_scalar(f'eigenvalue_max/module={name}', eigenvalues_max.item(), self._step_num_no_optimize)

            if self.lam_optimize:

                # log optimization params
                logger.info(f"Optimiting lam using {self.opt_steps_num} iterations")

                if self.lam_optimize_method == 'multistep':
                    prev_loss = self._update_lam_param_multistep(
                        lam_loss=self._lam_loss,
                        module=module,
                        next_modules=next_modules,
                    )
                elif self.lam_optimize_method == 'onestep':
                    prev_loss = self._update_lam_param_onestep(
                        module=module,
                        next_modules=next_modules,
                    )
                else:
                    raise ValueError(f"Invalid lam optimize method: {self.lam_optimize_method}")

            logger.info(f"Quantizing {name} using {num_samples} samples")
            with torch.no_grad(), align_module_device(
                module
            ), self._maybe_onload_hessian(module), CompressionLogger(
                module
            ) as comp_logger:

                if self.do_hessian_plot:
                    H_cal = self._hessians[module].clone()
                    _, _, save_path_fp = compute_hessian_metrics(
                        module, f"{name}_fp", H_cal=H_cal, save_dir=self._hessian_log_dir
                    )

                loss, quantized_weight, W_adj, scale, zero_point, g_idx, updated_hessian = quantize_weight(
                    module=module,
                    quant_args=quant_args,
                    hessians_dict=self._hessians,
                    blocksize=self.block_size,
                    percdamp=self.dampening_frac,
                    next_modules=next_modules,
                    lam_tensor=self._lam_tensor,
                    kernel_mode=self.kernel_mode
                )

                if self.do_hessian_plot:
                    logger.info(f"Plotting hessian for {name}")
                    save_path_q, save_path_adj = pushed_l2_hessian_eigentrace_after_gptq(
                        module=module,
                        H_cal=H_cal,
                        W_adj=W_adj,
                        quantized_weight=quantized_weight,
                        scale=scale,
                        zero_point=zero_point,
                        quant_args=quant_args,
                        g_idx=g_idx,
                        name=name,
                        save_dir=self._hessian_log_dir,
                    )

                    plot_eigenvalue_list([save_path_fp, save_path_q, save_path_adj], trunc_low = 40, trunc_high = 50)

                comp_logger.set_loss(loss.item())

            del self._hessians[module]

            update_offload_parameter(module, "weight", quantized_weight)
            update_offload_parameter(module, "weight_scale", scale)
            update_offload_parameter(module, "weight_zero_point", zero_point)
            if g_idx is not None:
                update_offload_parameter(module, "weight_g_idx", g_idx)

            # self._hessians[module] already deleted by quantize_weight
            del self._num_samples[module]

        self._step_num_no_optimize += 1

    def on_end(self, state: State, event: Event, **kwargs):
        """
        Finish calibrating by removing observers and calibration hooks
        """
        self.ended_ = True
        QuantizationMixin.end_calibration(self, state.model)
        self.remove_hooks()  # remove gptq hooks
        self._log_writer.close()

    def on_finalize(self, state: State, **kwargs) -> bool:
        """
        disable the quantization observers used by the OBCQ algorithm

        :param state: session state storing input model and calibration data
        """
        if not self.ended_:
            self.on_end(state, None)

        if len(self._num_samples) > 0:
            raise ValueError(f"Failed to compress {len(self._num_samples)} modules")

        self._hessians = dict()
        self._num_samples = dict()

        return True

    @contextlib.contextmanager
    def _maybe_onload_hessian(self, module: torch.nn.Module):
        if self.offload_hessians:
            device = get_execution_device(module)
            self._hessians[module] = self._hessians[module].to(device=device)

        yield

        if self.offload_hessians:
            if module in self._hessians:  # may have been deleted in context
                self._hessians[module] = self._hessians[module].to(device="cpu")