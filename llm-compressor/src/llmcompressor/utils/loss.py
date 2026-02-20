import torch
import torch.nn as nn
from llmcompressor.modifiers.utils.kernels import apply_conv
from abc import abstractmethod

__all__ = ["HessianLossNormed", "HessianLoss", "HessianLossNormCos", "HessianLossSoftCos", "HessianLossTrace"]

class BasicLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.with_eigens = True

    def get_with_eigens(self):
        return self.with_eigens

    @abstractmethod
    def forward(
            self,
            lam,
            module,
            next_modules,
            hessians,
            eigens,
            kernel_mode,
            eps=1e-8,
            proj_dim=10,
            reg_coef=1e-3,
            neg_weight=0.1):
        pass

class HessianLoss(BasicLoss):
    def __init__(self):
        super().__init__()
        self.with_eigens = True

    def forward(
            self,
            lam,
            module,
            next_modules,
            hessians,
            eigens,
            kernel_mode,
            eps=1e-8,
            proj_dim=10,
            reg_coef=1e-3,
            neg_weight=0.1):
        try:
            H = hessians[module]
            H_next = hessians[next_modules[0]]
            h = H + lam * apply_conv(H_next, mode=kernel_mode)
            h = (h + h.T) / 2
        except RuntimeError:
            return torch.tensor(0.0, dtype=torch.float32, device=H.device, requires_grad=True), None
        
        eigenvalues = torch.linalg.eigvalsh(h)
        sorted_eigenvalues = torch.sort(eigenvalues, descending=True)[0]
        sorted_eigenval_size = sorted_eigenvalues.shape[0]

        if sorted_eigenval_size >= 2:
            return sorted_eigenvalues[0] * sorted_eigenvalues[1], sorted_eigenvalues
        else:
            return torch.tensor(0.0, dtype=torch.float32, device=H.device, requires_grad=True), sorted_eigenvalues

class HessianLossNormed(BasicLoss):
    def __init__(self):
        super().__init__()
        self.with_eigens = True

    def forward(
            self,
            lam,
            module,
            next_modules,
            hessians,
            eigens,
            kernel_mode,
            eps=1e-8,
            proj_dim=10,
            reg_coef=1e-3,
            neg_weight=0.1):
        try:
            H = hessians[module]
            H_next = hessians[next_modules[0]]
            h = H + lam * apply_conv(H_next, mode=kernel_mode)
            h = (h + h.T) / 2
        except RuntimeError:
            return torch.tensor(0.0, dtype=torch.float32, device=H.device, requires_grad=True), None

        eigenvalues = torch.linalg.eigvalsh(h)
        sorted_eigenvalues = torch.sort(eigenvalues, descending=True)[0]
        sorted_eigenval_size = sorted_eigenvalues.shape[0]

        if sorted_eigenval_size >= 2:
            norm_eigens = torch.sum(torch.abs(eigenvalues))
            return sorted_eigenvalues[0] * sorted_eigenvalues[1] / (norm_eigens * norm_eigens), sorted_eigenvalues
        else:
            return torch.tensor(0.0, dtype=torch.float32, device=H.device, requires_grad=True), sorted_eigenvalues


class HessianLossNormCos(BasicLoss):
    def __init__(self):
        super().__init__()
        self.with_eigens = True

    def forward(
            self,
            lam,
            module,
            next_modules,
            hessians,
            eigens,
            kernel_mode,
            eps=1e-8,
            proj_dim=10,
            reg_coef=1e-3,
            neg_weight=0.1):
        try:
            H = hessians[module]
            H_next = hessians[next_modules[0]]
            h = H + lam * apply_conv(H_next, mode=kernel_mode)
            h = (h + h.T) / 2
            
            # Add regularization to diagonal
            h = h + 1e-6 * torch.eye(h.shape[0], device=h.device, dtype=h.dtype)
            
        except RuntimeError:
            return torch.tensor(0.0, dtype=torch.float32, device=H.device, requires_grad=True), None
        
        try:
            # Use torch.linalg.eig instead of eigh (handles ill-conditioned matrices better)
            eigenvalues, eigenvectors = torch.linalg.eig(h)
            
            # Convert to real (since h is symmetric, eigenvalues should be real)
            eigenvalues = eigenvalues.real
            eigenvectors = eigenvectors.real

            # Sort by absolute value
            sorted_indices = torch.argsort(torch.abs(eigenvalues), descending=True)
            sorted_eigenvalues = eigenvalues[sorted_indices]
            sorted_eigenvectors = eigenvectors[:, sorted_indices]
            
            if sorted_eigenvalues.shape[0] >= 2:
                norm_eigens = torch.sum(torch.abs(sorted_eigenvalues))
                
                eigenvec1 = sorted_eigenvectors[:, 0]
                eigenvec2 = sorted_eigenvectors[:, 1]
                
                # Normalize eigenvectors
                eigenvec1 = eigenvec1 / (torch.norm(eigenvec1) + 1e-8)
                eigenvec2 = eigenvec2 / (torch.norm(eigenvec2) + 1e-8)
                
                cos_sim = torch.abs(torch.dot(eigenvec1, eigenvec2))
                
                loss = sorted_eigenvalues[0] * sorted_eigenvalues[1] * (1 - cos_sim) / (norm_eigens * norm_eigens + 1e-12)
                return loss, sorted_eigenvalues
            else:
                return torch.tensor(0.0, dtype=torch.float32, device=H.device, requires_grad=True), sorted_eigenvalues

        except RuntimeError:
            # Return small loss if eigendecomposition still fails
            return torch.tensor(0.01, dtype=torch.float32, device=H.device, requires_grad=True), sorted_eigenvalues

class HessianLossSoftCos(BasicLoss):
    def __init__(self):
        super().__init__()
        self.with_eigens = True

    def forward(
            self,
            lam,
            module,
            next_modules,
            hessians,
            eigens,
            kernel_mode,
            eps=1e-8,
            proj_dim=10,
            reg_coef=1e-3,
            neg_weight=0.1):

        H = hessians[module]

        dummy_loss = torch.tensor(0.0, dtype=torch.float32, device=H.device, requires_grad=True)

        try:
            eigenvalues = eigens[module]['eigenvalues']
            eigenvectors = eigens[module]['eigenvectors']

            is_lam = False

            for i, module_next in enumerate(next_modules):
                if module_next is not None:
                    if (eigenvalues.shape == eigens[module_next]['eigenvalues'].shape and 
                        eigenvectors.shape == eigens[module_next]['eigenvectors'].shape):

                        is_lam = True
                        eigenvalues += lam[i] * eigens[module_next]['eigenvalues']
                        eigenvectors += lam[i] * eigens[module_next]['eigenvectors']

            if not is_lam:
                return dummy_loss, None

        except RuntimeError:
            return dummy_loss, None

        sorted_indices = torch.argsort(torch.abs(eigenvalues), descending=True)
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_indices = sorted_indices[:min(50, eigenvalues.shape[0])]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        if eigenvalues.shape[0] < 2:
            return dummy_loss, sorted_eigenvalues

        mask_pos = eigenvalues > 0
        mask_neg = eigenvalues < 0
        
        # Если отрицательных собственных значений нет, создаем искусственный вектор
        if not torch.any(mask_neg):
            # Берем наименьшее положительное собственное значение и делаем его "псевдо-отрицательным"
            min_pos_idx = torch.argmin(eigenvalues[mask_pos])
            neg_eigenvalues = -torch.abs(eigenvalues[mask_pos][min_pos_idx:min_pos_idx+1]) * neg_weight
            neg_eigenvectors = eigenvectors[:, mask_pos][:, min_pos_idx:min_pos_idx+1]
            
            pos_eigenvalues = eigenvalues[mask_pos]
            pos_eigenvectors = eigenvectors[:, mask_pos]
        elif not torch.any(mask_pos):
            # Если нет положительных, создаем "псевдо-положительный" вектор
            max_neg_idx = torch.argmax(eigenvalues[mask_neg])  # наименее отрицательный
            pos_eigenvalues = torch.abs(eigenvalues[mask_neg][max_neg_idx:max_neg_idx+1]) * neg_weight
            pos_eigenvectors = eigenvectors[:, mask_neg][:, max_neg_idx:max_neg_idx+1]
            
            neg_eigenvalues = eigenvalues[mask_neg]
            neg_eigenvectors = eigenvectors[:, mask_neg]
        else:
            # Обычный случай: есть и положительные, и отрицательные
            pos_eigenvalues = eigenvalues[mask_pos]
            pos_eigenvectors = eigenvectors[:, mask_pos]
            neg_eigenvalues = eigenvalues[mask_neg]
            neg_eigenvectors = eigenvectors[:, mask_neg]

        # Нормализуем веса для баланса
        if torch.sum(torch.abs(pos_eigenvalues)) > 0:
            pos_weights = pos_eigenvalues / torch.sum(torch.abs(pos_eigenvalues))
        else:
            pos_weights = torch.ones_like(pos_eigenvalues) / len(pos_eigenvalues)
            
        if torch.sum(torch.abs(neg_eigenvalues)) > 0:
            neg_weights = neg_eigenvalues / torch.sum(torch.abs(neg_eigenvalues))
        else:
            neg_weights = torch.ones_like(neg_eigenvalues) / len(neg_eigenvalues)
        
        # Усиливаем отрицательные веса, если они слишком малы
        neg_scale = 1.0
        if torch.mean(torch.abs(neg_eigenvalues)) < 0.1 * torch.mean(torch.abs(pos_eigenvalues)):
            neg_scale = torch.mean(torch.abs(pos_eigenvalues)) / (torch.mean(torch.abs(neg_eigenvalues)) + eps)
            neg_scale = torch.clamp(neg_scale, 1.0, 10.0)
        
        pos_vector = torch.sum(pos_weights.unsqueeze(0) * pos_eigenvectors, dim=1)
        neg_vector = torch.sum(neg_weights.unsqueeze(0) * neg_eigenvectors * neg_scale, dim=1)

        # Проекция на общее подпространство
        k = min(proj_dim, eigenvectors.shape[1])
        common_basis = eigenvectors[:, :k]
        
        pos_proj = common_basis.T @ pos_vector.unsqueeze(1)
        neg_proj = common_basis.T @ neg_vector.unsqueeze(1)
        
        pos_vector = (common_basis @ pos_proj).squeeze()
        neg_vector = (common_basis @ neg_proj).squeeze()

        # Гарантируем минимальные нормы
        pos_norm = torch.norm(pos_vector) + eps
        neg_norm = torch.norm(neg_vector) + eps
        
        # Если одна из норм все еще слишком мала, добавляем шум
        if pos_norm < eps * 10:
            pos_vector = pos_vector + torch.randn_like(pos_vector) * eps
            pos_norm = torch.norm(pos_vector)
        if neg_norm < eps * 10:
            neg_vector = neg_vector + torch.randn_like(neg_vector) * eps
            neg_norm = torch.norm(neg_vector)

        cos_sim = torch.abs(torch.dot(pos_vector, neg_vector) / (pos_norm * neg_norm))
        
        # Добавляем штраф за дисбаланс собственных значений
        balance_penalty = torch.abs(torch.mean(torch.abs(pos_eigenvalues)) - torch.mean(torch.abs(neg_eigenvalues))) / \
                         (torch.mean(torch.abs(eigenvalues)) + eps)

        loss = cos_sim - 0.05 * balance_penalty
        
        return loss, sorted_eigenvalues


class HessianLossTrace(BasicLoss):
    def __init__(self):
        super().__init__()
        self.with_eigens = False

    def forward(
            self,
            lam,
            module,
            next_modules,
            hessians,
            eigens,
            kernel_mode,
            eps=1e-8,
            proj_dim=10,
            reg_coef=1e-3,
            neg_weight=0.1):

        trace = eigens[module]['hessian_trace']
        H = hessians[module]
        dummy_loss = torch.tensor(0.0, dtype=torch.float32, device=H.device, requires_grad=True)
        max_eigen_val = eigens[module]['eigenvalues_max']

        # try:
        is_lam = False

        for i, module_next in enumerate(next_modules):
            if module_next is not None:
                is_lam = True
                trace += lam[i] * eigens[module_next]['hessian_trace']
                max_eigen_val = max_eigen_val + lam[i] * eigens[module_next]['eigenvalues_max']

        if not is_lam:
            return dummy_loss, None

        loss = (trace + max_eigen_val) / 1000.
        return loss, None

class HessianLossTraceScaled(BasicLoss):
    def __init__(self):
        super().__init__()
        self.with_eigens = False

    def forward(
            self,
            lam,
            module,
            next_modules,
            hessians,
            eigens,
            kernel_mode,
            eps=1e-8,
            proj_dim=10,
            reg_coef=1e-3,
            neg_weight=0.1):

        trace = eigens[module]['hessian_trace']
        H = hessians[module]
        dummy_loss = torch.tensor(0.0, dtype=torch.float32, device=H.device, requires_grad=True)
        max_eigen_val = eigens[module]['eigenvalues_max']

        # try:
        is_lam = False

        for i, module_next in enumerate(next_modules):
            if module_next is not None:
                is_lam = True
                trace += lam[i] * eigens[module_next]['hessian_trace']
                max_eigen_val = max_eigen_val + lam[i] * eigens[module_next]['eigenvalues_max']

        if not is_lam:
            return dummy_loss, None
        
        scale = 100 * (H.shape[0] + H.shape[1]) / 2

        loss = (trace + max_eigen_val) / scale
        return loss, None


class HessianLossTraceOnlyScaled(BasicLoss):
    def __init__(self):
        super().__init__()
        self.with_eigens = False

    def forward(
            self,
            lam,
            module,
            next_modules,
            hessians,
            eigens,
            kernel_mode,
            eps=1e-8,
            proj_dim=10,
            reg_coef=1e-3,
            neg_weight=0.1):

        trace = eigens[module]['hessian_trace']
        H = hessians[module]
        dummy_loss = torch.tensor(0.0, dtype=torch.float32, device=H.device, requires_grad=True)

        # try:
        is_lam = False

        for i, module_next in enumerate(next_modules):
            if module_next is not None:
                is_lam = True
                trace += lam[i] * eigens[module_next]['hessian_trace']

        if not is_lam:
            return dummy_loss, None
        
        scale = 100 * (H.shape[0] + H.shape[1]) / 2

        loss = (trace) / scale
        return loss, None

class HessianLossEigenMaxOnlyScaled(BasicLoss):
    def __init__(self):
        super().__init__()
        self.with_eigens = False

    def forward(
            self,
            lam,
            module,
            next_modules,
            hessians,
            eigens,
            kernel_mode,
            eps=1e-8,
            proj_dim=10,
            reg_coef=1e-3,
            neg_weight=0.1):

        H = hessians[module]
        dummy_loss = torch.tensor(0.0, dtype=torch.float32, device=H.device, requires_grad=True)
        max_eigen_val = eigens[module]['eigenvalues_max']

        # try:
        is_lam = False

        for i, module_next in enumerate(next_modules):
            if module_next is not None:
                is_lam = True
                max_eigen_val = max_eigen_val + lam[i] * eigens[module_next]['eigenvalues_max']

        if not is_lam:
            return dummy_loss, None
        
        scale = 100 * (H.shape[0] + H.shape[1]) / 2

        loss = max_eigen_val / scale
        return loss, None


class MSEPowerLawLoss(BasicLoss):
    def __init__(self):
        super().__init__()
        self.with_eigens = True
        
    def forward(
        self,
        lam,
        module,
        next_modules,
        hessians,
        eigens,
        kernel_mode,
        eps=1e-6
    ):
        H = hessians[module]
        dummy_loss = torch.tensor(0.0, dtype=torch.float32, device=H.device, requires_grad=True)

        try:
            # Get eigenvalues
            eigenvalues = eigens[module]['eigenvalues']
            is_lam = False
            
            # Combine with next modules
            for i, module_next in enumerate(next_modules):
                if module_next is not None:
                    if eigenvalues.shape == eigens[module_next]['eigenvalues'].shape:
                        is_lam = True
                        eigenvalues = eigenvalues + lam[i] * eigens[module_next]['eigenvalues']

            if not is_lam:
                return dummy_loss, None

        except RuntimeError:
            return dummy_loss, None

        # Take absolute values and sort
        abs_evals = torch.abs(eigenvalues)
        abs_evals, _ = torch.sort(abs_evals, descending=True)
        
        # Filter small values
        mask = abs_evals > eps
        if torch.sum(mask) < 3:
            return dummy_loss, abs_evals

        abs_evals = abs_evals[mask]
        
        # Estimate power law coefficient using log-log linear regression
        k = torch.arange(1, len(abs_evals) + 1, dtype=torch.float32, device=abs_evals.device)
        
        # Log transform
        log_k = torch.log(k)
        log_evals = torch.log(abs_evals)

        # Linear regression: log(λk) = log(λ1) - s * log(k)
        # Solve for s using least squares
        A = torch.stack([torch.ones_like(log_k), -log_k], dim=1)
        # solution = torch.linalg.lstsq(A, log_evals.unsqueeze(1)).solution

        ATA = A.T @ A  # [2, 2]
        ATb = A.T @ log_evals.unsqueeze(1)  # [2, 1]

        # Add small diagonal for numerical stability
        ATA = ATA + torch.eye(2, device=ATA.device) * eps

        # Solve using Cholesky or LU decomposition (differentiable)
        solution = torch.linalg.solve(ATA, ATb)

        log_lambda_1 = solution[0, 0]
        s = solution[1, 0]  # This is our power law coefficient
        
        # Generate estimated eigenvalues
        estimated_evals = torch.exp(log_lambda_1) * torch.pow(k, -s)
        
        # MSE loss
        loss = torch.nn.functional.mse_loss(estimated_evals, abs_evals) / len(abs_evals)
    
        return loss, (abs_evals, estimated_evals)


class ElboPowerLawLoss(BasicLoss):
    def __init__(self):
        super().__init__()
        self.with_eigens = True
        
    def forward(
        self,
        lam,
        module,
        next_modules,
        hessians,
        eigens,
        kernel_mode,
        eps=1e-6
    ):
        H = hessians[module]
        dummy_loss = torch.tensor(0.0, dtype=torch.float32, device=H.device, requires_grad=True)

        try:
            # Get eigenvalues
            eigenvalues = eigens[module]['eigenvalues']
            is_lam = False
            
            # Combine with next modules
            for i, module_next in enumerate(next_modules):
                if module_next is not None:
                    if eigenvalues.shape == eigens[module_next]['eigenvalues'].shape:
                        is_lam = True
                        eigenvalues = eigenvalues + lam[i] * eigens[module_next]['eigenvalues']

            if not is_lam:
                return dummy_loss, None

        except RuntimeError:
            return dummy_loss, None

        # Take absolute values and sort
        abs_evals = torch.abs(eigenvalues)
        abs_evals, _ = torch.sort(abs_evals, descending=True)
        
        # Filter small values
        mask = abs_evals > eps

        abs_evals = abs_evals[mask]
        
        # Estimate power law coefficient using log-log linear regression
        k = torch.arange(1, len(abs_evals) + 1, dtype=torch.float32, device=abs_evals.device)
        
        # Log transform
        log_k = torch.log(k)
        log_evals = torch.log(abs_evals)

        # Linear regression: log(λk) = log(λ1) - s * log(k)
        # Solve for s using least squares
        A = torch.stack([torch.ones_like(log_k), -log_k], dim=1)
        # solution = torch.linalg.lstsq(A, log_evals.unsqueeze(1)).solution

        ATA = A.T @ A  # [2, 2]
        ATb = A.T @ log_evals.unsqueeze(1)  # [2, 1]

        # Add small diagonal for numerical stability
        ATA = ATA + torch.eye(2, device=ATA.device) * eps

        # Solve using Cholesky or LU decomposition (differentiable)
        solution = torch.linalg.solve(ATA, ATb)

        log_lambda_1 = solution[0, 0]
        s = solution[1, 0]  # This is our power law coefficient
        
        # Generate estimated eigenvalues
        estimated_evals = torch.exp(log_lambda_1) * torch.pow(k, -s)
        
        # ELBO loss (Evidence Lower Bound)
        # Assuming Gaussian likelihood and prior
        # ELBO = E[log p(x|z)] - KL[q(z|x) || p(z)]
        
        # Likelihood term (negative MSE as log likelihood)
        log_likelihood = -torch.nn.functional.mse_loss(estimated_evals, abs_evals, reduction='sum')

        # KL divergence term (simplified, assuming Gaussian prior)
        # Prior: N(0, 1) for parameters, Posterior: N(s, sigma^2)
        sigma_s = torch.std(estimated_evals - abs_evals)
        kl_divergence = 0.5 * torch.sum(1 + 2 * torch.log(sigma_s) - sigma_s**2)

        # ELBO loss (negative ELBO for minimization)
        loss = -(log_likelihood - kl_divergence) / len(abs_evals)
        
        return loss, (abs_evals, estimated_evals)


class HessianLossCombined(BasicLoss):
    def __init__(self):
        super().__init__()
        self.with_eigens = True

    def forward(
            self,
            lam,
            module,
            next_modules,
            hessians,
            eigens,
            kernel_mode,
            eps=1e-8,
            proj_dim=10,
            reg_coef=1e-3,
            neg_weight=0.1):

        H = hessians[module]

        dummy_loss = torch.tensor(0.0, dtype=torch.float32, device=H.device, requires_grad=True)

        try:
            eigenvalues = eigens[module]['eigenvalues']
            eigenvectors = eigens[module]['eigenvectors']

            is_lam = False

            for i, module_next in enumerate(next_modules):
                if module_next is not None:
                    if (eigenvalues.shape == eigens[module_next]['eigenvalues'].shape and 
                        eigenvectors.shape == eigens[module_next]['eigenvectors'].shape):

                        is_lam = True
                        eigenvalues += lam[i] * eigens[module_next]['eigenvalues']
                        eigenvectors += lam[i] * eigens[module_next]['eigenvectors']

            if not is_lam:
                return dummy_loss, None

        except RuntimeError:
            return dummy_loss, None

        sorted_indices = torch.argsort(torch.abs(eigenvalues), descending=True)
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_indices = sorted_indices[:min(50, eigenvalues.shape[0])]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        if eigenvalues.shape[0] < 2:
            return dummy_loss, sorted_eigenvalues

        mask_pos = eigenvalues > 0
        mask_neg = eigenvalues < 0
        
        # Если отрицательных собственных значений нет, создаем искусственный вектор
        if not torch.any(mask_neg):
            # Берем наименьшее положительное собственное значение и делаем его "псевдо-отрицательным"
            min_pos_idx = torch.argmin(eigenvalues[mask_pos])
            neg_eigenvalues = -torch.abs(eigenvalues[mask_pos][min_pos_idx:min_pos_idx+1]) * neg_weight
            neg_eigenvectors = eigenvectors[:, mask_pos][:, min_pos_idx:min_pos_idx+1]
            
            pos_eigenvalues = eigenvalues[mask_pos]
            pos_eigenvectors = eigenvectors[:, mask_pos]
        elif not torch.any(mask_pos):
            # Если нет положительных, создаем "псевдо-положительный" вектор
            max_neg_idx = torch.argmax(eigenvalues[mask_neg])  # наименее отрицательный
            pos_eigenvalues = torch.abs(eigenvalues[mask_neg][max_neg_idx:max_neg_idx+1]) * neg_weight
            pos_eigenvectors = eigenvectors[:, mask_neg][:, max_neg_idx:max_neg_idx+1]
            
            neg_eigenvalues = eigenvalues[mask_neg]
            neg_eigenvectors = eigenvectors[:, mask_neg]
        else:
            # Обычный случай: есть и положительные, и отрицательные
            pos_eigenvalues = eigenvalues[mask_pos]
            pos_eigenvectors = eigenvectors[:, mask_pos]
            neg_eigenvalues = eigenvalues[mask_neg]
            neg_eigenvectors = eigenvectors[:, mask_neg]

        # Нормализуем веса для баланса
        if torch.sum(torch.abs(pos_eigenvalues)) > 0:
            pos_weights = pos_eigenvalues / torch.sum(torch.abs(pos_eigenvalues))
        else:
            pos_weights = torch.ones_like(pos_eigenvalues) / len(pos_eigenvalues)
            
        if torch.sum(torch.abs(neg_eigenvalues)) > 0:
            neg_weights = neg_eigenvalues / torch.sum(torch.abs(neg_eigenvalues))
        else:
            neg_weights = torch.ones_like(neg_eigenvalues) / len(neg_eigenvalues)
        
        # Усиливаем отрицательные веса, если они слишком малы
        neg_scale = 1.0
        if torch.mean(torch.abs(neg_eigenvalues)) < 0.1 * torch.mean(torch.abs(pos_eigenvalues)):
            neg_scale = torch.mean(torch.abs(pos_eigenvalues)) / (torch.mean(torch.abs(neg_eigenvalues)) + eps)
            neg_scale = torch.clamp(neg_scale, 1.0, 10.0)
        
        pos_vector = torch.sum(pos_weights.unsqueeze(0) * pos_eigenvectors, dim=1)
        neg_vector = torch.sum(neg_weights.unsqueeze(0) * neg_eigenvectors * neg_scale, dim=1)

        # Проекция на общее подпространство
        k = min(proj_dim, eigenvectors.shape[1])
        common_basis = eigenvectors[:, :k]
        
        pos_proj = common_basis.T @ pos_vector.unsqueeze(1)
        neg_proj = common_basis.T @ neg_vector.unsqueeze(1)
        
        pos_vector = (common_basis @ pos_proj).squeeze()
        neg_vector = (common_basis @ neg_proj).squeeze()

        # Гарантируем минимальные нормы
        pos_norm = torch.norm(pos_vector) + eps
        neg_norm = torch.norm(neg_vector) + eps
        
        # Если одна из норм все еще слишком мала, добавляем шум
        if pos_norm < eps * 10:
            pos_vector = pos_vector + torch.randn_like(pos_vector) * eps
            pos_norm = torch.norm(pos_vector)
        if neg_norm < eps * 10:
            neg_vector = neg_vector + torch.randn_like(neg_vector) * eps
            neg_norm = torch.norm(neg_vector)

        cos_sim = torch.abs(torch.dot(pos_vector, neg_vector) / (pos_norm * neg_norm))
        
        # Добавляем штраф за дисбаланс собственных значений
        balance_penalty = torch.abs(torch.mean(torch.abs(pos_eigenvalues)) - torch.mean(torch.abs(neg_eigenvalues))) / \
                         (torch.mean(torch.abs(eigenvalues)) + eps)

        loss_softcos = cos_sim - 0.05 * balance_penalty

        max_eigen_val = eigens[module]['eigenvalues_max']

        # try:
        is_lam = False

        for i, module_next in enumerate(next_modules):
            if module_next is not None:
                is_lam = True
                max_eigen_val = max_eigen_val + lam[i] * eigens[module_next]['eigenvalues_max']

        if not is_lam:
            return dummy_loss, None
        
        scale = 100 * (H.shape[0] + H.shape[1]) / 2

        loss_eigenmx = max_eigen_val / scale

        loss = loss_softcos + loss_eigenmx

        return loss, sorted_eigenvalues




LOSS_DICT = {
    'HessianLoss': HessianLoss,
    'HessianLossNormed': HessianLossNormed,
    'HessianLossNormCos': HessianLossNormCos,
    'HessianLossSoftCos': HessianLossSoftCos,
    'HessianLossTrace': HessianLossTrace,
    'HessianLossTraceScaled': HessianLossTraceScaled,
    'HessianLossTraceOnlyScaled': HessianLossTraceOnlyScaled,
    'HessianLossEigenMaxOnlyScaled': HessianLossEigenMaxOnlyScaled,
    'MSEPowerLawLoss': MSEPowerLawLoss,
    'ElboPowerLawLoss': ElboPowerLawLoss,
    'HessianLossCombined': HessianLossCombined,
}
