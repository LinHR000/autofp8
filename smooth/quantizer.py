from abc import ABC, abstractmethod

import torch


class QuantizerMixin(ABC):
    @abstractmethod
    def find_scale_zero(self, w: torch.Tensor):
        """
        Determines scale and zero-point for quantization.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantizes input tensor.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def pseudo_quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies pseudo quantization (forward pass simulation).
        Must be implemented by subclasses.
        """
        pass

    @property
    @abstractmethod
    def n_bit(self):
        """
        Returns the number of bits used for quantization.
        Must be implemented by subclasses.
        """
        pass

    @property
    @abstractmethod
    def maxq(self):
        """
        Returns the maximum quantization level.
        Must be implemented by subclasses.
        """
        pass

    @property
    @abstractmethod
    def scales(self):
        """
        Returns the scale factors used for quantization.
        Must be implemented by subclasses.
        """
        pass

    @property
    @abstractmethod
    def zeros(self):
        """
        Returns the zero points used for quantization.
        Must be implemented by subclasses.
        """
        pass

    @property
    @abstractmethod
    def symm_q(self):
        """
        Indicates whether symmetric quantization is used.
        Must be implemented by subclasses.
        """
        pass


class Quantizer(QuantizerMixin):
    """
    RTN(round-to-nearest) Quantizer for weight quantization.
    This is a simple baseline quantizer.
    """

    def __init__(
        self,
        n_bit: int,
        *,
        per_tensor: bool = False,
        symm_q: bool = False,
        group_size: int = 128,
        zeropoint: bool = True,
        mse: bool = False,
        norm=2.4,
        grid=100,
        max_shrink=0.8,
    ):
        self._n_bit = n_bit
        self._group_size = group_size
        self._symm_q = symm_q
        self._per_tensor = per_tensor
        self._zeropoint = zeropoint

        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.max_shrink = max_shrink

        if not symm_q:
            assert zeropoint, "asymm quantization must have zeropoint"
            self._maxq = 2**n_bit - 1
        else:
            self._maxq = 2 ** (n_bit - 1) - 1

    def _find_scale_zero_per_tensor(self, w: torch.Tensor):
        assert (
            self._symm_q
        ), "per-tensor quantization only support symmetric quantization right now!"
        scales = w.abs().max().float() / self._maxq

        self._scales = scales

    def _find_scale_zero_per_channel(self, w: torch.Tensor):
        # w: c_out, c_in
        org_shape = w.shape
        device = w.device

        if self._group_size > 0:
            assert w.shape[1] % self._group_size == 0
            w = w.view(-1, self._group_size)

        tmp = torch.zeros(w.shape[0], dtype=w.dtype, device=device)
        xmin = torch.minimum(w.amin(1), tmp)
        xmax = torch.maximum(w.amax(1), tmp)

        if self._symm_q:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        zero_range = (xmin == 0) & (xmax == 0)
        xmin[zero_range] = -1
        xmax[zero_range] = +1

        if self._symm_q:
            scales = xmax.float() / self._maxq
            zeros = torch.full_like(scales, self._maxq + 1)
        else:
            scales = (xmax - xmin).clamp(1e-5).float() / self._maxq
            zeros = torch.round(-xmin / scales).clamp(0, self._maxq)

        shape = [org_shape[0]] + [-1]
        self._scales = scales.reshape(shape)
        self._zeros = zeros.reshape(shape)

    def find_scale_zero(self, w):
        if self._per_tensor:
            self._find_scale_zero_per_tensor(w)
        else:
            self._find_scale_zero_per_channel(w)

    @property
    def n_bit(self):
        return self._n_bit

    @property
    def maxq(self):
        return self._maxq

    @property
    def group_size(self):
        return self._group_size

    @property
    def scales(self):
        return self._scales

    @property
    def zeros(self):
        return self._zeros

    @property
    def symm_q(self):
        return self._symm_q

    def quantize(self, x: torch.Tensor):
        if not hasattr(self, "_scales"):
            self.find_scale_zero(x)

        if self._per_tensor:
            return self._quantize_per_tensor(x)
        else:
            return self._quantize_per_channel(x)

    def _quantize_per_tensor(self, x: torch.Tensor):
        min_int = -self._maxq - 1
        max_int = self._maxq
        return torch.clamp(x.div(self._scales).round(), min_int, max_int)

    def _pseudo_quantize_per_tensor(self, x: torch.Tensor):
        q = self._quantize_per_tensor(x)
        return q * self._scales

    def _quantize_per_channel(self, x: torch.Tensor):
        org_shape = x.shape
        if self._group_size > 0:
            assert x.shape[1] % self._group_size == 0
            assert self._scales.shape[1] == x.shape[1] // self._group_size
            x = x.view(-1, self._group_size)
            scales = self._scales.view(-1, 1)
            zeros = self._zeros.view(-1, 1)
        else:
            scales = self._scales
            zeros = self._zeros
        if self._zeropoint:
            max_int = 2 * self._maxq + 1 if self._symm_q else self._maxq
            q = torch.clamp(torch.round(x / scales) + zeros, 0, max_int)
        else:
            q = torch.clamp(torch.round(x / scales), -self._maxq - 1, self._maxq)
        return q.view(org_shape)

    def _pseudo_quantize_per_channel(self, x: torch.Tensor):
        q = self._quantize_per_channel(x)
        org_shape = q.shape
        if self._group_size > 0:
            q = q.view(-1, self._group_size)
            scales = self._scales.view(-1, 1)
            zeros = self._zeros.view(-1, 1)
        else:
            scales = self._scales
            zeros = self._zeros
        dq = scales * (q - zeros) if self._zeropoint else scales * q
        return dq.view(org_shape)

    def pseudo_quantize(self, x: torch.Tensor):
        if not hasattr(self, "_scales"):
            self.find_scale_zero(x)

        if self._per_tensor:
            return self._pseudo_quantize_per_tensor(x)
        else:
            return self._pseudo_quantize_per_channel(x)
