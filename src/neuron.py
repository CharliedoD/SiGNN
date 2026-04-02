"""Spiking neuron models and surrogate gradient functions.

This module implements:
- Various surrogate gradient functions for training SNNs (SuperSpike, MultiGauss,
  Triangle, Arctan, Sigmoid).
- The BLIF (Bidirectional Leaky Integrate-and-Fire) neuron model.
"""

from math import pi

import torch
import torch.nn as nn

# Global constants for neuron dynamics
GAMMA = 0.2
THRESH_DECAY = 0.7


def reset_net(net: nn.Module):
    """Reset all neuron states in a network.

    Parameters
    ----------
    net : nn.Module
        The network whose neuron states should be reset.
    """
    for m in net.modules():
        if hasattr(m, "reset"):
            m.reset()


def heaviside(x: torch.Tensor):
    """Heaviside step function."""
    return x.ge(0).float()


def gaussian(x, mu, sigma):
    """Gaussian PDF with broadcasting.

    Parameters
    ----------
    x : torch.Tensor
        Input values.
    mu : torch.Tensor
        Mean of the Gaussian.
    sigma : torch.Tensor
        Standard deviation of the Gaussian.
    """
    return (
        torch.exp(-((x - mu) ** 2) / (2 * sigma * sigma))
        / (sigma * torch.sqrt(2 * torch.tensor(pi)))
    )


# ---------------------------------------------------------------------------
# Surrogate gradient spike functions
# ---------------------------------------------------------------------------

class BaseSpike(torch.autograd.Function):
    """Baseline spiking function."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class SuperSpike(BaseSpike):
    """Spike function with SuperSpike surrogate gradient.

    Reference: "SuperSpike: Supervised Learning in Multilayer Spiking Neural
    Networks", Zenke et al. 2018.
    """

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        sg = 1 / (1 + alpha * x.abs()) ** 2
        return grad_input * sg, None


class MultiGaussSpike(BaseSpike):
    """Spike function with multi-Gaussian surrogate gradient.

    Reference: "Accurate and efficient time-domain classification...",
    Yin et al. 2021.
    """

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        zero = torch.tensor(0.0)
        sg = (
            1.15 * gaussian(x, zero, alpha)
            - 0.15 * gaussian(x, alpha, 6 * alpha)
            - 0.15 * gaussian(x, -alpha, 6 * alpha)
        )
        return grad_input * sg, None


class TriangleSpike(BaseSpike):
    """Spike function with triangular surrogate gradient (Bellec et al. 2020)."""

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        sg = torch.nn.functional.relu(1 - alpha * x.abs())
        return grad_input * sg, None


class ArctanSpike(BaseSpike):
    """Spike function with arctan surrogate gradient (Fang et al. 2020/2021)."""

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        sg = 1 / (1 + alpha * x * x)
        return grad_input * sg, None


class SigmoidSpike(BaseSpike):
    """Spike function with sigmoid surrogate gradient."""

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        sgax = (x * alpha).sigmoid_()
        sg = (1.0 - sgax) * sgax * alpha
        return grad_input * sg, None


# ---------------------------------------------------------------------------
# Convenient spike function wrappers
# ---------------------------------------------------------------------------

def superspike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(10.0)):
    return SuperSpike.apply(x - thresh, alpha)


def mgspike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(0.5)):
    return MultiGaussSpike.apply(x - thresh, alpha)


def sigmoidspike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(1.0)):
    return SigmoidSpike.apply(x - thresh, alpha)


def trianglespike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(1.0)):
    return TriangleSpike.apply(x - thresh, alpha)


def arctanspike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(10.0)):
    return ArctanSpike.apply(x - thresh, alpha)


SURROGATE = {
    "sigmoid": sigmoidspike,
    "triangle": trianglespike,
    "arctan": arctanspike,
    "mg": mgspike,
    "super": superspike,
}


# ---------------------------------------------------------------------------
# Neuron models
# ---------------------------------------------------------------------------

class BLIF(nn.Module):
    """Bidirectional Leaky Integrate-and-Fire (BLIF) neuron.

    Parameters
    ----------
    hid : int
        Hidden dimension size (number of neurons).
    v_threshold : float
        Initial membrane voltage threshold.
    v_reset : float
        Reset voltage after spike.
    alpha : float
        Surrogate gradient smoothing factor.
    surrogate : str
        Name of the surrogate gradient function.
    intensity : int
        Spike intensity (reserved for future use).
    """

    def __init__(self, hid=128, v_threshold=1.0, v_reset=0.0, alpha=1.0,
                 surrogate="triangle", intensity=1):
        super().__init__()
        self.register_parameter(
            "v_threshold",
            nn.Parameter(torch.as_tensor(v_threshold, dtype=torch.float32)),
        )
        self.v_th = v_threshold
        self.v_th_in = v_threshold
        self.surrogate = SURROGATE.get(surrogate)
        self.v_reset = v_reset
        self.tau = nn.Parameter(torch.full((hid,), 1.0))
        self.dec = nn.Parameter(torch.full((hid,), 1.0))
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        self.act = nn.Sigmoid()
        self.reset()

    def reset(self):
        """Reset neuron membrane potential and threshold to initial values."""
        self.v = 0.0
        self.v_th = self.v_threshold.item()

    def forward(self, dv):
        """Forward pass: charge -> fire -> reset -> threshold update.

        Parameters
        ----------
        dv : torch.Tensor
            Input voltage change, shape ``(batch_size, hid)``.

        Returns
        -------
        torch.Tensor
            Binary spike tensor.
        """
        tu = self.act(self.tau.unsqueeze(0).expand(dv.shape[0], -1))
        dec = self.act(self.dec.unsqueeze(0).expand(dv.shape[0], -1))

        # 1. Charge
        self.v = self.v - (self.v - self.v_reset) * tu + (1 - tu) * dv

        # 2. Bidirectional fire
        spike = self.surrogate(self.v, self.v_th, self.alpha) + self.surrogate(
            -self.v_th, self.v, self.alpha
        )

        # 3. Reset
        self.v = (1 - spike) * self.v + spike * self.v_reset

        # 4. Threshold update
        self.v_th = self.v_th * dec + (1 - dec) * spike

        return spike
