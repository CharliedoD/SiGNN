from math import pi
import torch
import torch.nn as nn
import torch.nn.functional as F



gamma = 0.2
thresh_decay = 0.7



def reset_net(net: nn.Module):
    for m in net.modules():
        if hasattr(m, 'reset'):
            m.reset()


def heaviside(x: torch.Tensor):
    return x.ge()


def gaussian(x, mu, sigma):
    """
    Gaussian PDF with broadcasting.
    """
    return torch.exp(-((x - mu) * (x - mu)) / (2 * sigma * sigma)) / (sigma * torch.sqrt(2 * torch.tensor(pi)))


class BaseSpike(torch.autograd.Function):
    """
    Baseline spiking function.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class SuperSpike(BaseSpike):
    """
    Spike function with SuperSpike surrogate gradient from
    "SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks", Zenke et al. 2018.

    Design choices:
    - Height of 1 ("The Remarkable Robustness of Surrogate Gradient...", Zenke et al. 2021)
    - alpha scaled by 10 ("Training Deep Spiking Neural Networks", Ledinauskas et al. 2020)
    """

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        sg = 1 / (1 + alpha * x.abs()) ** 2
        return grad_input * sg, None


class MultiGaussSpike(BaseSpike):
    """
    Spike function with multi-Gaussian surrogate gradient from
    "Accurate and efficient time-domain classification...", Yin et al. 2021.

    Design choices:
    - Hyperparameters determined through grid search (Yin et al. 2021)
    """

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        zero = torch.tensor(0.0)  # no need to specify device for 0-d tensors
        sg = (
                1.15 * gaussian(x, zero, alpha)
                - 0.15 * gaussian(x, alpha, 6 * alpha)
                - 0.15 * gaussian(x, -alpha, 6 * alpha)
        )
        return grad_input * sg, None


class TriangleSpike(BaseSpike):
    """
    Spike function with triangular surrogate gradient
    as in Bellec et al. 2020.
    """

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        sg = torch.nn.functional.relu(1 - alpha * x.abs())
        return grad_input * sg, None


class ArctanSpike(BaseSpike):
    """
    Spike function with derivative of arctan surrogate gradient.
    Featured in Fang et al. 2020/2021.
    """

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        sg = 1 / (1 + alpha * x * x)
        return grad_input * sg, None


class SigmoidSpike(BaseSpike):

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        sgax = (x * alpha).sigmoid_()
        sg = (1. - sgax) * sgax * alpha
        return grad_input * sg, None


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


SURROGATE = {'sigmoid': sigmoidspike, 'triangle': trianglespike, 'arctan': arctanspike,
             'mg': mgspike, 'super': superspike}



    
    
class BLIF(nn.Module):
    def __init__(self, hid=128, v_threshold=1.0, v_reset=0., alpha=1.0,
                 surrogate='triangle', intensity=1):
        super().__init__()
        self.register_parameter("v_threshold", nn.Parameter(
            torch.as_tensor(v_threshold, dtype=torch.float32)))
        self.v_th = v_threshold
        self.v_th_in = v_threshold
        self.surrogate = SURROGATE.get(surrogate)
        self.v_reset = v_reset
        self.tau = torch.nn.Parameter(torch.full((hid,), 1.0))
        self.dec = torch.nn.Parameter(torch.full((hid,), 1.0))
        self.register_buffer("alpha", torch.as_tensor(
            alpha, dtype=torch.float32))
        self.act = nn.Sigmoid()
        self.reset()

    def reset(self):
        self.v = 0.
        self.v_th = self.v_threshold.item()
        
    def forward(self, dv):
        tu = self.act(self.tau.unsqueeze(0).expand(dv.shape[0], -1))
        dec = self.act(self.dec.unsqueeze(0).expand(dv.shape[0], -1))
        # 1. charge
        self.v = self.v - (self.v - self.v_reset) * tu + (1 - tu) * dv
        
        # 2. bidirectional fire
        spike = self.surrogate(self.v, self.v_th, self.alpha) + self.surrogate(-self.v_th, self.v, self.alpha)  
        
        # 3. reset
        self.v = (1 - spike) * self.v + spike * self.v_reset
        
        # 4. threhold updates
        self.v_th = self.v_th * dec +  (1 - dec) * spike

        return spike
