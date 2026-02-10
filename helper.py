"""
Helper module for VQ-VAE-2 implementation.
Provides base class for easier module initialization.
"""

import torch
import torch.nn as nn


class HelperModule(nn.Module):
    """
    Base class that allows defining module structure in a `build` method
    instead of `__init__`. This makes subclassing cleaner.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.build(*args, **kwargs)
    
    def build(self, *args, **kwargs):
        """Override this method to define the module structure."""
        raise NotImplementedError("Subclasses must implement the build method")


def get_parameter_count(model: nn.Module) -> int:
    """
    Count the total number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
