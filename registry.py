
#%% Imports:

import torch.nn as nn

'''
This file contains the code to register PyTorch models for clean build from a configuration file.

'''

#%% Main code:

# Initialization:
MODEL_REGISTRY: dict[str, type[nn.Module]] = {}

def register_model(name: str):
    def decorator(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered")
        MODEL_REGISTRY[name] = cls
        return cls                      # return the class unchanged
    return decorator


def build_model(cfg) -> nn.Module:
    try:
        model_cls = MODEL_REGISTRY[cfg.name]
    except KeyError:
        raise ValueError(
            f"Unknown model '{cfg.name}'. Available: {sorted(MODEL_REGISTRY)}"
        ) from None
    return model_cls(cfg)
