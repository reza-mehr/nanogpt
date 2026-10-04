
#%% Imports:

import torch.nn as nn

'''
This file contains the code to register PyTorch models for clean build from a configuration file.

'''

#%% Main code:

# Initialization:
MODEL_REGISTRY: dict[str, type[nn.Module]] = {}

def register_model(name: str):
    '''
    Function to register a model.

    Parameters
    ----------
    name: str
        Model class name.

    Raises
    ------
    If the requested model name is already registered.

    '''
    def decorator(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered")
        MODEL_REGISTRY[name] = cls
        return cls                      # return the class unchanged
    return decorator


def build_model(cfg) -> nn.Module:
    '''
    Function to build a model instance.

    Parameters
    ----------
    cfg: TrainConfig
        Train configs; see config.py for details.

    Returns
    -------
    model: nn.Module
        Constructed model.

    Raises
    ------
    ValueError
        If an unknown model is requested.

    '''
    try:
        model_cls = MODEL_REGISTRY[cfg.model.name]
    except KeyError:
        raise ValueError(
            f"Unknown model '{cfg.model.name}'. Available: {sorted(MODEL_REGISTRY)}"
        ) from None
    return model_cls(cfg)
