
#%% Imports:

'''
This file contains the code to register PyTorch tokenizers and models for clean build from a configuration file.

'''

#%% Registry class:

class Registry:
    '''Class to define a registry for certain 'kind' of classes, e.g. models or tokenizers. Maps names to classes; filled by the @register decorator at import time.'''

    def __init__(self, kind: str):
        '''
        Parameters
        ----------
        kind: string
            Type of registry.

        Attributes
        ----------
        kind
        _classes: dict
            Registered classes of the desired 'kind'.

        '''
        self.kind, self._classes = kind, {}


    def register(self, name: str):
        '''
        Function to register a class.

        Parameters
        ----------
        name: str
            Registry key for the class.

        Raises
        ------
        ValueError
            If the requested class name is already registered.

        '''
        def decorator(cls):
            if name in self._classes:
                raise ValueError(f"{self.kind} '{name}' is already registered")
            self._classes[name] = cls
            return cls
        return decorator


    def get(self, name: str):
        '''
        Function to get a class object given its name.

        Parameters
        ----------
        name: str
            Registry key for the class.

        Returns
        -------
        class:
            Object associated with registered name.

        Raises
        ------
        ValueError
            If an unknown class is requested.

        '''
        if name not in self._classes:
            raise ValueError(f"Unknown {self.kind} '{name}'. Available: {sorted(self._classes)}")
        return self._classes[name]


# Registries of 'model' and 'tokenizer' kinds:
MODELS = Registry('model')
TOKENIZERS = Registry('tokenizer')
