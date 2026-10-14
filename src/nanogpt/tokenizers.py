
#%% Imports:

from abc import ABC, abstractmethod
import tiktoken

from nanogpt.registry import TOKENIZERS

'''
This file contains the code to construct tokenizers. All tokenizers have consistent interfaces.

'''

#%% Common interface to build tokenizers:

def build_tokenizer(meta: dict):
    '''Function to build a tokenizer given a relavent 'meta' dictionary.'''
    return TOKENIZERS.get(meta['tokenizer']).from_meta(meta)


#%% Generic tokenizer class:

class Tokenizer(ABC):
    '''Class to implement a common interface for all tokenizers.'''

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @abstractmethod
    def encode(self, s: str) -> list[int]: ...

    @abstractmethod
    def decode(self, ids: list[int]) -> str: ...

    @property
    @abstractmethod
    def start_id(self) -> int:
        '''Token to begin generation from when the prompt is empty.'''

    @property
    @abstractmethod
    def end_id(self) -> int | None:
        '''Token to stop generation at; None means generate the full length.'''

    @abstractmethod
    def to_meta(self) -> dict:
        '''Everything needed to rebuild this tokenizer, saved in meta.json and checkpoints.'''

    @classmethod
    @abstractmethod
    def from_meta(cls, meta: dict) -> 'Tokenizer':
        '''Rebuild the tokenizer from the output of self.to_meta().'''


#%% Character tokenizer class:

@TOKENIZERS.register('char')
class CharTokenizer:
    '''Class to implement character-level tokenizer where the vocabulary is the set of characters in the text dataset.'''
    def __init__(self, text: str | list):
        '''
        Parameters
        ----------
        text: str or list
            Dataset or all characters in the dataset.

        Attributes
        ----------
        stoi: dict
            Maps characters to tokens.
        itos: dict
            Inverse map from tokens to their corresponding characters.

        '''
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}


    @property
    def vocab_size(self) -> int:
        '''Returns the unique number of characters in the dataset.'''
        return len(self.stoi)


    def encode(self, s: str) -> list[int]:
        '''
        Function to encode a given string using the generated token mapping.

        Parameters
        ----------
        s: str
            String to tokenize.

        Returns
        -------
        x: list[int]
            Tokenized list corresponding to the input string.

        '''
        return [self.stoi[c] for c in s]


    def decode(self, ids: list[int]) -> str:
        '''
        Function to decode a given list of tokens to corresponding string.

        Parameters
        ----------
        ids: list[int]
            List of tokens to decode.

        Returns
        -------
        s: str
            Decoded string.

        '''
        return ''.join(self.itos[i] for i in ids)


    @property
    def start_id(self) -> int:
        '''Token to begin generation from when the prompt is empty.'''
        return self.stoi['\n']


    @property
    def end_id(self) -> int | None:
        '''Token to stop generation at; None means generate the full length.'''
        return None


    def to_meta(self):
        '''Function to store metadata needed to reconstruct the class instance without needing the dataset.'''
        meta = {'tokenizer': 'char', 'vocab_size': self.vocab_size,
                'chars': ''.join(self.itos[i] for i in range(self.vocab_size))}
        return meta


    @classmethod
    def from_meta(cls, meta: dict) -> 'CharTokenizer':
        '''Function to construct an instance of the class given the output of self._meta() method.'''
        return cls(meta['chars'])                   # sorted(set(chars)) reproduces the same mapping


#%% OpenAI Tiktoken tokenizer:

@TOKENIZERS.register('gpt2')
class GPT2Tokenizer:
    '''Class to implement GPT-2 byte-pair encoding (BPE) tokenizer (tiktoken).'''

    def __init__(self):
        '''
        Attributes
        ----------
        enc: tiktoken.get_encoding instance
            To encode and decode text to token ids and back.
        eot: int
            End of text token id to split stories.

        '''
        self.enc = tiktoken.get_encoding('gpt2')
        self.eot = self.enc.eot_token            # <|endoftext|>, ID 50256

    @property
    def vocab_size(self) -> int:
        return self.enc.n_vocab                  # 50257 tokens

    def encode(self, s: str) -> list[int]:
        return self.enc.encode_ordinary(s)       # ignores special-token syntax in the text

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)

    @property
    def start_id(self) -> int:
        '''Token to begin generation from when the prompt is empty.'''
        return self.eot

    @property
    def end_id(self) -> int | None:
        '''Token to stop generation at; None means generate the full length.'''
        return self.eot

    def to_meta(self) -> dict:
        return {'tokenizer': 'gpt2', 'vocab_size': self.vocab_size, 'eot': self.eot}

    @classmethod
    def from_meta(cls, meta: dict) -> 'GPT2Tokenizer':
        return cls()                            # fixed vocabulary: nothing to restore
