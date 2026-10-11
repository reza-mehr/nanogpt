
#%% Imports:

import tiktoken

'''
This file contains the code to construct tokenizers. All tokenizers have consistent interfaces.

'''

#%% Character tokenizer class:

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


#%% OpenAI Tiktoken tokenizer:

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
