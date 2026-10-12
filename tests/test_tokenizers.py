
#%% Imports:

import pytest

from nanogpt.tokenizers import build_tokenizer, CharTokenizer, GPT2Tokenizer

'''
This file contains the code to test the tokenizers.

'''

#%% Global variables:

SAMPLE = 'Once upon a time, there was a little dog.\nHe liked to run!'

#%% Tests:

def make_tokenizers():
    return [CharTokenizer(SAMPLE), GPT2Tokenizer()]

@pytest.mark.parametrize('tok', make_tokenizers(), ids=['char', 'gpt2'])
def test_round_trip(tok):
    assert tok.decode(tok.encode(SAMPLE)) == SAMPLE

@pytest.mark.parametrize('tok', make_tokenizers(), ids=['char', 'gpt2'])
def test_rebuild_from_meta(tok):
    rebuilt = build_tokenizer(tok.to_meta())
    assert type(rebuilt) is type(tok)
    assert rebuilt.encode(SAMPLE) == tok.encode(SAMPLE)
    assert rebuilt.vocab_size == tok.vocab_size

@pytest.mark.parametrize('tok', make_tokenizers(), ids=['char', 'gpt2'])
def test_special_ids_valid(tok):
    assert 0 <= tok.start_id < tok.vocab_size
    assert tok.end_id is None or 0 <= tok.end_id < tok.vocab_size
    assert tok.vocab_size < 2**16                      # fits the uint16 file format

def test_unknown_tokenizer_raises():
    with pytest.raises(ValueError, match='Unknown tokenizer'):
        build_tokenizer({'tokenizer': 'nope'})
