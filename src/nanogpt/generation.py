
#%% Imports:

from pathlib import Path
import torch
from dataclasses import dataclass
import tyro

from nanogpt.tokenizers import Tokenizer

'''
This file contains the code to generate text using checkpoint from a trained model.

'''

#%% Functions:

def truncate_at_end(ids: list[int], end_id: int | None, start: int) -> list[int]:
    '''Cut ids at the first end_id at or after position start; unchanged if none.'''
    if end_id is not None and end_id in ids[start:]:
        return ids[:ids.index(end_id, start)]
    return ids


def generate_text(model, tok: Tokenizer, prompt: str, new_tokens: int, temperature: float, top_k: int | None, use_cache: bool = True, seed: int = 0) -> str:
    '''
    Function to generate new text using a trained model.

    Parameters
    ----------
    model: nn.Module
        Model to use for generation.
    tok: Tokenizer
        Tokenizer to decode generated token ids.
    prompt: string
        Promopt to use for generation.
    new_tokens: int
        Number of new tokens to generate.
    temperature: float, optional
        Divides the logits before softmax.
    top_k: int, optional
        Keep only the top_k most likely tokens.
    use_cache: bool
        Use key-value cache if True. The default is True.
    seed: int
        Random seed to use.

    Returns
    -------
    new_text: string
        The user prompt followed by the generated text.

    Raises
    ------
    ValueError
     If key-value caching is requested and the prompt + new_tokens exceed the block size.

    '''
    device = next(model.parameters()).device
    torch.manual_seed(seed)

    # Process the prompt (the start token marks the beginning of a document):
    idx = [tok.start_id] + tok.encode(prompt)

    # Error handling (the KV cache holds at most 'block_size' tokens):
    block_size = model.cfg.block_size
    if use_cache and len(idx) + new_tokens > block_size:
        raise ValueError(f'Prompt ({len(idx)} tokens) + new_tokens ({new_tokens}) '
                        f'exceeds the model block size ({block_size}); use --no-use-cache '
                        f'for longer generation.')

    # Generate and drop the start token:
    context = torch.tensor([idx], dtype=torch.long, device=device)      # (1, T)
    new_ids = model.generate(context, new_tokens, temperature, top_k, use_cache=use_cache)[0].tolist()[1:]

    # Stop at the first end-of-document token after the prompt, if any:
    new_ids = truncate_at_end(new_ids, tok.end_id, start=len(idx) - 1)

    return tok.decode(new_ids)

