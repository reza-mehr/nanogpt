
#%% Imports:

from pathlib import Path
import torch
from dataclasses import dataclass
import tyro

from nanogpt.checkpoint import load_model_tokenizer

'''
This file contains the code to generate text using a trained model.

'''
#%% Global variables:

device = (
    'cuda'
    if torch.cuda.is_available()
    else 'mps'
    if torch.backends.mps.is_available()
    else 'cpu'
)

#%% Profiling configurations:

@dataclass
class GenerateConfig:
    run_name: str                       # run folder name, empty means a timestamp
    runs_root: Path = Path('runs')      # parent folder for all runs
    ckpt: str = 'best.pt'               # checkpoint to use

    prompt: str = ''
    new_tokens: int = 200               # number of new tokens to generate
    use_cache: bool = True              # use key-value cache if True, ignored for Bigram model

    seed: int = 1337                    # random seed
    device: str = device                # device to run the training on


#%% Main code:

def generate(g: GenerateConfig) -> str:
    '''
    Function to generate new text using a trained model.

    Parameters
    ----------
    g: GenerateConfig
        Generation configurations.

    Returns
    -------
    new_text: string
        The user prompt followed by the generated text.

    '''
    torch.manual_seed(g.seed)

    # Load the model and tokenizer from the checkpoint:
    ckpt_path = g.runs_root / g.run_name / g.ckpt
    model, tok, cfg = load_model_tokenizer(ckpt_path, g.device)

    # Process the prompt (the start token marks the beginning of a document):
    idx = [tok.start_id] + tok.encode(g.prompt)
    context = torch.tensor([idx], dtype=torch.long, device=g.device)    # (1, T)

    # Error handling (the KV cache holds at most block_size tokens):
    block_size = cfg['model']['block_size']
    if g.use_cache and len(idx) + g.new_tokens > block_size:
        raise ValueError(f'Prompt ({len(idx)} tokens) + new_tokens ({g.new_tokens}) '
                        f'exceeds the model block size ({block_size}); use --no-use-cache '
                        f'for longer generation.')

    # Generate and drop the start token:
    new_ids = model.generate(context, g.new_tokens, use_cache=g.use_cache)[0].tolist()[1:]

    # Stop at the first end-of-document token after the prompt, if any:
    n_prompt = len(idx) - 1
    if tok.end_id is not None and tok.end_id in new_ids[n_prompt:]:
        new_ids = new_ids[:new_ids.index(tok.end_id, n_prompt)]

    return tok.decode(new_ids)


#%% Main call:

if __name__ == '__main__':
    new_text = generate(tyro.cli(GenerateConfig))
    print('-'*50)
    print(new_text)
