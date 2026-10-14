
#%% Imports:

from pathlib import Path
import torch
from dataclasses import dataclass
import tyro

from nanogpt.checkpoint import load_model_tokenizer
from nanogpt.generation import generate_text

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
    temperature: float = 1.0            # divides the logits before softmax, values < 1 sharpen the distribution
    top_k: int | None = None            # keep only the top_k most likely tokens
    use_cache: bool = True              # use key-value cache if True, ignored for Bigram model

    seed: int = 1337                    # random seed
    device: str = device                # device to run the training on

#%% Main function:

def main(g: GenerateConfig):
    '''
    Function to generate text from a trained model.

    Parameters
    ----------
    g: GenerateConfig
        Generation configurations.

    Returns
    -------
    new_text: string
        The user prompt followed by the generated text.

    '''
    # Load the model and tokenizer from the checkpoint:
    ckpt_path = g.runs_root / g.run_name / g.ckpt
    model, tok, cfg = load_model_tokenizer(ckpt_path, g.device)

    return generate_text(model, tok, g.prompt, g.new_tokens, g.temperature, g.top_k, g.use_cache, g.seed)


#%% Main call:

if __name__ == '__main__':
    new_text = main(tyro.cli(GenerateConfig))
    print('-' * 50)
    print(new_text)
