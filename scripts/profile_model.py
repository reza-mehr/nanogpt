
#%% Imports:

from pathlib import Path
import torch
from torch.profiler import ProfilerActivity, profile, schedule
import tyro

from dataclasses import dataclass

from nanogpt.benchmarking import make_step, sync_device
from nanogpt.config import ModelConfig

'''
This file contains the profiling code for the model.

'''

#%% Profiling configurations:

@dataclass
class ProfileConfig:
    scaled_dot_prod: bool = True
    ln_torch: bool = True
    T: int = 512
    device: str = 'mps'
    batch_size: int = 16
    rows: int = 15                  # number of operations to show in the table; -1 shows all
    out_dir: str = 'profiles'       # folder for trace files


#%% Functions:

def main(p: ProfileConfig) -> None:

    # Construct the model, optimizer, and corresponding step function:
    cfg = ModelConfig(name='gpt', vocab_size=65, block_size=p.T, n_embed=384,
                      head_num=6, head_size=64, n_blocks=6, dropout=0.0,
                      scaled_dot_prod=p.scaled_dot_prod, ln_torch=p.ln_torch)
    step = make_step(cfg, p.device, p.batch_size, p.T)

    # Profiler activities:
    activities = [ProfilerActivity.CPU]
    if p.device == 'cuda': activities.append(ProfilerActivity.CUDA)

    # Skip 1 step, warm up for 2 steps (recorded but discarded), then record step 3:
    sched = schedule(wait=1, warmup=2, active=3)
    with profile(activities=activities, schedule=sched, profile_memory=True, record_shapes=True) as prof:
        for _ in range(6):
            step()
            sync_device(p.device)
            prof.step()               # inform the profiler that a step finished

    # Print the results:
    print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=p.rows, max_name_column_width=40))

    attn_ops = {e.key for e in prof.key_averages() if 'attention' in e.key.lower()}
    print('\nAttention ops that ran:', sorted(attn_ops) or 'none (manual path)')

    # Save results:
    out = Path(p.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    impl = 'sdpa' if p.scaled_dot_prod else 'manual'
    trace = out / f'trace_{impl}_T{p.T}_{p.device}.json'
    prof.export_chrome_trace(str(trace))
    print(f'Timeline: open {trace} at https://ui.perfetto.dev')


#%% Main function:

if __name__ == '__main__':
    main(tyro.cli(ProfileConfig))
