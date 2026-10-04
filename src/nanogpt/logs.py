
#%% Imports:
import logging
import sys
from pathlib import Path

from nanogpt.config import TrainConfig

#%% Global variables:

LOGGER_NAME = 'nanogpt'

#%% Main code:

def setup_logger(cfg: TrainConfig) -> logging.Logger:
    '''
    Function to set up logging to stdout and out_dir/train.log.
    Safe to call more than once per process.

    Parameters
    ----------
    cfg: TrainConfig
        Configurations to use.

    Returns
    -------
    logger: logging.Logger
        Constructed logger.

    '''
    # Initialization:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(cfg.log_level)
    logger.propagate = False                        # don't also print via the root logger
    fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    out_dir = cfg.out_dir

    # Clear handlers from a previous run:
    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)

    # Recollect the handles:
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(out_dir / 'train.log', mode='a')):
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    return logger
