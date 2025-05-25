import h5py
import time
import tqdm
import argparse
import functools

import numpy as np
from multiprocessing import Pool

from utils import construct_problem_matrices, construct_Y
from optimizers import optim_wrapper, perform_lasso

def message(msg):
    msg = textwrap.fill(msg, 78)
    print(msg)

def process_stack(obs_frames, timestamps, N=33, tau=0.011, n=8, T=0.005,
                  lmd=0.1, kappa=0.1, pixel_indices=):
    """Deblur an observed image stack.
    
    Parameters
    ----------
    obs_frames : np.ndarray
        Observed frame stack of shape (Frame Number, Columns, Rows)
    timestamps : np.ndarray
        The timestamps of the captured obserived frames.
    N : int, optional
        Number of frames used for deblurring. Default is 33.
    tau : float, optional
        Thermal time constant of camera. Default is 0.011.
    n : int, optional
        Max level of haar wavelets. Default is 8.
    T : float, optional
        Sampling period of camera in seconds. Default is 0.005
    lmd : float, optional
        Weight for sparsity. Default is 0.1.
    kappa : float, optional
        Weight for observations. Default is 0.1.
    frame_start : int, optional
        Index of the frame to start deblurring at. Default is 0.
    """
    timestamps = timestamps - timestamps[0]
    num_frames = len(timestamps)

    if not int(np.log2(N - 1)) == np.log2(N - 1):
        msg = f"Warning: Window size ({N}) is not of the form 2**l + 1. "
        msg += "Unreliable results."
        message(msg)

    # if not np.mod(N, 2):
    #   msg = f"Warning: Window size ({N}) is not of the form 2**l + 1. "
    #   msg += "Unreliable results."
    #   message(msg)

    p = Pool()
    for ipix