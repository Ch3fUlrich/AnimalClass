# imports
# Statistics
import seaborn as sns
import pandas as pd
import numpy as np

# Plotting
import matplotlib as mlp
import matplotlib.pyplot as plt, mpld3 #plotting and html plots
plt.style.use('dark_background')
#plt.style.use('default')
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Regular Expression searching
import re

# Suite2p for TIFF file analysis
import suite2p
from suite2p.run_s2p import run_s2p, default_ops
from suite2p.registration import register


# Used for Popups
import tkinter as tk

import nest_asyncio

# for progress bar support
from tqdm import tqdm

# interact with system
import os
import sys
import copy


# statistics
import scipy
import math


# Mesc file analysis
import h5py
from tifffile import tifffile, imread
import pathlib


# add root directory to be able to import packages
# todo: make all packages installable so they can be called/imported by environment
module_path = os.path.abspath(os.path.join('../'))
sys.path.append(module_path)

from manifolds.donlabtools.utils.calcium import calcium
from manifolds.donlabtools.utils.calcium.calcium import *
from Classes import *
from Helper import *


def main(part):
    # Init Directories and Notebook settings
    root_dir = "/scicore/home/donafl00/mauser00/code/AnimalClass/Scicore/data"  
    Animal.root_dir = root_dir
    #root_dir = "\\\\toucan-all.scicore.unibas.ch\\donafl00-calcium$\\Users\\Sergej\\Steffen_Experiments"  
    animals = load_all(root_dir) # Load all animals
    animal_id = "DON-009192"
    session_id = "20220223"
    animal = animals[animal_id]
    session = animal.sessions[session_id]
    for s2p_path in session.s2p_folder_paths:
        if s2p_path.split("suite2p_")[-1] == part:
            c = run_cabin_corr(root_dir, os.path.join(s2p_path, "plane0"), animal_id, session_id)
            c.corr_parallel_flag = True
            c.zscore = True 
            c.n_tests_zscore = 1000
            c.n_cores = 32
            c.recompute_correlation = False
            c.binning_window = 30        # binning window in frames
            c.subsample = 1              # subsample traces by this factor
            c.scale_by_DFF = True        # scale traces by DFF
            c.shuffle_data = True
            c.subselect_moving_only = False
            c.subselect_quiescent_only = False
            c.make_correlation_dirs()
            c.compute_correlations()

if __name__ == "__main__":
    arguments = sys.argv[1:]
    main(part=arguments[0])