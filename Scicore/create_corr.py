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

# Init Directories and Notebook settings
#root_dir = "\\\\toucan-all.scicore.unibas.ch\\donafl00-calcium$\\Users\\Sergej\\Steffen_Experiments"  
root_dir = "/scicore/projects/donafl00-calcium/Users/Sergej/Steffen_Experiments"  
Animal.root_dir = root_dir


def main(wanted_animal_ids = ["all"], wanted_session_ids=["all"]):
    animals = load_all(root_dir, wanted_animal_ids=wanted_animal_ids, wanted_session_ids=wanted_session_ids) # Load all animals

    for animal_id, animal in animals.items():
        print(f"{animal_id}: {list(animal.sessions.keys())}")

    create_corr(animals)

def create_corr(animals):
    for animal_id, animal in animals.items():
        for session_id, session in animal.sessions.items():
            do_cabincoor(session, unit="")
            do_cabincoor(session, unit="merged")
            session.load_corr_matrix(unit_id="all")
            session.load_corr_matrix(unit_id="merged")
            delete_bin_tiff(session)
            

def do_cabincoor(session, unit=""):
    for s2p_path in session.s2p_folder_paths:
            splitted_path = s2p_path.split("suite2p_")
            if splitted_path[-1] == unit or len(splitted_path)==1:
                c = run_cabin_corr(root_dir, os.path.join(s2p_path, "plane0"), session.animal_id, session.session_id)
                c.corr_parallel_flag = True
                c.zscore = True 
                c.n_tests_zscore = 1000
                c.n_cores = 32
                c.recompute_correlation = False
                c.binning_window = 30        # binning window in frames
                c.subsample = 1              # subsample traces by this factor
                c.scale_by_DFF = True        # scale traces by DFF
                c.shuffle_data = False
                c.subselect_moving_only = False
                c.subselect_quiescent_only = False
                c.make_correlation_dirs()
                c.compute_correlations()

def delete_bin_tiff(session):
    #Delete binaries
    del_tiff = True
    for s2p_folder in session.s2p_folder_paths:
        s2p_folder_ending = s2p_folder.split("suite2p")[-1]
        print(s2p_folder_ending)
        iscell_path = os.path.join(s2p_folder, "plane0", "iscell.npy")
        iscell_count = -1
        print(iscell_path)
        if os.path.exists(iscell_path):
            iscell = np.load(iscell_path)[:,0]
            iscell_count = sum(iscell)
        
        notgel_path = os.path.join(s2p_folder, "plane0", "cell_drying.npy")
        notgel_count = -1
        if os.path.exists(notgel_path):
            notgel = np.load(notgel_path)==0
            notgel_count = sum(notgel)
        print(iscell_count)
        print(notgel_count)
        if s2p_folder_ending == "":
            binary_path = os.path.join(s2p_folder, "plane0", "data.bin")
            if os.path.exists(binary_path):
                os.remove(binary_path)
        elif iscell_count != -1 and notgel_count !=-1:
            binary_path = os.path.join(s2p_folder, "plane0", "data.bin")
            print(binary_path)
            if os.path.exists(binary_path):
                os.remove(binary_path)
        else:
            del_tiff = False

    #Delete Tiffs
    if del_tiff:
        for tiff_path in session.tiff_data_paths:
            if os.path.exists(tiff_path):
                os.remove(tiff_path)  

if __name__ == "__main__":
    arguments = sys.argv[1:]
    wanted_animal_ids = sys.argv[1:2] if len(arguments) >= 1 else ["all"]
    wanted_session_ids = sys.argv[2:3] if len(arguments) >= 2 else ["all"]
    if len(arguments) > 3:
        print("Command line usage: <animal_id> <session_id>")
        print("If an argument is not specified the corresponding argument is set to 'all'")
    print(f"Start Cleaning {wanted_animal_ids}, {wanted_session_ids}")
    main(wanted_animal_ids=wanted_animal_ids, wanted_session_ids=wanted_session_ids)#skip_animal=["DON-009191"], skip_session=["20220225"]
