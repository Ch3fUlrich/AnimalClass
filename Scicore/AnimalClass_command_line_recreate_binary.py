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
from Classes import Analyzer, Session, Animal, Vizualizer, Unit, Binary_loader, Merger, load_all
from Helper import *


def main(wanted_animal_ids = ["all"], wanted_session_ids=["all"], generate=True, delete=False, skip_animal=[], skip_session=[]):
    #TODO: skipping option is not integrated
    # Init Directories and Notebook settings
    root_dir = "/scicore/projects/donafl00-calcium/Users/Sergej/Steffen_Experiments"  
    Animal.root_dir = root_dir

    year_list = ["2021", "2022"]
    animals, bad_sessions = load_all(root_dir, units="all", wanted_animal_ids=wanted_animal_ids, wanted_session_ids=wanted_session_ids, generate=generate, delete=False) # Load all animals
    to_fix = [["DON-008497", "20220210"], ["DON-008497", "20220211"], ["DON-008499", "20220205"], ["DON-009191", "20220314"]
            ,["DON-009191", "20220216"], ['DON-009191', '20220316'], ['DON-009192', '20220307']]
    for animal_id, session_id in to_fix:
        try:
            animal = animals[animal_id].sessions[session_id]
        except:
            continue
        print(animal_id)
        print(session_id)
        animal.get_tiff_data_paths(generate=True, regenerate=False, units="all")
        animal.get_s2p_folder_paths(generate=True, regenerate=True, units="single")
    

if __name__ == "__main__":
    arguments = sys.argv[1:]
    wanted_animal_ids = sys.argv[1:2] if len(arguments) >= 1 else ["all"]
    wanted_session_ids = sys.argv[2:3] if len(arguments) >= 2 else ["all"]
    if len(arguments) > 3:
        print("Command line usage: <animal_id> <session_id>")
        print("If an argument is not specified the corresponding argument is set to 'all'")
    print(f"Start Cleaning {wanted_animal_ids}, {wanted_session_ids}")
    main(wanted_animal_ids=wanted_animal_ids, wanted_session_ids=wanted_session_ids, generate=False)
    #skip_animal=["DON-009191"], skip_session=["20220225"]