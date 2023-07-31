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
#module_path = os.path.abspath(os.path.join('../'))
#print(module_path)
#sys.path.append(module_path)

from manifolds.donlabtools.utils.calcium import calcium
from manifolds.donlabtools.utils.calcium.calcium import *
from Classes import Analyzer, Session, Animal, Vizualizer, Unit, Binary_loader, Merger, load_all
from Helper import *


def main(wanted_animal_ids = ["all"], wanted_session_ids=["all"], generate=False):
    # Init Directories and Notebook settings
    root_dir = "/scicore/projects/donafl00-calcium/Users/Sergej/Steffen_Experiments"  
    #root_dir = "D:\\Steffen_Experiments"  
    #root_dir = "F:\\Steffen_Experiments"
    Animal.root_dir = root_dir
    #root_dir = "D:\\Rodrigo"

    year_list = ["2021", "2022"]
    animals, bad_sessions = load_all(root_dir, animal_ids=wanted_animal_ids, sessions=wanted_session_ids, generate=generate) # Load all animals
    #animals = load_all(root_dir, generate=True) # Load all animals
    #animals = load_all(root_dir, generate=True, units="single")#, delete=True) # Load all animals
    #animal.sessions[session_id].run_suite2p(regenerate=True, units="S1")
    #animal.sessions[session_id].run_suite2p(regenerate=True, units=["S1", "S2"])
    #animals["DON-009192"].sessions["20220306"].get_s2p_folder_paths(generate=True, regenerate=True, units="single")
    #animals["DON-009192"].sessions["20220221"].get_s2p_folder_paths(generate=True, regenerate=True, units="single")

    fps = 30
    seconds = 60
    window_size = fps*seconds # 1 minutes
    viz = Vizualizer(animals)

    plotting = True
    print(f"broken sessions: {bad_sessions}")

if __name__ == "__main__":
    arguments = sys.argv[1:]
    wanted_animal_ids = sys.argv[1:2][0]
    wanted_session_ids = sys.argv[2:3][0]
    print(arguments)
    print(wanted_animal_ids)
    #main()