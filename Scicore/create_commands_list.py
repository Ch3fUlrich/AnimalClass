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


def main(wanted_animal_ids = ["all"], wanted_session_ids=["all"], skip_animal=[], skip_session=[]):
    #TODO: skipping option is not integrated
    with open("commands.cmd", 'w') as f:
        root_dir = "\\\\toucan-all.scicore.unibas.ch\\donafl00-calcium$\\Users\\Sergej\\Steffen_Experiments"  
        Animal.root_dir = root_dir
        animals, bad_sessions = load_all(root_dir, animal_ids=wanted_animal_ids, sessions=wanted_session_ids) # Load all animals
        for animal_id, animal in animals.items():
            if animal_id in wanted_animal_ids or "all" in wanted_animal_ids:

                for session_id, session in animal.sessions.items():
                    if session_id in wanted_session_ids or "all" in wanted_session_ids:
                        f.write(f"python /scicore/projects/donafl00-calcium/Users/Sergej/AnimalClass_command_line.py {animal_id} {session_id}\n")

if __name__ == "__main__":
    arguments = sys.argv[1:]
    wanted_animal_ids = sys.argv[1:2] if len(arguments) >= 1 else ["all"]
    wanted_session_ids = sys.argv[2:3] if len(arguments) >= 2 else ["all"]
    if len(arguments) > 3:
        print("Command line usage: <animal_id> <session_id>")
        print("If an argument is not specified the corresponding argument is set to 'all'")
    print(f"Creating commands.cmd for {wanted_animal_ids}, {wanted_session_ids}")
    main(wanted_animal_ids, wanted_session_ids)#skip_animal=["DON-009191"], skip_session=["20220225"]