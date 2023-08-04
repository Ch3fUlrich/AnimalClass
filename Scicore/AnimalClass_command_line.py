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
    #root_dir = "\\\\toucan-all.scicore.unibas.ch\\donafl00-calcium$\\Users\\Sergej\\Steffen_Experiments"  
    #root_dir = "D:\\Steffen_Experiments"  
    #root_dir = "F:\\Steffen_Experiments"
    #root_dir = "D:\\Rodrigo"

    year_list = ["2021", "2022"]
    animals, bad_sessions = load_all(root_dir, wanted_animal_ids=wanted_animal_ids, wanted_session_ids=wanted_session_ids, generate=generate, delete=delete) # Load all animals
    #animals = load_all(root_dir, generate=True) # Load all animals
    #animals = load_all(root_dir, generate=True, units="single")#, delete=True) # Load all animals
    #animal.sessions[session_id].run_suite2p(regenerate=True, units="S1")
    #animal.sessions[session_id].run_suite2p(regenerate=True, units=["S1", "S2"])
    #animals["DON-009192"].sessions["20220306"].get_s2p_folder_paths(generate=True, regenerate=True, units="single")
    #animals["DON-009192"].sessions["20220221"].get_s2p_folder_paths(generate=True, regenerate=True, units="single")



    fps = 30
    seconds = 60
    window_size = fps*seconds # 1 minutes
    viz = Vizualizer(animals, save_dir = Animal.root_dir)

    for animal_id, animal in animals.items():
        print(f"{animal_id}: {list(animal.sessions.keys())}")
    load_all_procedure = "generating" if generate else "loading"
    print(f"Error {load_all_procedure} sessions: {bad_sessions}")
    clean_animals(animals, skip_animal=skip_animal, skip_session=skip_session, delete_used_subsessions=False)

def clean_animals(animals, skip_animal=[], skip_session=[], regenerate=False, delete_used_subsessions=False):
    plotting = False
    bad_sessions = []
    viz = Vizualizer(animals, save_dir = Animal.root_dir)
    for animal_id, animal in animals.items():
        for session_id, session in animal.sessions.items():
            #if animal_id in skip_animal and session_id in skip_session:
            #    continue
            print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Starting {animal_id} {session_id} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print(f"-----------------------------------Generating Initial Suite2P Files-----------------------------------")
            session.run_suite2p(regenerate=False, units="all")
            session.get_cabincorr_data_paths(generate=True, regenerate=regenerate, units="all")


            print(f"-----------------------------------Rerun Suite2P if data.bin is missing-----------------------------------")
            # Rerunning Suite2p if binary file is not present
            bin_fname = "data.bin"
            for s2p_path in session.s2p_folder_paths:
                binary_file_present = False
                part_to_rerun = False
                for part in session.session_parts:
                    if part in s2p_path:
                        binary_file_present = os.path.exists(os.path.join(s2p_path, "plane0", bin_fname))
                        if binary_file_present:
                            break
                        else:
                            part_to_rerun = part
                    if part_to_rerun:
                        print(f"binary file not present in {s2p_path}")
                        session.run_suite2p(regenerate=True, units=part_to_rerun)
            #session.get_cabincorr_data_paths(regenerate=False, units="single")
            #session.get_cabincorr_data_paths(regenerate=True, units="single")

            
            print(f"-----------------------------------Loading Units-----------------------------------")
            units = session.get_units(get_geldrying=True)

            print(f"-----------------------------------Merging Units-----------------------------------")
            merged_unit = session.merge_units(generate=True, regenerate=regenerate, delete_used_subsessions=delete_used_subsessions)
            #merged_unit = session.merge_units(generate=True, regenerate=False, delete_used_subsessions=True)

            dir_exist_create(os.path.join(viz.save_dir, animal_id))
            dir_exist_create(os.path.join(viz.save_dir, animal_id, session_id))
            viz.save_dir = os.path.join(viz.save_dir, animal_id, session_id)

            if plotting:
                print(f"-----------------------------------Plotting-----------------------------------")
                print(f"-----------------------------------Plotting Individual Munits-----------------------------------")
                for unit_id, unit in units.items():
                    viz.unit_footprints(unit)
                    viz.unit_contours(unit)
                    viz.traces(unit.fluoresence, num_cells="all", animal_id=animal_id, session_id=session_id, unit_id=unit.unit_id)
                    #viz.save_rasters_fig(unit.c, animal_id=animal_id, session_id=session_id, unit_id=unit.unit_id)
                    # Plot Good Bad fluorescence data in Batches of size 10
                    session_figure_dir = viz.save_dir
                    batch_save_dir = os.path.join(viz.save_dir, "batch_10")
                    dir_exist_create(batch_save_dir)
                    viz.save_dir = batch_save_dir
                    viz.unit_fluorescence_good_bad(unit, batch_size="all", starting=0)
                    #viz.unit_fluorescence_good_bad(unit, batch_size=10, starting=0)
                    viz.save_dir = session_figure_dir

                print(f"-----------------------------------Plotting Full Session-----------------------------------")
                # Plot Full Session Unit 
                unit_all = session.get_Unit_all()
                viz.unit_footprints(unit_all)
                viz.unit_contours(unit_all)
                viz.traces(unit_all.fluoresence, num_cells=100, animal_id=animal_id, session_id=session_id, unit_id=unit_all.unit_id)


                print(f"-----------------------------------Plotting Contours-----------------------------------")
                # print contours of all combination of units size 2
                ##################################S2P Registration (Footprint position shift determination)##############################
                best_unit = session.get_most_good_cell_unit()
                min_num_usefull_cells = best_unit.num_not_geldrying() / 3
                units = session.get_usefull_units(min_num_usefull_cells)
                for unit_id, unit in units.items():
                    if unit_id != best_unit.unit_id:
                        if unit.yx_shift == [0, 0]:
                            session.calc_unit_yx_shifts(best_unit, units)
                            break
                from itertools import permutations
                unit_ids = list(units.keys())
                combinations = list(permutations(unit_ids, 2))
                ##Plotting original contours
                #for combination in combinations:
                #    if combination[0] < combination[1]:
                #        plt.figure(figsize=(20, 20))
                #        viz.multi_unit_contours(units, combination=combination, plot_center=True)
                #Plotting shifted contours
                for combination in combinations:
                    if combination[0] < combination[1]:
                        plt.figure(figsize=(20, 20))
                        viz.multi_unit_contours(units, combination=combination, plot_center=True, shift=True)

                print(f"-----------------------------------Plotting Merged Session-----------------------------------")
                # Plot merged contours
                merged_contours = best_unit.contours
                for unit_id, unit in units.items():
                    if unit_id == best_unit.unit_id:
                        continue
                    merged_contours = np.concatenate([merged_contours, unit.contours])
                plt.figure(figsize=(10, 10))
                title_comment = f" {animal_id}_{session_id}_MUnit_{merged_unit.unit_id} merged"
                viz.contours(merged_contours, comment=title_comment)
                plt.savefig(os.path.join(viz.save_dir, f"Contours_{title_comment.replace(' ', '_')}.png"), dpi=300)

                # Plot deduplicated contours
                viz.unit_contours(merged_unit)   

                # plot contours without geldrying
                plt.figure(figsize=(10, 10))
                title_comment = f" {animal_id}_{session_id}_MUnit {merged_unit.unit_id} not geldrying"
                viz.contours(np.array(merged_unit.contours)[merged_unit.cell_geldrying==False], comment=title_comment)
                plt.savefig(os.path.join(viz.save_dir, f"Contours_{title_comment.replace(' ', '_')}.png"), dpi=300)


                #viz.unit_fluorescence_good_bad(merged_unit, batch_size=10, interactive=False, plot_duplicates=False)
                viz.unit_fluorescence_good_bad(merged_unit, batch_size="all", interactive=False, plot_duplicates=False)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    wanted_animal_ids = sys.argv[1:2] if len(arguments) >= 1 else ["all"]
    wanted_session_ids = sys.argv[2:3] if len(arguments) >= 2 else ["all"]
    if len(arguments) > 3:
        print("Command line usage: <animal_id> <session_id>")
        print("If an argument is not specified the corresponding argument is set to 'all'")
    print(f"Start Cleaning {wanted_animal_ids}, {wanted_session_ids}")
    main(wanted_animal_ids=wanted_animal_ids, wanted_session_ids=wanted_session_ids)#skip_animal=["DON-009191"], skip_session=["20220225"]