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

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
from ipywidgets import interact

# Regular Expression searching
import re

# Suite2p for TIFF file analysis
import suite2p
from suite2p.run_s2p import run_s2p, default_ops
from suite2p.registration import register

# Used for Popups
import tkinter as tk

import nest_asyncio

import parmap
# for progress bar support
from tqdm import tqdm

# interact with system
import os
import sys
import copy
import shutil
import psutil


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
from Helper import *
from manifolds.donlabtools.utils.calcium import calcium
from manifolds.donlabtools.utils.calcium.calcium import *
def load_all(root_dir, animal_ids=["all"], generate=False, regenerate=False, units="single", delete=False):
    """
    Loads animal data from the specified root directory for the given animal IDs.

    Parameters:
    - root_dir (string): The root directory path where the animal data is stored.
    - animal_ids (list, optional): A list of animal IDs to load. Default is ["all"].
    - generate (bool, optional): If True, generates new session data. Default is False.
    - regenerate (bool, optional): If True, regenerates existing session data. Default is False.
    - units (string, optional): Specifies the units. Default is "single".
    - delete (bool, optional): If True, deletes session data. Default is False.

    Returns:
    - animals_dict (dict): A dictionary containing animal IDs as keys and corresponding Animal objects as values.
    """
    animal_ids = get_directories(root_dir)
    animals_dict = {}

    # Search for animal_ids
    for animal_id in animal_ids:
        if animal_id in animal_ids or animal_ids[0] == "all":
            sessions_path = os.path.join(root_dir, animal_id)
            sessions = get_directories(sessions_path)
            yaml_file_name = os.path.join(root_dir, animal_id, f"{animal_id}.yaml")
            animal = Animal(yaml_file_name)
            # Search for 2P Sessions
            for session in sessions:
                try:
                    animal.get_session_data(session, generate=generate, regenerate=regenerate, units=units, delete=delete)
                except:
                    print(f"no session data found for session {session} from {animal_id}")
            animals_dict[animal_id] = animal
    return animals_dict

class Analyzer:
    # Pearson and histogram plot and save
    mean_threshold = 0.1
    std_threshold = 0.15
    correct_mean = 0.007428876195354758


    def __init__(self, animals={}):
        self.animals = animals
        self.good = self.bad = self.evaluate_datasets_count()

    def good_mean_std(self, mean, std):
        return True if mean < Analyzer.mean_threshold or std > Analyzer.std_threshold else False

    def evaluate_datasets_count(self, animals=None):
        good = 0
        bad = 0
        if animals == None:
            animals = self.animals
        for animal_id, animal in animals.items():
            try:
                for session_id, session in animal.sessions.items():
                    corr_matrix, pval_matrix = session.load_corr_matrix()
                    mean = np.mean(corr_matrix.flatten())
                    std = np.std(corr_matrix.flatten())
                    if self.good_mean_std(mean, std):
                        good += 1
                    else:
                        bad += 1
            except:
                print("Error while evaluation datasets")
        return good, bad

    def lin_reg(self, data):
        length = np.arange(len(data))
        lin_reg = scipy.stats.linregress(length, data)
        return lin_reg

    def get_linreg_slope_intercept(self, data):
        linreg = self.lin_reg(data)
        return linreg.slope, linreg.intercept

    def cont_mean_increase(self, mean_stds, num_bad_means = 30*60*1.5, 
                       num_not_bad_means=30*60*0.9):
        """
        Check if the mean of the data increases for 1.5 minutes without a 0.9 minutes break (30fps)

        Args:
            data (numpy.ndarray): A 2D numpy array containing mean and standard deviation values.

        Returns:
            bool: True if the mean values are within the threshold, False otherwise.
        """
        bad = False
        reason = ""
        bad_mean_counter = 0
        maybe_not_bad_counter = 0
        old_mean = mean_stds[0][0]
        min_std = np.min(mean_stds[:, 1])

        for pos, mean_std in enumerate(mean_stds[1:]):
            mean = mean_std[0]
            mean_diff = mean-old_mean
            mean_diff -=  min_std/(1/(abs(mean_diff/mean)))
            old_mean = mean
            if math.isnan(mean):
                bad = True
                reason = "nan"
                break
            if mean_diff > 0: 
                bad_mean_counter += 1
            else:
                maybe_not_bad_counter += 1
                if maybe_not_bad_counter > num_not_bad_means:
                    bad_mean_counter = 0
                    maybe_not_bad_counter = 0

            if bad_mean_counter >= num_bad_means: # 1 minute wide window mean to high for 1 minute  
                bad = True
                reason = "cont. increase"
                break
        return bad, reason+" c: "+str(bad_mean_counter)+" not bad "+str(maybe_not_bad_counter)#, pos/30

    def cont_mode_increase(self, mode_stds, num_bad_modes = 30*60*1, 
                       num_not_bad_modes=30*60*0.45):
        """
        !!!!!!!!!!!!Not usefull takes too long!!!!!!!!!!!!!!!!!!!!!!!!!!!
        Check if the mode of the data increases for 1.5 minutes without a 0.7 minutes break (30fps)

        Args:
            data (numpy.ndarray): A 2D numpy array containing mean and standard deviation values.

        Returns:
            bool: True if the mean values are within the threshold, False otherwise.
        """
        print("Warning: this method is not finetuned")
        bad = False
        reason = ""
        bad_mode_counter = 0
        maybe_not_bad_counter = 0
        old_mode = mode_stds[0][0]
        min_std = np.min(mode_stds[:, 1])

        for pos, mode_std in enumerate(mode_stds[1:]):
            mode = mode_std[0]
            mode_diff = mode-old_mode
            mode_diff -=  min_std/(1/(abs(mode_diff/mode)))
            old_mode = mode
            if math.isnan(mode):
                bad = True
                reason = "nan"
                break
            if mode_diff > 0: 
                bad_mode_counter += 1
            else:
                maybe_not_bad_counter += 1
                if maybe_not_bad_counter > num_not_bad_modes:
                    bad_mode_counter = 0
                    maybe_not_bad_counter = 0

            if bad_mode_counter >= num_bad_modes: # 1 minute wide window mode to high for 1 minute  
                bad = True
                reason = "num bad"
                break
        return bad, reason+" c: "+str(bad_mode_counter)+" not bad "+str(maybe_not_bad_counter)#, pos/30

    def geldrying(self, m_stds, bad_minutes=1.5, not_bad_minutes=0.9, mode="mean"):
        """
        Geldrying detection
        Args:
            data (numpy.ndarray): A 2D numpy array containing mean/mode and standard deviation values.

        Returns:
            bool: True if the standard deviation values are within the threshold, False otherwise.
        """
        #TODO: improve good bad detection currently only for geldrying used
        if mode == "mean":
            bad, reason = self.cont_mean_increase(m_stds, num_bad_means = 30*60*bad_minutes, num_not_bad_means=30*60*not_bad_minutes) 
        elif mode == "mode":
            bad, reason = self.cont_mode_increase(m_stds, num_bad_modes = 30*60*bad_minutes, num_not_bad_modes=30*60*not_bad_minutes) 
        return bad, reason
    
    def sliding_window(self, arr, window_size, step_size=1):
        """
        Generate sliding windows of size window_size over an array.

        Args:
            arr (list): The input array.
            k (int): The size of the sliding window.

        Yields:
            list: A sliding window of size window_size.

        Returns:
            None: If the length of the array is less than window_size.
        """
        n = len(arr)
        if n < window_size:
            return None
        window = arr[:window_size]
        yield window
        for i in range(window_size, n, step_size):
            window = np.append(window[step_size:], arr[i])
            yield window

    def sliding_mode_std(self, arr, window_size):
        """
        Compute the mode for each sliding window of size k over an array.

        Args:
            arr (list): The input array.
            k (int): The size of the sliding window.

        Returns:
            list: A list of mode values for each sliding window.
        """
        num_windows = len(arr)-window_size+1
        mode_stds = np.zeros([num_windows, 2])
        for num, window in enumerate(self.sliding_window(arr, window_size)):
            mode_stds[num, 0], count = scipy.stats.mode(window)
            mode_stds[num, 1] = np.std(window)
        return mode_stds

    def sliding_mean_std(self, arr, window_size):
        """
        Compute the sliding window mean and standard deviation of an array.

        Parameters:
        arr (array-like): Input array.
        k (int): Window size.

        Returns:
        list: A list of tuples containing the mean and standard deviation of each window.
        """
        num_windows = len(arr)-window_size+1
        mean_stds = np.zeros([num_windows, 2])
        for num, window in enumerate(self.sliding_window(arr, window_size)):
            mean_stds[num, 0] = np.mean(window)
            mean_stds[num, 1] = np.std(window)
        return np.array(mean_stds)

    def get_all_sliding_cell_stat(self, fluoresence, window_size=30*60, parallel=True, processes=16, mode="mean"):
        """
        Calculate the mean and standard deviation of sliding window (default: 30*60 = 1 sec.) fluorescence for each cell.

        Args:
            fluoresence (numpy.ndarray): A 3D numpy array containing fluorescence data for each cell.

        Returns:
            numpy.ndarray: A (cells, frames, 2) Dimensional numpy array containing the mean [:,:,0] and 
            standard deviation [:,:,1] of fluorescence for each cell.

        Example:
            means = np.array(get_all_sliding_cell_stat)[:,:,0]
            stds = np.array(get_all_sliding_cell_stat)[:,:,1]
        """
        if mode=="mean":
            get_all_sliding_cell_stat = parmap.map(self.sliding_mean_std, fluoresence, window_size, pm_processes=processes, 
                                    pm_pbar=True, pm_parallel=parallel)
        elif mode=="mode":
            get_all_sliding_cell_stat = parmap.map(self.sliding_mode_std, fluoresence, window_size, pm_processes=processes, 
                                    pm_pbar=True, pm_parallel=parallel)
        return get_all_sliding_cell_stat

class Session:
    def __init__(self, animal_id, session_id, generate=False, regenerate=False, units="all", delete=False, age=None, session_date=None) -> None:
        print(f"Loading session: {animal_id} {session_id}")
        self.animal_id = animal_id
        self.session_id = session_id
        self.session_date = session_date
        self.session_dir = os.path.join(Animal.root_dir, animal_id, session_id, Animal.dir_)
        self.calcium_object = None

        self.age = age
        
        self.mesc_data_path = self.get_mesc_data_path()
        self.session_parts = self.get_session_parts()  #TODO: WARNING! units could not start at 0
        self.tiff_data_paths = self.get_tiff_data_paths(generate=generate, regenerate=regenerate, units=units, delete=delete)
        self.s2p_folder_paths = self.get_s2p_folder_paths(generate=generate, regenerate=regenerate, units=units, delete=delete)

        self.cabincorr_data_paths = self.get_cabincorr_data_paths(generate=generate, regenerate=regenerate, units=units)
        #TODO: load suite2p files? how is RAM?
        #TODO: implement cabincorr functions for filtering correct data
        #self.corr_mean, self.corr_std = self.get_corr_mean_std()
        print(f"Finished {animal_id}: {session_id}")

    def get_mesc_data_path(self):
        # Search for MESC file names needed for TIFF creation
        files_list = get_files(self.session_dir, ending="mesc")
        for file_name in files_list:
            #TODO: Pipeline to get mesc_data_path not perfect
            if re.search("S1", file_name) == None and re.search("S2", file_name) == None and re.search("S3", file_name) == None:
                continue
            else:
                self.mesc_data_path = os.path.join(self.session_dir, file_name)
                return self.mesc_data_path
        return None

    def get_list_of_session_parts(self, file_name):
        session_parts = file_name.split(".")[0].split("_")[-1].split("-")
        return [session for session in session_parts if session[0]=="S"]
    
    def get_session_parts(self, file_name = None):
        session_parts_list = []
        if file_name == None:
            file_name = self.mesc_data_path

        session_parts_list = self.get_list_of_session_parts(file_name)
        
        tiff_session_parts = []
        tiff_files_list = get_files(self.session_dir, ending="tiff")
        for tiff_file_name in tiff_files_list:
            tiff_session_parts += self.get_list_of_session_parts(tiff_file_name)

        self.session_parts = list(np.unique(session_parts_list + tiff_session_parts))

        if file_name == None:
            self.session_parts = []

        return self.session_parts

    def get_tiff_data_paths(self, generate=False, regenerate=False, units="all", delete=False):

        tiff_data_paths = []
        self.tiff_data_paths = []
        if regenerate:
            if units == "single" or units == "all":
                for unit in self.session_parts:
                    tiff_data_paths.append(self.generate_tiff_from_mesc(regenerate=regenerate, units=unit, delete=delete))
            else:
                for unit in units:
                    tiff_data_paths.append(self.generate_tiff_from_mesc(regenerate=regenerate, units=unit, delete=delete))
            
        files_list = get_files(self.session_dir, ending="tiff")
        for file_name in files_list:
            tiff_data_paths.append(os.path.join(self.session_dir, file_name))
        
        self.tiff_data_paths = tiff_data_paths

        if generate:
            if units == "single" or units == "all":
                for unit in self.session_parts:
                    tiff_data_paths.append(self.generate_tiff_from_mesc(regenerate=regenerate, units=unit, delete=delete))
            else:
                for unit in units:
                    tiff_data_paths.append(self.generate_tiff_from_mesc(regenerate=regenerate, units=unit, delete=delete))
        self.tiff_data_paths = np.unique(tiff_data_paths).tolist()
        return self.tiff_data_paths

    def generate_tiff_from_mesc(self, units="all", delete=False, regenerate=False):
        if isinstance(units, str):
            units = [units]

        if units[0] == "all":
            tiff_file_name = mesc_file_name.replace('.mesc','.tif')
            units = self.get_session_parts()
        else:
            tiff_file_name = os.path.join(self.session_dir, f"{self.animal_id}_{self.session_id}_{Animal.dir_}_")
            for unit in units:
                tiff_file_name += unit + "-"
            tiff_file_name = tiff_file_name[:-1] + ".tiff"
        
            
        if tiff_file_name not in self.tiff_data_paths or regenerate:
            mesc_file_name = self.mesc_data_path

            if mesc_file_name == None:
                print("No MESC file found")
            else:
                # merging all mescs tiff
                print("Merging Mesc to Tiff...")
                

                sess_list = []
                for unit in units:
                    temp = unit.replace("S",'')
                    temp = 'MUnit_'+str(int(temp)-1) #TODO: error prone method, because units not equall to number of session
                    print ("session loaded: ", temp)
                    sess_list.append(temp)

                data = []
                with h5py.File(mesc_file_name, 'r') as file:
                    #
                    for sess in sess_list:
                        print ("processing: ", sess)
                        temp = file['MSession_0'][sess]['Channel_0'][()]
                        print ("    data loaded size: ", temp.shape)
                        data.append(temp)
                data = np.vstack(data)
                print(data.shape)

                tifffile.imwrite(tiff_file_name, data)
                if delete:
                    os.remove(mesc_file_name)
                print("Finished generating TIFF from MESC data.")
        else:
            print(".mesc -> .tiff file already done... skipping conversion...")

        return tiff_file_name

    def get_s2p_folder_paths(self, generate=False, regenerate=False, units="all", delete=False):
        self.s2p_folder_paths = []

        if regenerate:
            if units == "single":
                for unit in self.session_parts:
                    self.run_suite2p(regenerate=regenerate, units=unit, delete=delete)
            else:
                self.run_suite2p(regenerate=regenerate, units=units, delete=delete)

        dir_exist_create(os.path.join(self.session_dir, "tif"))
        s2p_folder_paths = get_directories(os.path.join(self.session_dir, "tif"))
        for folder_name in s2p_folder_paths:
            self.s2p_folder_paths.append(os.path.join(self.session_dir, "tif", folder_name))
        
        #FIXME: Check if this still working
        suite2p_folder = os.path.join(self.session_dir, "tif", "suite2p")
        if units == "all":
            fluoresence_path = search_file(suite2p_folder, "F.npy")
            if fluoresence_path != None:
                self.s2p_folder_paths.append(suite2p_folder)
            elif generate:
                self.run_suite2p(regenerate=regenerate, units=units, delete=delete)

        elif units == "single":
            for unit in self.session_parts:
                suite2p_single = suite2p_folder + unit
                fluoresence_path = search_file(suite2p_single, "F.npy")
                if fluoresence_path != None:
                    self.s2p_folder_paths.append(suite2p_folder)
                elif generate:
                    for unit in self.session_parts:
                        self.run_suite2p(regenerate=regenerate, units=unit, delete=delete)

        else: # custom session combination
            for unit in units:
                suite2p_folder += "_"+unit
            fluoresence_path = search_file(suite2p_folder, "F.npy")
            if fluoresence_path != None:
                self.s2p_folder_paths.append(suite2p_folder)
            elif generate:
                self.run_suite2p(regenerate=regenerate, units=unit, delete=delete)

        self.s2p_folder_paths = np.unique(self.s2p_folder_paths).tolist()
        return self.s2p_folder_paths

    """def get_s2p_folder_paths(self, generate=False, regenerate=False, units="all", delete=False):
        self.s2p_folder_paths = []

        if generate and regenerate:
            if units == "single":
                for unit in self.session_parts:
                    self.run_suite2p(regenerate=regenerate, units=unit, delete=delete)


        s2p_folder_paths = get_directories(os.path.join(self.session_dir, "tif"))
        for folder_name in s2p_folder_paths:
            self.s2p_folder_paths.append(os.path.join(self.session_dir, "tif", folder_name))
        
        for unit in self.session_parts:
            suite2p_folder = os.path.join(self.session_dir, "tif", "suite2p") + unit
            fluoresence_path = search_file(suite2p_folder, "F.npy")
            if fluoresence_path != None:
                self.s2p_folder_paths.append(suite2p_folder)
            else:
                if generate and not regenerate:
                    if units == "single":
                        for unit in self.session_parts:
                            self.run_suite2p(regenerate=regenerate, units=unit, delete=delete)
        return np.unique(self.s2p_folder_paths).tolist()"""

    def run_suite2p(self, regenerate=False, units="all", delete=False):
        save_folder="tif\\suite2p"
        tiff_file_name = f"{self.animal_id}_{self.session_id}_{Animal.dir_}"
        
        tiff_file_names = []
        if units != "all":
            if isinstance(units, str):
                units = [units]
            for unit in units:
                save_folder=save_folder + "_" + unit
                tiff_file_names.append(tiff_file_name+"_"+unit+".tiff")
        else:
            for unit in self.session_parts:
                tiff_file_names.append(tiff_file_name+"_"+unit+".tiff")

        dir_exist_create(os.path.join(self.session_dir, save_folder))

        current_fluoresence_data_path = search_file(os.path.join(self.session_dir, save_folder), "F.npy")
        if current_fluoresence_data_path != None and not regenerate:
            return current_fluoresence_data_path
        
        for tiff_file_name in tiff_file_names:
            data_path = os.path.join(self.session_dir, tiff_file_name)
            if data_path not in self.tiff_data_paths:
                print(f"Failed to run Suite2P \n No Tiff found: {data_path}")
                return None    

        print("Starting Suite2p...")
        # set your options for running
        ops = default_ops() # populates ops with the default options

        # provide an h5 path in 'h5py' or a tiff path in 'data_path'
        # db overwrites any ops (allows for experiment specific settings)
        db = {
            'batch_size': 500,      # we will decrease the batch_size in case low RAM on computer
            'fs': 30,               # sampling rate of recording, determines binning for cell detection
            'look_one_level_down': False, # whether to look in ALL subfolders when searching for tiffs
            'data_path': [self.session_dir], # a list of folders with tiffs 
                                    # (or folder of folders with tiffs if look_one_level_down is True, or subfolders is not empty)
            'save_folder': save_folder,
            #'threshold_scaling': 2.0, # we are increasing the threshold for finding ROIs to limit the number of non-cell ROIs found (sometimes useful in gcamp injections)
            #'tau': 1.25,           # timescale of gcamp to use for deconvolution
            #'nimg_init': 500,      # Can create errors... how many frames to use to compute reference image for registration
            'tiff_list': tiff_file_names,
            'allow_overlap': False,  #extract signals from pixels which belong to two ROIs. By default, any pixels which belong to two ROIs (overlapping pixels) are excluded from the computation of the ROI trace.
            'delete_bin': False,    # delete binary files afterwards
            'keep_movie_raw': False, # keep the binary file of the non-registered frames
            #'reg_tif': True,        # write the registered binary to tiff files
            'move_bin': True,       # If True and ops['fast_disk'] is different from ops[save_disk], the created binary file is moved to ops['save_disk']
            'save_disk': os.path.join(self.session_dir, save_folder) # Move the bin files to this location afterwards
            #'combined': False      # combine results across planes in separate folder “combined” at end of processing.
            }
        # run one experiment
        opsEnd = run_s2p(ops=ops, db=db)
        self.s2p_folder_paths.append(os.path.join(self.session_dir, save_folder, "plane0"))
        if delete:
            print("Removing Tiff...")
            os.remove(data_path)
        print("Finished Suite2p.")

    def get_cabincorr_data_paths(self, generate=False, regenerate=False, units="all"):
        self.cabincorr_data_paths = []

        if regenerate:
            if units == "single":
                for unit in self.session_parts:
                    self.run_cabincorr(regenerate=regenerate, units=unit)

        s2p_dirs = self.get_s2p_folder_paths()
        
        for s2pdir in s2p_dirs:
            cabincorr_file_path = search_file(s2pdir, Animal.cabincorr_file_name)
            if cabincorr_file_path != None:
                self.cabincorr_data_paths.append(cabincorr_file_path)

        if generate:
            if units == "single":
                for unit in self.session_parts:
                    self.run_cabincorr(regenerate=regenerate, units=unit)
            else:
                self.run_cabincorr(regenerate=regenerate, units=units)
        return self.cabincorr_data_paths
        
    def run_cabincorr(self, regenerate=False, units="all"):
        #TODO: create cabincorr package
        suite2p_folder = os.path.join(self.session_dir, "tif", "suite2p")

        if units != "all":
            if isinstance(units, str):
                units = [units]
            for unit in units:                
                suite2p_folder = suite2p_folder + "_" + unit

        current_cabincorr_data_path = search_file(suite2p_folder, Animal.cabincorr_file_name)

        if current_cabincorr_data_path != None and not regenerate:
            return current_cabincorr_data_path
        
        current_fluoresence_data_path = search_file(suite2p_folder, "F.npy")

        if current_fluoresence_data_path == None:
            print(f"Failed to run CaBinCorr \n No Suite2P data found: {suite2p_folder}")
            return None 

        print("Starting CaBinCorr...")
        #TODO: update code to newest version of cabincorr
        #Init
        c = calcium.Calcium()
        c.root_dir = Animal.root_dir
        c.data_dir = os.path.join(suite2p_folder, "plane0")
        c.animal_id = self.animal_id 
        c.session = self.session_id
        c.detrend_model_order = 1
        c.recompute_binarization = False
        c.remove_ends = False
        c.detrend_filter_threshold = 0.001
        c.mode_window = 30*30
        c.percentile_threshold = 0.000001
        c.dff_min = 0.02

        #
        c.load_suite2p()

        #
        c.load_binarization()
        current_cabincorr_data_path = search_file(suite2p_folder, Animal.cabincorr_file_name)
        self.cabincorr_data_paths.append(current_cabincorr_data_path)
        #TODO: Save every other information in parameters maybe save whole c object? Also for multiple version of suite2p data?
        #Generate correlations
        #self.calcium_object = c
        return current_cabincorr_data_path
    
    def load_cabincorr_data(self, units="all"):
        if units != "all":
            units_name = "_".join(units)
            for path in self.cabincorr_data_paths:
                try:
                    session_part = int(path.split("suite2p_")[1].split("\\")[0])
                    if units_name == session_part:
                        bin_traces_zip = np.load(path)
                except:
                    #TODO: create a better solution for filtering out folder without session path
                    print("No CaBincorrPath found")
                    continue
        else:
            bin_traces_zip = np.load(self.cabincorr_data_paths[0]) #TODO: will give false results if the correct folder is not choosen
        return bin_traces_zip
    
    def load_corr_matrix(self):
        #TODO: change after cabincorr package is finished
        corr_file_names = ["allcell_correlation_array_upphase.npy", "allcell_correlation_array_filtered.npy"]
        for corr_file_name in corr_file_names:
            for s2p_folder in self.s2p_folder_paths:
                corr_matrix_path = search_file(s2p_folder, corr_file_name)
                if corr_matrix_path != None:
                    break
            if corr_matrix_path != None:
                    break
        corr_pval_matrix = np.load(corr_matrix_path) # 1D correlation matrix, 2D pvalues
        corr_matrix = corr_pval_matrix[:,:,0]
        pval_matrix = corr_pval_matrix[:,:,1]
        return corr_matrix, pval_matrix

    def load_cell(self):
        #TODO: update to new cabincorr version
        return None

class Animal:
    root_dir = "F:\\Steffen_Experiments" 
    dir_ = r'002P-F'
    cabincorr_file_name = "binarized_traces.npz"
    
    def __init__(self, yaml_file_path) -> None:
        self.sessions = {}
        self.year, self.day_of_birth, self.animal_id, self.pdays, self.session_dates, self.session_names, self.sex = self.load_data(yaml_file_path)
        self.animal_dir = os.path.join(Animal.root_dir, self.animal_id)
        print(f"Loading animal: {self.animal_id}")

    def load_data(self, yaml_file_path):
        with open(yaml_file_path) as f:
            lines = f.readlines()
        pdays = []
        session_dates = []
        session_names = []
        skip = 0
        for num, line in enumerate(lines):
            if skip > 0:
                skip -= 1
                continue
            if "cohort_year" in line:
                cohort_year = int(lines[num+1].split("- ")[1])
            if "dob" in line:
                dob = line.split(": ")[1][1:-2].strip()
            if "name: D" in line:
                print(line)
                animal_id = line.split(": ")[1].strip()
            if "pdays" in line:
                pdays = self.get_array_from_text_list(lines[num+1:], "session_dates")
                pdays = [int(pday) for pday in pdays]
                skip = len(pdays)
            if "session_dates" in line:
                session_dates = self.get_array_from_text_list(lines[num+1:], "session_names")
                skip = len(session_dates)
            if "session_names" in line:
                session_names = self.get_array_from_text_list(lines[num+1:], "sex")
                skip = len(session_names)
            if "sex" in line:
                sex = line.split(": ")[1].strip()
        return cohort_year, dob, animal_id, pdays, session_dates, session_names, sex

    def get_array_from_text_list(self, text_list, stop_word = ""):
        """
        This function takes in a list of text strings and an optional stop word as arguments. It returns a filtered list of strings.
        
        :param text_list: A list of text strings to be filtered.
        :type text_list: list
        :param stop_word: An optional argument that specifies a word to stop the filtering process. Default is an empty string.
        :type stop_word: str
        :return: A filtered list of strings.
        :rtype: list
        """
        filtered_list = []
        for line in text_list:
            if stop_word not in line:
                value = line[1:].strip().replace("'", "")
                filtered_list.append(value)
            else:
                break
        return filtered_list

    def get_session_data(self, session_id, generate=False, regenerate=False, units="all", delete=False):
        yaml_file_index = self.session_names.index(session_id)
        session = Session(self.animal_id, session_id, generate=generate, regenerate=regenerate, 
                        units=units, delete=delete, age=self.pdays[yaml_file_index], 
                        session_date=self.session_dates[yaml_file_index])
        self.sessions[session_id] = session
        return session
           
    def get_overview(self):
        print("-----------------------------------------------")
        print(f"{self.animal_id} born: {self.day_of_birth} sex: {self.sex}")
        overview_df = pd.DataFrame(columns = ['session_name', 'date', 'P', 'suite2p_folder_paths'])#, 'duration [min]'])
        for session_id, session in self.sessions.items():
            overview_df.loc[len(overview_df)] = {'session_name': session_id, 'date': session.session_date, 'P':session.age, 'suite2p_folder_paths':session.s2p_folder_paths}
        print(overview_df)
        print("-----------------------------------------------")
        return overview_df

class Vizualizer:
    def __init__(self, animals={}, save_dir=Animal.root_dir):
        self.animals = animals
        self.save_dir = os.path.join(save_dir, "figures")
        dir_exist_create(self.save_dir)
        # Collor pallet for plotting
        self.colors = mlp.colormaps["rainbow"](range(0,300))

    def add_animal(self, animal):
        self.animals[animal.animal_id] = animal

    def create_colorsteps(self, min_value, max_value, max_color_number=300):
        """
        This function calculates the number of color steps between a minimum and maximum value.
        
        :param min_value: The minimum value in the range.
        :type min_value: int or float
        :param max_value: The maximum value in the range.
        :type max_value: int or float
        :param max_color_number: The maximum number of colors to use, defaults to 250.
        :type max_color_number: int, optional
        :return: The number of color steps between the minimum and maximum values.
        :rtype: int
        """
        value_diff = max_value-min_value if max_value-min_value != 0 else 1
        return round(max_color_number/(value_diff))

    def plot_colorsteps_example(self):
        # Colorexample
        for num, c in enumerate(self.colors):
            plt.plot([num, num], color=c, linewidth=2)

        handles = []
        for age in [0, 15, 30, 50, 75, 100, 125, 150, 180, 200, 220, 240]:
                handles.append(Line2D([0], [0], color=self.colors[age], linewidth=2, linestyle='-', label=f"Age {age}"))

        plt.legend(handles=handles)
        plt.show()

        #### save figures
    
    def bursts(self, animal_id, session_id, fluoresence_type="raw", num_cells="all", unit_id="all", dpi=300, fps="30"):

        #TODO: insert possibility to filter for good cells?
        #is_cells_ids = np.where(calcium_object.iscell==1)[0]
        #is_not_cells_ids = np.where(calcium_object.iscell==0)[0]
        #num_is_cells = is_cells_ids.shape[0] #get is cells
        #calcium_object.plot_traces(calcium_object.F_filtered, np.arange(num_is_cells))
    

        #for s2p_folder in self.animals[animal_id].sessions[session].s2p_folder_paths:
        bin_traces_zip = self.animals[animal_id].sessions[session_id].load_cabincorr_data(unit=unit_id)
        fluorescence = bin_traces_zip[f"F_{fluoresence_type}"]
        self.traces(fluorescence, animal_id, session_id, unit_id, num_cells, dpi, fps)
        return fluorescence

    def traces(self, fluorescence, animal_id, session_id, unit_id="all", num_cells="all", fit_line=False, dpi=300, fps="30",
               xlabel=f"seconds", 
               ylabel='Fluoresence based on Ca in Cell',
               title=f"Bursts from "):
        # plot fluorescence
        fluorescence = np.array(fluorescence)
        fluorescence = np.transpose(fluorescence) if len(fluorescence.shape)==2 else fluorescence
        plt.figure()
        plt.figure(figsize=(12, 7))
        if num_cells != "all":
            plt.plot(fluorescence[:, :int(num_cells)])
        else:
            plt.plot(fluorescence)

        if unit_id!="all":
            file_name = f"{animal_id}_{session_id}_Unit_{unit_id}"
        else:
            file_name = f"{animal_id}_{session_id}"


        seconds = 5
        num_frames = 30*seconds
        num_x_ticks = 50
        written_label_steps = 2

        x_time = [int(frame/num_frames)*seconds for frame in range(len(fluorescence)) if frame%num_frames==0] 
        steps = round(len(x_time)/(2*num_x_ticks))
        x_time_shortened = x_time[::steps]
        x_pos = np.arange(0, len(fluorescence), num_frames)[::steps] 
        
        x_labels = [time if num%written_label_steps==0 else "" for num, time in enumerate(x_time_shortened)]
        plt.xticks(x_pos, x_labels, rotation=40, fontsize=8)
        plt.title(title+f"{file_name}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        file_title = title.replace(" ", "_")

        if fit_line and len(fluorescence.shape)==1: #TODO: add to 2d data?
            #TODO: move calculations to Analyzer
            anz = Analyzer()
            slope, intercept = anz.get_linreg_slope_intercept(fluorescence)
            length = range(len(fluorescence))
            plt.plot(length, intercept+length*slope, color = "r")

        plt.savefig(os.path.join(self.save_dir, f"{file_title}{file_name}.png"),
                    dpi=dpi)
        plt.show()

    def save_rasters_fig(self, calcium_object, animal_id, session_id, unit_id="all"): #TODO: Update to classes
        #TODO: yes?
        show_rasters_savelocation = os.path.join(calcium_object.data_dir, "figures")
        show_rasters_savelocation_name = os.path.join(show_rasters_savelocation, "rasters.png")
        own_location_name = os.path.join(self.save_dir, f"Rasters_{animal_id}_{session_id}_Unit_{unit_id}.png")

        dir_exist_create(os.path.join(calcium_object.data_dir, "figures"))
        del_present_file(own_location_name)
        del_present_file(show_rasters_savelocation_name)

        calcium_object.show_rasters(save_image=True)

        #change picture location
        os.rename(show_rasters_savelocation_name, own_location_name)    

    def pearson_hist(self, animal_id, session_id, dpi=300, 
                                title = "Pearson Correlation and Histogram",
                                hist_title='Pearson Correlation Coefficient Histogram',
                                hist_xlabel="Coefficients combined in 0.1 size bins",
                                hist_ylabel="Number of coefficients in bin",
                                facecolor="tab:blue"):
        
        # Create a figure and two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

        corr_matrix, pval_matrix = self.animals[animal_id].sessions[session_id].load_corr_matrix()

        # First subplot
        sns.heatmap(corr_matrix, annot=False, cmap='YlGnBu', ax=ax1)
        ax1.set_xlabel("Neuron id")
        ax1.set_ylabel("Neuron id")
        ax1.set_title('Pearson Correlation Matrix')

        # Second subplot
        hist_data = corr_matrix if isinstance(corr_matrix, np.ndarray) else corr_matrix.to_numpy()
        sns.histplot(data=hist_data.flatten(), binwidth=0.1, ax=ax2, facecolor=facecolor)
        ax2.set_title(hist_title)
        ax2.set_xlabel(hist_xlabel)
        ax2.set_ylabel(hist_ylabel)
        plt.savefig(os.path.join(self.save_dir, title),
                    dpi=dpi)
        plt.show()
        return corr_matrix, pval_matrix

    def pearson_kde(self, filters=[], dpi=300):
        # Plot Kernel density Estimation
        filtered_animals = filter_animals(self.animals, filters)
        unique_sorted_ages, min_age, max_age = get_age_range(filtered_animals)
        colorsteps = self.create_colorsteps(min_age, max_age)
        
        plt.figure()
        plt.figure(figsize=(12, 7))
        
        for animal_id, animal in filtered_animals.items():
            for session_id, session in animal.sessions.items():
                age = session.age
                try:
                    corr_matrix, pval_matrix = session.load_corr_matrix()
                except:
                    continue
                sns.kdeplot(data=corr_matrix.flatten(), color=self.colors[(age-min_age)*colorsteps], linewidth=1)#, fill=True, alpha=.001,)#, hist_kws=dict(edgecolor="k", linewidth=2))
        handles = []
        line_plot_steps = 1
        if len(unique_sorted_ages) > 17:
            line_plot_steps = round(len(unique_sorted_ages)/17)

        for age in np.unique(unique_sorted_ages[::line_plot_steps]):
                handles.append(Line2D([0], [0], color=self.colors[(age-min_age)*colorsteps], linewidth=2, linestyle='-', label=f"Age {age}"))
        #handles=[Patch(color="tab:red", label="Bad=mean+sigma > 0.3"), Patch(color="tab:blue", label="Good=mean+sigma < 0.3")]
        plt.xlabel("Correlation")
        plt.ylabel("Frequency")
        plt.title(f"{filters} KDE of all cell correlations")
        plt.legend(handles=handles)
        plt.savefig(os.path.join(self.save_dir,f"All_Correlation_Coefficient_KDE_{filters}.png"), dpi=300)
        plt.show()

        # Plot Bars to compare 2 numbers
    
    def plot_means_stds(self, filters=[], dpi=300, x_tick_jumps = 4):
        mean_threshold = Analyzer.mean_threshold
        std_threshold = Analyzer.std_threshold

        filtered_animals = filter_animals(self.animals, filters)
        unique_sorted_ages, min_age, max_age = get_age_range(filtered_animals)
        colorsteps = self.create_colorsteps(0, len(filtered_animals))
        drawn_animal_ids = []
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        for number, (animal_id, animal) in enumerate(filtered_animals.items()):
            ages = []
            means = []
            stds = []
            for session_id, session in animal.sessions.items():
                try:
                    corr_matrix, pval_matrix = session.load_corr_matrix()
                except:
                    continue
                ages.append(session.age)
                means.append(np.mean(corr_matrix))
                stds.append(np.std(corr_matrix))
                drawn_animal_ids.append(animal_id)
            if animal_id in drawn_animal_ids:
                ax1.plot(ages, means, color=self.colors[number*colorsteps], marker=".")
                ax2.plot(ages, stds, color=self.colors[number*colorsteps], marker=".")

        age_labels = [str(age) if num%x_tick_jumps==0 else "" for num, age in enumerate(unique_sorted_ages)]
        unique_draws_animal_ids = np.unique(drawn_animal_ids)
        lines = [Line2D([0], [0], color=self.colors[number*colorsteps], linewidth=3, linestyle='-', label=unique_draws_animal_ids[number]) for number in range(len(unique_draws_animal_ids))]
        title = f"{filters}_Means_and_Standard_Deviations.png"


        ax1.axhline(y = mean_threshold, color = 'r', linestyle = '--', label="Mean Threshold")
        ax1.set_xticks(unique_sorted_ages, age_labels, rotation=40, ha='right', rotation_mode='anchor')
        ax1.set_xlabel("Age in days")
        ax1.set_ylabel("Mean")
        ax1.set_title(f'{filters} Means of correlations')
        mean_threshold_legend_object = Line2D([0], [0], color='r', linewidth=2, linestyle='--', label=f"Mean thr={mean_threshold}")
        ax1_handles= lines+[mean_threshold_legend_object]


        ax2.axhline(y=std_threshold, color = 'r', linestyle = '--', label="Std Threshold")
        ax2.set_xticks(unique_sorted_ages, age_labels, rotation=40, ha='right', rotation_mode='anchor')
        ax2.set_xlabel("Age in days")
        ax2.set_ylabel("Standard Deviation")
        ax2.set_title(f"{filters} Std of correlations")
        std_threshold_legend_object = Line2D([0], [0], color='r', linewidth=2, linestyle='--', label=f"Std thr={std_threshold}")
        ax2_handles= lines+[std_threshold_legend_object]

        
        ax1.legend(handles=ax1_handles)
        ax2.legend(handles=ax2_handles)
        plt.savefig(os.path.join(self.save_dir, title), dpi=300)
        plt.show()

    def plot_good_bad(self, filters=[]):
        filtered_animals = filter_animals(self.animals, filters)
        anz = Analyzer(filtered_animals)
        good, bad = anz.evaluate_datasets_count()
        plt.figure()
        plt.bar(1, bad, 1, label="Bad datasets: bad", color="red")
        plt.bar(2, good, 1, label="Good datasets: good", color="green")
        plt.title(f"bad_vs_good_{filters}")
        plt.xticks([1], [""])
        plt.ylabel("Count")
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, f"bad_vs_good_{filters}.png"))
        plt.show()

    def sanky_diagram(self):
        """
        Not implemented, only example
        """
        #TODO: implement
        #https://plotly.com/python/sankey-diagram/
        import plotly.graph_objects as go

        fig = go.Figure(go.Sankey(
            arrangement = "snap",
            node = {
                "label": ["A", "B", "C", "D", "E", "F"],
                "x": [0.2, 0.1, 0.5, 0.7, 0.3, 0.5],
                "y": [0.7, 0.5, 0.2, 0.4, 0.2, 0.3],
                'pad':10},  # 10 Pixels
            link = {
                "source": [0, 0, 1, 2, 5, 4, 3, 5],
                "target": [5, 3, 4, 3, 0, 2, 2, 3],
                "value": [1, 2, 1, 1, 1, 1, 1, 2]}))

        fig.show()
        pass

    def unit_footprints(self, unit, cmap=None):
        # plot footprints of a unit
        plt.figure()
        title = f"{unit.animal_id}_{unit.session_id}_MUnit_{unit.unit_id}"
        footprints = unit.footprints
        plt.title(f"{len(footprints)} footprints {title}")
        self.footprints(footprints, cmap=cmap)
        plt.savefig(os.path.join(self.save_dir, f"Footprints_{title}.png"), dpi=300)

    def footprints(self, footprints, cmap=None):
        # plot all footprints
        for footprint in footprints:
            idx = np.where(footprint==0)
            footprint[idx] = np.nan
            plt.imshow(footprint, cmap=cmap)
        plt.gca().invert_yaxis()

    def unit_contours(self, unit):
        # Plot Contours
        plt.figure(figsize=(10,10))
        title = f"{unit.animal_id}_{unit.session_id}_MUnit_{unit.unit_id}"
        contours = unit.contours
        self.contours(contours)
        plt.title(f"{len(contours)} contours {title}")
        plt.savefig(os.path.join(self.save_dir, f"Contours_{title}.png"), dpi=300)

    def contour_to_point(self, contour):
        x_mean = np.mean(contour[:, 0])
        y_mean = np.mean(contour[:, 1])
        return np.array([x_mean, y_mean])

    def contours(self, contours, color=None, plot_center=False, comment=""): #plot_contours_points
        for contour in contours:
            y_corr = contour[:, 0]
            x_corr = contour[:, 1]
            plt.plot(x_corr, y_corr, color = color)
            if plot_center:
                xy_mean = self.contour_to_point(contour)
                plt.plot(xy_mean[1], xy_mean[0], ".", color = color)
        plt.title(f"{len(contours)} Contours{comment}")

    def multi_contours(self, multi_contours, plot_center=False, colors=["red", "green", "blue", "yellow", "purple", "orange", "cyan"]):
        for contours, col in zip(multi_contours, colors):
            self.contours(contours, color=col, plot_center=plot_center)

    def multi_unit_contours(self, units, combination=None, plot_center=False, shift=False):
        """
        units : dict
        combination : list of dict keys
        """
        handles = []
        plot_contours = []
        plot_colors = []
        combination = list(units.keys()) if combination==None else combination
        colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan"]
        for (unit_id, unit), col in zip(units.items(), colors):
            if unit_id not in combination:
                continue
            good_cell_contours = np.array(unit.contours)[unit.cell_geldrying==False] if len(unit.cell_geldrying)==len(unit.contours) else np.array(unit.contours)
            #shift contours
            good_cell_contours = [good_cell_contour - unit.yx_shift for good_cell_contour in good_cell_contours] if shift else good_cell_contours
            plot_colors.append(col)
            plot_contours.append(good_cell_contours)
            shift_label = f" y: {unit.yx_shift[0]}  x: {unit.yx_shift[1]}" if shift else ""
            handles.append(Line2D([0], [0], color=col, linewidth=2, linestyle='-', label=f"MUnit: {unit_id}{shift_label}"))
        self.multi_contours(plot_contours, colors=plot_colors, plot_center=plot_center)
        plt.title(f"Contours for MUnits: {combination}")
        plt.legend(handles=handles, fontsize=20)
        shift_label = f"_shifted" if shift else ""
        plt.savefig(os.path.join(self.save_dir, f"Contours_MUnits_{combination}{shift_label}.png"), dpi=300)
        plt.show()

    def unit_fluorescence_good_bad(self, unit, batch_size=10, starting=0, interactive=False, plot_duplicates=True):
        
        cell_geldrying = unit.get_geldrying_cells()
        fluoresence = unit.fluoresence

        title = f"{unit.animal_id}_{unit.session_id}_MUnit_{unit.unit_id}"

        if not plot_duplicates:
            if isinstance(unit.dedup_cell_ids, np.ndarray):
                cell_geldrying = cell_geldrying[unit.dedup_cell_ids]
                fluoresence = fluoresence[unit.dedup_cell_ids]

        cell_geldrying = cell_geldrying[starting:]
        fluoresence = fluoresence[starting:]
        cell_geldrying_batches = split_array(cell_geldrying, batch_size)
        fluoresence_batches = split_array(fluoresence, batch_size)
        num_batches = len(fluoresence_batches)

        for i, (cell_geldrying_batch, fluoresence_batch) in enumerate(zip(cell_geldrying_batches, fluoresence_batches)):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))

            for num, (cell_geldrying, neuron_data) in enumerate(zip(cell_geldrying_batch, fluoresence_batch)):
                cell_number = (i)*batch_size + num if batch_size != "all" else num
                cell_number += starting
                if not cell_geldrying:
                    ax1.plot(neuron_data, label=f"Cell: {cell_number}")# {unit.cell_geldrying_reasons[cell_number]}")
                else:
                    ax2.plot(neuron_data, label=f"Cell: {cell_number}")# {unit.cell_geldrying_reasons[cell_number]}")

            bad = sum(cell_geldrying_batch)
            good = len(cell_geldrying_batch)-bad

            seconds = 5
            num_frames = 30*seconds
            x_pos = np.arange(0, len(neuron_data), num_frames)
            x_time = [int(frame/num_frames)*seconds for frame in range(len(neuron_data)) if frame%num_frames==0] 
            num_written_labels = round(len(x_time)/100)
            x_labels = [time if time%num_written_labels==0 else "" for time in x_time]
            ax1.set_xticks(x_pos, x_labels, rotation=40, fontsize=8)
            ax2.set_xticks(x_pos, x_labels, rotation=40, fontsize=8)

            batch_title = f"Batch_{i+1}_of_{num_batches}"
            legend_fontsize = 10
            fig.suptitle(f"F of {good+bad} Cells {title} {batch_title}", fontsize=20)
            ax1.set_title(f'Good Cells: {good}')
            ax1.set_ylim(bottom=-0.1, top=0.8)
            ax2.set_ylabel("F_filtered")
            ax2.set_xlabel("seconds")
            ax2.set_title(f'Bad Cells: {bad}')
            ax2.set_ylim(bottom=-0.1, top=0.8)

            if not batch_size == "all" and not batch_size > 16:
                ax1.legend(fontsize=legend_fontsize)
                ax2.legend(fontsize=legend_fontsize) 

            plt.savefig(os.path.join(self.save_dir, f"F_slide_{title}_{batch_title}.png"), dpi=300)
            plt.show()
            dir_exist_create(os.path.join(self.save_dir,"html"))
            #interactive html
            
            if interactive:
                mpld3.save_html(fig, os.path.join(self.save_dir, "html", f"F_slide_{title}_{batch_title}.html"))

    def binary_frames(self, frames, num_images_x=2):
        num_frames = frames.shape[0]
        fig, ax = plt.subplots(round(num_frames/num_images_x), num_images_x, figsize =(15, 15))
        fig.suptitle(f"Binary Frames", fontsize=20)
        for i, image in enumerate(frames):
            x = int(i/num_images_x)
            y = i%num_images_x
            ax[x, y].imshow(image)
            ax[x, y].invert_yaxis()
        plt.show()
class Unit:
    def __init__(self, suite2p_folder_path, session, unit_id):
        self.suite2p_folder_path = suite2p_folder_path
        self.animal_id = session.animal_id
        self.session_id = session.session_id
        self.session_dir = session.session_dir
        self.unit_id = unit_id
        #self.dedup_cell_ids = None
        self.c, self.contours, self.footprints = self.run_cabin_corr()
        self.dedup_cell_ids = None
        self.get_all_sliding_cell_stat = None
        self.fluoresence = butter_lowpass_filter(self.c.dff, cutoff=0.5, fs=30, order=2)
        self.cell_geldrying = None
        self.load_geldrying()
        self.cell_geldrying_reasons = None
        self.ops = self.define_ops()
        self.refImg = None
        self.yx_shift = [0, 0]
        

    def run_cabin_corr(self, deduplicate=False):
        #Merging cell footprints
        c = calcium.Calcium()
        c.root_dir = Animal.root_dir
        c.data_dir = os.path.join(self.suite2p_folder_path, "plane0")
        print(c.data_dir) #TODO: remove when finished
        c.animal_id = self.animal_id 
        c.session = self.session_id
        c.detrend_model_order = 1
        c.recompute_binarization = False
        c.remove_ends = False
        c.detrend_filter_threshold = 0.001
        c.mode_window = 30*30
        c.percentile_threshold = 0.000001
        c.dff_min = 0.02

        #
        c.load_suite2p()

        c.load_binarization()

        # getting contours and footprints
        c.load_footprints()
        contours = c.contours
        footprints = c.footprints
        return c, contours, footprints

    def get_geldrying_cells(self, mode="mean"):
        #detect gel_drying with sliding mean change. Too long increase of mean = bad
        #returns boolean list of cells, where True is a cell labeled as drying 
        if type(self.cell_geldrying) is np.ndarray:
            return self.cell_geldrying
        if type(self.get_all_sliding_cell_stat) is not np.ndarray:
            anz = Analyzer()
            self.get_all_sliding_cell_stat = anz.get_all_sliding_cell_stat(fluoresence=self.fluoresence, mode=mode)
        anz = Analyzer()
        self.cell_geldrying = np.full([len(self.get_all_sliding_cell_stat)], True)
        self.cell_geldrying_reasons = [""]*len(self.get_all_sliding_cell_stat)
        for i, mean_stds in enumerate(self.get_all_sliding_cell_stat):
            self.cell_geldrying[i], self.cell_geldrying_reasons[i] = anz.geldrying(mean_stds, mode=mode) 
        self.geldrying_to_npy()
        return self.cell_geldrying
    
    def geldrying_to_npy(self):
        fname = "cell_drying.npy"
        fpath = os.path.join(self.suite2p_folder_path, "plane0", fname)
        np.save(fpath, self.cell_geldrying)

    def load_geldrying(self):
        fname = "cell_drying.npy"
        fpath = os.path.join(self.suite2p_folder_path, "plane0", fname)
        try:
            self.cell_geldrying = np.load(fpath)
        except:
            self.cell_geldrying = None
        return self.cell_geldrying
    
    def get_reference_image(self, n_frames_to_be_acquired=1000, image_x_size=512, image_y_size=512):
        if self.refImg is None:
            b_loader = Binary_loader()
            frames = b_loader.load_binary_frames(self.suite2p_folder_path, n_frames_to_be_acquired=n_frames_to_be_acquired, image_x_size=image_x_size, image_y_size=image_y_size)
            self.refImg = register.compute_reference(frames, ops=self.ops)
        return self.refImg
    
    def define_ops(self):
        ops = register.default_ops()
        ops["nonrigid"] = False
        return ops
    
    def calc_yx_shift(self, refAndMasks, num_align_frames=1000, image_x_size=512, image_y_size=512):
        if self.yx_shift == [0, 0]:
            b_loader = Binary_loader()
            frames = b_loader.load_binary_frames(self.suite2p_folder_path, n_frames_to_be_acquired=num_align_frames, image_x_size=image_x_size, image_y_size=image_y_size)
            frames, ymax, xmax, cmax, ymax1, xmax1, cmax1, _ = register.register_frames(refAndMasks, frames, ops=self.ops)
        self.yx_shift = [np.mean(ymax), np.mean(xmax)]
        return self.yx_shift

class Binary_loader:
    def load_binary(self, data_path, n_frames_to_be_acquired, fname="data.bin", image_x_size=512, image_y_size=512):
        # load binary file from suite2p_folder from unit
        image_size=image_x_size*image_y_size
        fpath = os.path.join(data_path, "plane0", fname)
        binary = np.memmap(fpath,
                            dtype='uint16',
                            mode='r',
                            shape=n_frames_to_be_acquired*image_size)
        return binary
    
    def get_binary_frames(self, binary, n_frames_to_be_acquired, image_x_size=512, image_y_size=512):
        # load binary frames from binary
        image_size = image_x_size * image_y_size
        num_frames = round(len(binary)/image_size)
        n_frames_to_be_acquired = n_frames_to_be_acquired if n_frames_to_be_acquired < num_frames else num_frames
        frames = np.reshape(binary, [n_frames_to_be_acquired, image_x_size, image_y_size])
        return frames
    
    def load_binary_frames(self, data_path, n_frames_to_be_acquired, fname="data.bin", image_x_size=512, image_y_size=512):
        binary = self.load_binary(data_path, n_frames_to_be_acquired=n_frames_to_be_acquired, image_x_size=image_x_size, image_y_size=image_y_size)
        binary_frames = self.get_binary_frames(binary, n_frames_to_be_acquired=n_frames_to_be_acquired, image_x_size=image_x_size, image_y_size=image_y_size)
        return binary_frames.copy()
    
class Merger:
    def shift_stat_cells(self, stat, yx_shift, image_x_size=512, image_y_size=512):
        # stat files first value ist y-value second is x-value
        new_stat = copy.deepcopy(stat)

        for num, cell_stat in enumerate(new_stat):
            y_shifted = []
            for y in cell_stat["ypix"]:
                y_shifted.append(round(y-yx_shift[0]))
            cell_stat["ypix"] = np.array(y_shifted)
            
            x_shifted = []
            for x in cell_stat["xpix"]:
                x_shifted.append(round(x-yx_shift[1]))
            cell_stat["xpix"] = np.array(x_shifted)

            #center of cell_stat
            med = cell_stat["med"]
            med_shifted = [round(med[0]-yx_shift[0]), round(med[1]-yx_shift[1])]
            cell_stat["med"] = med_shifted
        return new_stat
    
    def correct_abroad_cell(self, cell_stat, image_x_size=512, image_y_size=512):
        """
        !!!!!!!DEPRECATED!!!!!!!!!!!!!!!!!!!
        correct cells, which are out of pixel range
        
        removes x,y pixels, and corresponding lam, overlap which are out of bound
        decreases npix, npix_soma count by number of removed pixels
        not adjusted (because not used for getting Fluoresence traces):
          med, compact, solidity, npix_norm, npix_norm_no_crop, neuropil_mask, radius, aspect_ratio
        """
        dimensions = ["ypix", "xpix"]
        pixel_attributes = ["ypix", "xpix", "lam", "overlap"]
        # filter for positive pixels y, x >= 0
        for dimension in dimensions:
            in_frame_pixels = np.where(cell_stat[dimension]>=0)[0]
            num_removed_pixels =  len(cell_stat[dimension])-len(in_frame_pixels)
            if num_removed_pixels == 0:
                continue
            for pixel_attribute in pixel_attributes:
                cell_stat[pixel_attribute] = cell_stat[pixel_attribute][in_frame_pixels]
            cell_stat["npix"] -= num_removed_pixels
            cell_stat["npix_soma"] -= num_removed_pixels

        # filter for positive pixels y, x <= image_x_size # < or <=
        for dimension in dimensions:
            max_pos = image_y_size if dimension=="ypix" else image_x_size
            in_frame_pixels = np.where(cell_stat[dimension]<max_pos)[0] 
            num_removed_pixels = len(cell_stat[dimension])-len(in_frame_pixels)
            if num_removed_pixels == 0:
                continue
            for pixel_attribute in pixel_attributes:
                cell_stat[pixel_attribute] = cell_stat[pixel_attribute][in_frame_pixels]
            cell_stat["npix"] -= num_removed_pixels
            cell_stat["npix_soma"] -= num_removed_pixels
        return cell_stat
    
    def merge_stat(self, units, best_unit, image_x_size=512, image_y_size=512):
        """
        shift and merge, deduplicate, stat files with best_unit as reference position
        """
        stats = psutil.virtual_memory()  # returns a named tuple
        available = getattr(stats, 'available')
        byte_to_gb = 1/1000000000
        available_ram_gb = available*byte_to_gb
        print("Setting Number of Batches according to free RAM")
        num_batches = 32
        num_batches_range = [16, 12, 4, 2, 1]
        ram_range = [16, 32, 64, 128]
        for batches, ram in zip(num_batches_range, ram_range):
            if available_ram_gb < ram:
                num_batches = batches
                break
        print(f"Available RAM: {round(available_ram_gb)}GB setting number of batches to {num_batches}")

        merged_footprints = best_unit.footprints
        merged_stat = best_unit.c.stat
        for unit_id, unit in units.items():
            if unit_id == best_unit.unit_id:
                continue
            shifted_unit_stat = self.shift_stat_cells(unit.c.stat, yx_shift=unit.yx_shift, image_x_size=image_x_size, image_y_size=image_y_size)
            shifted_footprints = self.stat_to_footprints(shifted_unit_stat)
            clean_cell_ids, merged_footprints = self.merge_deduplicate_footprints(merged_footprints, shifted_footprints, parallel=True, num_batches=num_batches)
            merged_stat = np.concatenate([merged_stat, shifted_unit_stat])[clean_cell_ids]
        merged_stat_no_abroad = self.remove_abroad_cells(merged_stat, units, image_x_size=image_x_size, image_y_size=image_y_size)
        return merged_stat_no_abroad
    
    def remove_abroad_cells(self, merged_stat, units, image_x_size=512, image_y_size=512):
        # removing out of bound cells 
        remove_cells = []
        for cell_num, cell in enumerate(merged_stat):
            abroad = False
            #check for every shift 
            for unit_id, unit in units.items():
                if abroad:
                    break
                yx_shift = unit.yx_shift
                for axis in ["ypix", "xpix"]:
                    shift = yx_shift[0] if axis=="ypix" else yx_shift[1]
                    max_location = image_y_size if axis=="ypix" else image_y_size
                    shifted = cell[axis]+shift

                    # check if cell is out of bound
                    if sum(shifted>=max_location)>0 or sum(shifted<0)>0:
                        abroad = True
                        break
            if abroad:
                remove_cells.append(cell_num)
                
        for abroad_cell in remove_cells[::-1]:
            merged_stat = np.delete(merged_stat, abroad_cell)
            print(f"removed cell {abroad_cell}")
        return merged_stat

    def merge_s2p_files(self, units, stat, ops):
        """
        Merges F, Fneu, spks, iscell from individual Units
        Does not merge the individual corrected stat files
        Does not merge ops
        """
        path = units[list(units.keys())[0]].suite2p_folder_path
        path = os.path.join(path, "plane0")
        merged_F = np.load(os.path.join(path, "F.npy"))
        merged_Fneu = np.load(os.path.join(path,   "Fneu.npy"))
        merged_spks = np.load(os.path.join(path,   "spks.npy"))
        merged_iscell = np.load(os.path.join(path, "iscell.npy"))
        for unit_id, unit in units.items():
            if unit_id == list(units.keys())[0]:
                continue
            path = unit.suite2p_folder_path
            path = os.path.join(path, "plane0")
            F =  np.load(os.path.join(path, "F.npy"))
            merged_F = np.concatenate([merged_F, F], axis=1)
            Fneu =  np.load(os.path.join(path, "Fneu.npy"))
            merged_Fneu = np.concatenate([merged_Fneu, Fneu], axis=1)
            spks =  np.load(os.path.join(path, "spks.npy"))
            merged_spks = np.concatenate([merged_spks, spks], axis=1)
            is_cell = np.load(os.path.join(path, "iscell.npy"))
            merged_iscell *= is_cell
        
        root = path.split("suite2p")[0]
        merged_s2p_path = os.path.join(root, "suite2p_merged")
        dir_exist_create(merged_s2p_path)
        merged_s2p_path = os.path.join(root, "suite2p_merged", "plane0")
        dir_exist_create(merged_s2p_path)

        np.save(os.path.join(merged_s2p_path, "F.npy"), merged_F)
        np.save(os.path.join(merged_s2p_path, "Fneu.npy"), merged_Fneu)
        np.save(os.path.join(merged_s2p_path, "spks.npy"), merged_spks)
        np.save(os.path.join(merged_s2p_path, "iscell.npy"), merged_iscell)

        np.save(os.path.join(merged_s2p_path, "stat.npy"), stat)
        np.save(os.path.join(merged_s2p_path, "ops.npy"), ops)
        return merged_F, merged_Fneu, merged_spks, merged_iscell
    
    def stat_to_footprints(self, stat, dims=[512, 512]):
        imgs = []
        for k in range(len(stat)):
            x = stat[k]['xpix']
            y = stat[k]['ypix']

            # save footprint
            img_temp = np.zeros((dims[0], dims[1]))
            img_temp[x, y] = stat[k]['lam']

            img_temp_norm = (img_temp - np.min(img_temp)) / (np.max(img_temp) - np.min(img_temp))
            imgs.append(img_temp_norm)

        imgs = np.array(imgs)

        footprints = imgs
        return footprints

    def find_overlaps1(self, ids, footprints):
        #
        intersections = []
        for k in ids:
            temp1 = footprints[k]
            idx1 = np.vstack(np.where(temp1 > 0)).T

            #
            for p in range(k + 1, footprints.shape[0], 1):
                temp2 = footprints[p]
                idx2 = np.vstack(np.where(temp2 > 0)).T
                res = array_row_intersection(idx1, idx2)

                #
                if len(res) > 0:
                    percent1 = res.shape[0] / idx1.shape[0]
                    percent2 = res.shape[0] / idx2.shape[0]
                    intersections.append([k, p, res.shape[0], percent1, percent2])
        #
        return intersections
        # this computes spatial overlaps between cells; doesn't take into account temporal correlations            
        print ("... computing cell overlaps ...")
        
        ids = np.array_split(np.arange(footprints.shape[0]), 30)

        if parallel:
            res = parmap.map(find_overlaps1,
                            ids,
                            footprints,
                            #c.footprints_bin,
                            pm_processes=n_cores,
                            pm_pbar=True)
        else:
            res = []
            for k in trange(len(ids)):
                res.append(find_overlaps1(ids[k],
                                            footprints,
                                            #c.footprints_bin
                                            ))

        df = make_overlap_database(res)
        return df

    def generate_batch_cell_overlaps(self, footprints, parallel=True, recompute_overlap=False, n_cores=16, num_batches=3):
        # this computes spatial overlaps between cells; doesn't take into account temporal correlations
            
        print ("... computing cell overlaps ...")
        
        ids = np.array_split(np.arange(footprints.shape[0]), 30)

        if parallel:
            batches = np.array_split(ids, num_batches)
            results = np.array([])
            for batch in batches:
                res = parmap.map(find_overlaps1,
                                batch,
                                footprints,
                                #c.footprints_bin,
                                pm_processes=n_cores,
                                pm_pbar=True)
                results = np.concatenate([results, res])
            res = results
        else:
            res = []
            for k in trange(len(ids)):
                res.append(find_overlaps1(ids[k],
                                            footprints,
                                            #c.footprints_bin
                                            ))
        df = make_overlap_database(res)
        return df

    def find_candidate_neurons_overlaps(self, df_overlaps, corr_array=None, deduplication_use_correlations=False, corr_max_percent_overlap=0.25, corr_threshold=0.3):

        dist_corr_matrix = []
        for index, row in df_overlaps.iterrows():
            cell1 = int(row['cell1'])
            cell2 = int(row['cell2'])
            percent1 = row['percent_cell1']
            percent2 = row['percent_cell2']

            if deduplication_use_correlations:

                if cell1 < cell2:
                    corr = corr_array[cell1, cell2, 0]
                else:
                    corr = corr_array[cell2, cell1, 0]
            else:
                corr = 0

            dist_corr_matrix.append([cell1, cell2, corr, max(percent1, percent2)])

        dist_corr_matrix = np.vstack(dist_corr_matrix)

        #####################################################
        # check max overlap
        idx1 = np.where(dist_corr_matrix[:, 3] >= corr_max_percent_overlap)[0]
        
        # skipping correlations is not a good idea
        #   but is a requirement for computing deduplications when correlations data cannot be computed first
        if deduplication_use_correlations:
            idx2 = np.where(dist_corr_matrix[idx1, 2] >= corr_threshold)[0]   # note these are zscore thresholds for zscore method
            idx3 = idx1[idx2]
        else:
            idx3 = idx1

        #
        candidate_neurons = dist_corr_matrix[idx3][:, :2]

        return candidate_neurons

    def make_correlated_neuron_graph(self, num_cells, candidate_neurons):
        adjacency = np.zeros((num_cells, num_cells))
        for i in candidate_neurons:
            adjacency[int(i[0]), int(i[1])] = 1

        G = nx.Graph(adjacency)
        G.remove_nodes_from(list(nx.isolates(G)))

        return G

    def delete_duplicate_cells(self, num_cells, G, corr_delete_method='highest_connected_no_corr'):
        # delete multi node networks
        #
        if corr_delete_method=='highest_connected_no_corr':
            connected_cells, removed_cells = del_highest_connected_nodes_without_corr(G)
        # 
        print ("Removed cells: ", len(removed_cells))
        clean_cells = np.delete(np.arange(num_cells),
                                removed_cells)

        #
        clean_cell_ids = clean_cells
        removed_cell_ids = removed_cells
        connected_cell_ids = connected_cells

        return clean_cell_ids

    def merge_deduplicate_footprints(self, footprints1, footprints2, parallel=True, num_batches=4):
        merged_footprints = np.concatenate([footprints1, footprints2])
        num_cells = len(merged_footprints)

        df_overlaps = self.generate_batch_cell_overlaps(merged_footprints, recompute_overlap=True, parallel=parallel, num_batches=num_batches)
        candidate_neurons = self.find_candidate_neurons_overlaps(df_overlaps, corr_array=None, deduplication_use_correlations=False, corr_max_percent_overlap=0.25, corr_threshold=0.3)
        G = self.make_correlated_neuron_graph(num_cells, candidate_neurons)
        clean_cell_ids = self.delete_duplicate_cells(num_cells, G)
        cleaned_merged_footprints = merged_footprints[clean_cell_ids]
        return clean_cell_ids, cleaned_merged_footprints
    
    def shift_update_unit_s2p_files(self, unit, new_stat, image_x_size=512, image_y_size=512, deduplicate = False):
        data_path = os.path.join(unit.suite2p_folder_path, "plane0")
        # shift merged mask
        shift_to_unit = np.array([-1, -1]) * unit.yx_shift
        shifted_unit_stat = self.shift_stat_cells(new_stat, yx_shift=shift_to_unit, image_x_size=image_x_size, image_y_size=image_y_size)

        backup_s2p_files(data_path, note="backup")
        update_s2p_files(data_path, shifted_unit_stat)