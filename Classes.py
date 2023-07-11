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

# Used for Popups
import tkinter as tk

import nest_asyncio

import parmap
# for progress bar support
from tqdm import tqdm

# interact with system
import os
import sys

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
        Check if the mean of the data increases for 1.5 minutes without a 0.7 minutes break (30fps)

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
            if math.isnan(mean): #mean > max_plausible_mean or :
                bad = True
                reason = "nan"
                break
            if mean_diff > 0: #Analyzer.correct_mean + threshold:# or math.isnan(mean):
                bad_mean_counter += 1
            else:
                maybe_not_bad_counter += 1
                if maybe_not_bad_counter > num_not_bad_means:
                    bad_mean_counter = 0
                    maybe_not_bad_counter = 0

            if bad_mean_counter >= num_bad_means: # 1 minute wide window mean to high for 1 minute  
                bad = True
                reason = "num bad"
                break

        return bad, reason+" c: "+str(bad_mean_counter)+" not bad "+str(maybe_not_bad_counter)#, pos/30

    def all_stds_good(self, mean_stds, std_threshold = 2):
        """
        Check if the standard deviation of the data are within a certain threshold.

        Args:
            data (numpy.ndarray): A 2D numpy array containing mean and standard deviation values.

        Returns:
            bool: True if the standard deviation values are within the threshold, False otherwise.
        """
        bad = False
        # Check if the cell is active
        if np.nanmean(mean_stds[:, 1]) < Analyzer.correct_std/4: 
            bad = True
            return bad
        ## Check if cells are to noisy
        for mean_std in mean_stds:
            std = mean_std[1]
            if math.isnan(std) or abs(std) > Analyzer.correct_std + std_threshold:
                bad = True
                break
        return bad

    def geldrying(self, mean_stds, bad_minutes=1.5, not_bad_minutes=0.9):
        """
        Geldrying detection
        Check if the mean and standard deviation (std not used!!!!!!) of the data are within a certain threshold.

        Args:
            data (numpy.ndarray): A 2D numpy array containing mean and standard deviation values.

        Returns:
            bool: True if the standard deviation values are within the threshold, False otherwise.
        """
        #TODO: improve good bad detection currently only for geldrying used
        bad, reason = self.cont_mean_increase(mean_stds, num_bad_means = 30*60*bad_minutes, num_not_bad_means=30*60*not_bad_minutes) 
        return bad, reason
    
    def sliding_window(self, arr, window_size):
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
        for i in range(window_size, n):
            window = np.append(window[1:], arr[i])
            yield window

    def sliding_mode(self, arr, window_size):
        """
        Compute the mode for each sliding window of size k over an array.

        Args:
            arr (list): The input array.
            k (int): The size of the sliding window.

        Returns:
            list: A list of mode values for each sliding window.
        """
        modes = []
        for window in self.sliding_window(arr, window_size):
            mode, count = scipy.stats.mode(window)
            modes.append(mode[0])
        return modes

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

    def get_all_sliding_cell_F_mean_stds(self, fluoresence, window_size=30*60, parallel=True, processes=16):
        """
        Calculate the mean and standard deviation of sliding window (default: 30*60 = 1 sec.) fluorescence for each cell.

        Args:
            fluoresence (numpy.ndarray): A 3D numpy array containing fluorescence data for each cell.

        Returns:
            numpy.ndarray: A (cells, frames, 2) Dimensional numpy array containing the mean [:,:,0] and 
            standard deviation [:,:,1] of fluorescence for each cell.

        Example:
            means = np.array(sliding_cell_F_mean_stds)[:,:,0]
            stds = np.array(sliding_cell_F_mean_stds)[:,:,1]
        """
        sliding_cell_F_mean_stds = []
        F_mean_stds = parmap.map(self.sliding_mean_std, fluoresence, window_size, pm_processes=processes, 
                                pm_pbar=True, pm_parallel=parallel)
        
        sliding_cell_F_mean_stds.append(F_mean_stds)
        sliding_cell_F_mean_stds = np.array(sliding_cell_F_mean_stds)
        return sliding_cell_F_mean_stds

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
        else:
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
            'keep_movie_raw': True, # keep the binary file of the non-registered frames
            'reg_tif': True,        # write the registered binary to tiff files
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
