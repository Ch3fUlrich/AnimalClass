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
from matplotlib.patches import Rectangle

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

    def evaluate_datasets_count(self, animals=None, generate_corr=False, remove_geldrying=True):
        good = 0
        bad = 0
        if animals == None:
            animals = self.animals
        for animal_id, animal in animals.items():
            try:
                for session_id, session in animal.sessions.items():
                    corr_matrix, pval_matrix = session.load_corr_matrix(generate_corr=generate_corr, remove_geldrying=remove_geldrying)
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
                       num_not_bad_means=30*60*0.5):
        """
        Check if the mean of the data increases for 1.5 minutes without a 0.5 minutes break (30fps)

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
        Check if the mode of the data increases for 1.5 minutes without a 0.45 minutes break (30fps)

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

    def geldrying(self, m_stds, bad_minutes=1.5, not_bad_minutes=0.5, mode="mean"):
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

    def get_all_sliding_cell_stat(self, fluorescence, window_size=30*60, parallel=True, processes=16, mode="mean"):
        """
        Calculate the mean and standard deviation of sliding window (default: 30*60 = 1 sec.) fluorescence for each cell.

        Args:
            fluorescence (numpy.ndarray): A 3D numpy array containing fluorescence data for each cell.

        Returns:
            numpy.ndarray: A (cells, frames, 2) Dimensional numpy array containing the mean [:,:,0] and 
            standard deviation [:,:,1] of fluorescence for each cell.

        Example:
            means = np.array(get_all_sliding_cell_stat)[:,:,0]
            stds = np.array(get_all_sliding_cell_stat)[:,:,1]
        """
        if mode=="mean":
            get_all_sliding_cell_stat = parmap.map(self.sliding_mean_std, fluorescence, window_size, pm_processes=processes, 
                                    pm_pbar=True, pm_parallel=parallel)
        elif mode=="mode":
            get_all_sliding_cell_stat = parmap.map(self.sliding_mode_std, fluorescence, window_size, pm_processes=processes, 
                                    pm_pbar=True, pm_parallel=parallel)
        return get_all_sliding_cell_stat

class Session:
    def __init__(self, animal_id, session_id, generate=False, regenerate=False, unit_ids="all", delete=False, age=None, session_date=None) -> None:
        print(f"Loading session: {animal_id} {session_id}")
        self.animal_id = animal_id
        self.session_id = session_id
        self.session_date = session_date
        self.session_dir = os.path.join(Animal.root_dir, animal_id, session_id, Animal.dir_)
        self.calcium_object = None

        self.age = age
        
        # load session information
        self.mesc_data_path = self.get_mesc_data_path()
        self.session_parts = self.get_session_parts()
        self.tiff_data_paths = self.get_tiff_data_paths(generate=generate, regenerate=regenerate, unit_ids=unit_ids, delete=delete)
        self.s2p_folder_paths = self.get_s2p_folder_paths(generate=generate, regenerate=regenerate, unit_ids=unit_ids, delete=delete)
        self.units = None
        self.merged_unit = None
        self.cell_geldrying = None

        # load cabincorr data
        self.cabincorr_data_paths = self.get_cabincorr_data_paths(generate=generate, regenerate=regenerate, unit_ids=unit_ids)
        #TODO: implement cabincorr functions for filtering correct data
        self.cells = None
        #self.corr_mean, self.corr_std = self.get_corr_mean_std()
        print(f"Finished {animal_id}: {session_id}")

    def get_mesc_data_path(self):
        # Search for MESC file names needed for TIFF creation
        files_list = get_files(self.session_dir, ending="mesc")
        for file_name in files_list:
            #TODO: Pipeline to get mesc_data_path not perfect
            if re.search("S1", file_name) == None and re.search("S2", file_name) == None and re.search("S3", file_name) == None and re.search("S4", file_name) == None:
                continue
            else:
                self.mesc_data_path = os.path.join(self.session_dir, file_name)
                return self.mesc_data_path
        return None

    def get_list_of_session_parts(self, file_name):
        session_parts = file_name.split("\\")[-1].split(".")[0].split("_")[-1].split("-")
        return [session for session in session_parts if session[0]=="S" ]
    
    def get_session_parts(self, file_name = None):
        session_parts = []
        
        # get session parts from MESC file name if available
        if file_name == None:
            if self.mesc_data_path:
                session_parts = self.get_list_of_session_parts(self.mesc_data_path)
        else:
            session_parts = self.get_list_of_session_parts(file_name)
        
        # get session parts from TIF file names if MESC file not available
        if len(session_parts)==0:
            tiff_session_parts = []
            tiff_files_list = get_files(self.session_dir, ending="tiff")
            for tiff_file_name in tiff_files_list:
                tiff_session_parts += self.get_list_of_session_parts(tiff_file_name)

            session_parts = list(np.unique(tiff_session_parts))

        self.session_parts = session_parts
        return self.session_parts

    def get_tiff_data_paths(self, generate=False, regenerate=False, unit_ids="all", delete=False):
        delete = False #FIXME: Mesc is probably always usefull.
        tiff_data_paths = []
        self.tiff_data_paths = []
        if regenerate:
            if unit_ids == "single" or unit_ids == "all":
                for unit in self.session_parts:
                    tiff_data_paths.append(self.generate_tiff_from_mesc(regenerate=regenerate, unit_ids=unit, delete=delete))
            else:
                for unit in unit_ids:
                    tiff_data_paths.append(self.generate_tiff_from_mesc(regenerate=regenerate, unit_ids=unit, delete=delete))
            
        files_list = get_files(self.session_dir, ending="tiff")
        for file_name in files_list:
            tiff_data_paths.append(os.path.join(self.session_dir, file_name))
        
        self.tiff_data_paths = tiff_data_paths

        if generate:
            if unit_ids == "single" or unit_ids == "all":
                for unit in self.session_parts:
                    tiff_data_paths.append(self.generate_tiff_from_mesc(regenerate=regenerate, unit_ids=unit, delete=delete))
            else:
                for unit in unit_ids:
                    tiff_data_paths.append(self.generate_tiff_from_mesc(regenerate=regenerate, unit_ids=unit, delete=delete))
        self.tiff_data_paths = np.unique(tiff_data_paths).tolist()
        return self.tiff_data_paths

    def generate_tiff_from_mesc(self, unit_ids="all", delete=False, regenerate=False):
        fps = 30
        if isinstance(unit_ids, str):
            unit_ids = [unit_ids]

        if "all" in unit_ids:
            tiff_file_name = self.mesc_data_path.replace('.mesc','.tif')
            unit_ids = self.get_session_parts()
        else:
            tiff_file_name = os.path.join(self.session_dir, f"{self.animal_id}_{self.session_id}_{Animal.dir_}_")
            for unit in unit_ids:
                tiff_file_name += unit + "-"
            tiff_file_name = tiff_file_name[:-1] + ".tiff"
        
            
        if tiff_file_name not in self.tiff_data_paths or regenerate:
            mesc_file_name = self.mesc_data_path

            if mesc_file_name == None:
                print("No MESC file found")
            else:
                # merging all mescs tiff
                print("Merging Mesc to Tiff...")
                
                # Get MUnit number list of first Mescfile session MSession_0
                with h5py.File(mesc_file_name, 'r') as file:
                    munits = file[list(file.keys())[0]]
                    fluorescence_recording_session_numbers = []
                    for name, unit in munits.items():
                        # if recording has at least 5 minutes
                        if unit["Channel_0"].shape[0] > fps*60*5: 
                            unit_number = name.split("_")[-1]
                            fluorescence_recording_session_numbers.append(int(unit_number))

                # Get number_shift, if a bad fluorescence file was manually deleted
                # CAUTION: if deletion is done in the middle the correction will break
                biggest_session_number = max([int(unit.replace("S", ""))-1 for unit in self.session_parts])
                biggest_possible_session_number = max(fluorescence_recording_session_numbers)
                number_shift = biggest_session_number-biggest_possible_session_number

                sess_list = []
                for unit in unit_ids:
                    temp = unit.replace("S",'')
                    temp = 'MUnit_'+str(int(temp)-1-number_shift)
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

    def get_s2p_folder_paths(self, generate=False, regenerate=False, unit_ids="all", delete=False):
        self.s2p_folder_paths = []

        if regenerate:
            if unit_ids == "single":
                for unit in self.session_parts:
                    self.run_suite2p(regenerate=regenerate, unit_ids=unit, delete=delete)
            else:
                self.run_suite2p(regenerate=regenerate, unit_ids=unit_ids, delete=delete)

        dir_exist_create(os.path.join(self.session_dir, "tif"))
        s2p_folder_paths = get_directories(os.path.join(self.session_dir, "tif"))
        for folder_name in s2p_folder_paths:
            self.s2p_folder_paths.append(os.path.join(self.session_dir, "tif", folder_name))
        
        suite2p_folder = os.path.join(self.session_dir, "tif", "suite2p")
        if unit_ids == "all":
            fluorescence_path = search_file(suite2p_folder, "F.npy")
            if fluorescence_path != None:
                self.s2p_folder_paths.append(suite2p_folder)
            elif generate:
                self.run_suite2p(regenerate=regenerate, unit_ids=unit_ids, delete=delete)

        elif unit_ids == "single":
            for unit in self.session_parts:
                suite2p_single = suite2p_folder + unit
                fluorescence_path = search_file(suite2p_single, "F.npy")
                if fluorescence_path != None:
                    self.s2p_folder_paths.append(suite2p_folder)
                elif generate:
                    for unit in self.session_parts:
                        self.run_suite2p(regenerate=regenerate, unit_ids=unit, delete=delete)

        else: # custom session combination
            for unit in unit_ids:
                suite2p_folder += "_"+unit
            fluorescence_path = search_file(suite2p_folder, "F.npy")
            if fluorescence_path != None:
                self.s2p_folder_paths.append(suite2p_folder)
            elif generate:
                self.run_suite2p(regenerate=regenerate, unit_ids=unit, delete=delete)
        
        if delete:
            print("Removing Tiff...")
            for data_path in self.tiff_data_paths:
                os.remove(data_path)

        self.s2p_folder_paths = np.unique(self.s2p_folder_paths).tolist()
        return self.s2p_folder_paths

    def run_suite2p(self, regenerate=False, unit_ids="all", delete=False):
        
        # Binary data from Suite2p for all MUnits is not needed for merging
        delete_bin = True if unit_ids=="all" else False
        move_bin = False if unit_ids=="all" else True

        save_folder=os.path.join("tif", "suite2p")
        tiff_file_name = f"{self.animal_id}_{self.session_id}_{Animal.dir_}"
        
        tiff_file_names = []
        if unit_ids != "all":
            if isinstance(unit_ids, str):
                unit_ids = [unit_ids]
            for unit in unit_ids:
                save_folder=save_folder + "_" + unit
                tiff_file_names.append(tiff_file_name+"_"+unit+".tiff")
        else:
            for unit in self.session_parts:
                tiff_file_names.append(tiff_file_name+"_"+unit+".tiff")

        dir_exist_create(os.path.join(self.session_dir, save_folder))

        current_fluorescence_data_path = search_file(os.path.join(self.session_dir, save_folder), "F.npy")
        if current_fluorescence_data_path != None and not regenerate:
            return current_fluorescence_data_path
        
        for tiff_file_name in tiff_file_names:
            data_path = os.path.join(self.session_dir, tiff_file_name)
            if data_path not in self.tiff_data_paths:
                print(f"Failed to run Suite2P \n No Tiff found: {data_path}")
                return None    

        print("Starting Suite2p...")
        # set your options for running
        ops = default_ops() # populates ops with the default options

        # deleting binary file from old s2p run
        s2p_temp_binary_location = os.path.join(self.session_dir, "suite2p", "plane0", "data.bin")
        print(f"Deleting old binary file from {s2p_temp_binary_location}")
        #if os.path.exists(s2p_temp_binary_location):
        #    os.remove(s2p_temp_binary_location)

        s2p_binary_file = os.path.join(save_folder, "plane0", "data.bin")
        if os.path.exists(s2p_binary_file):
            shutil.copy(s2p_binary_file, s2p_temp_binary_location)
            print(f"Reusing binary file {s2p_binary_file}")

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
            'delete_bin': delete_bin,    # delete binary files afterwards
            'keep_movie_raw': False, # keep the binary file of the non-registered frames
            #'reg_tif': True,        # write the registered binary to tiff files
            'move_bin': move_bin,       # If True and ops['fast_disk'] is different from ops[save_disk], the created binary file is moved to ops['save_disk']
            'save_disk': os.path.join(self.session_dir, save_folder) # Move the bin files to this location afterwards
            #'combined': False      # combine results across planes in separate folder “combined” at end of processing.
            }
        # run one experiment
        opsEnd = run_s2p(ops=ops, db=db)
        self.s2p_folder_paths.append(os.path.join(self.session_dir, save_folder))
        self.s2p_folder_paths = np.unique(self.s2p_folder_paths).tolist()
        print("Finished Suite2p.")

    def get_cabincorr_data_paths(self, generate=False, regenerate=False, unit_ids="all"):
        self.cabincorr_data_paths = []

        if regenerate:
            if unit_ids == "single":
                for unit in self.session_parts:
                    self.run_cabincorr(regenerate=regenerate, unit_ids=unit)

        s2p_dirs = self.get_s2p_folder_paths()
        
        for s2pdir in s2p_dirs:
            cabincorr_file_path = search_file(s2pdir, Animal.cabincorr_file_name)
            if cabincorr_file_path != None:
                self.cabincorr_data_paths.append(cabincorr_file_path)

        if generate:
            if unit_ids == "single":
                for unit in self.session_parts:
                    self.run_cabincorr(regenerate=regenerate, unit_ids=unit)
            else:
                self.run_cabincorr(regenerate=regenerate, unit_ids=unit_ids)
        return self.cabincorr_data_paths
        
    def run_cabincorr(self, regenerate=False, unit_ids="all"):
        #TODO: create cabincorr package
        suite2p_folder = os.path.join(self.session_dir, "tif", "suite2p")

        if unit_ids != "all":
            if isinstance(unit_ids, str):
                unit_ids = [unit_ids]
            for unit in unit_ids:                
                suite2p_folder = suite2p_folder + "_" + unit

        current_cabincorr_data_path = search_file(suite2p_folder, Animal.cabincorr_file_name)

        if current_cabincorr_data_path != None and not regenerate:
            return current_cabincorr_data_path
        
        current_fluorescence_data_path = search_file(suite2p_folder, "F.npy")

        if current_fluorescence_data_path == None:
            print(f"Failed to run CaBinCorr \n No Suite2P data found: {suite2p_folder}")
            return None 

        print("Starting CaBinCorr...")
        c = run_cabin_corr(Animal.root_dir, data_dir=os.path.join(suite2p_folder, "plane0"),
                           animal_id=self.animal_id, session_id=self.session_id)
        

        current_cabincorr_data_path = search_file(suite2p_folder, Animal.cabincorr_file_name)
        self.cabincorr_data_paths.append(current_cabincorr_data_path)
        return current_cabincorr_data_path
    
    def load_cabincorr_data(self, unit_id="all"):
        bin_traces_zip = None
        for path in self.cabincorr_data_paths:
            path_unit = path.split("suite2p")[-1].split("\plane0")[0]
            if path_unit == "_"+unit_id or unit_id == "all" and len(path_unit)==0:
                if os.path.exists(path):
                    bin_traces_zip = np.load(path, allow_pickle=True)
                else:
                    print("No CaBincorrPath found")
        return bin_traces_zip
    
    def get_cells(self, merged=True):
        found = False
        s2p_path = None
        if merged:
            print(f"Searing for suite2p_merged folder...")
            for s2p_path in self.s2p_folder_paths:
                if "merged" in s2p_path:
                    found = True
                    print(f"Loading Cells from {s2p_path}") 
                    break
        if not found or merged == False:
            if merged == True:
                print(f"Path to suite2p_merged not found.")
            print(f"Searching for standard Suite2p folder...")
            for s2p_path in self.s2p_folder_paths:
                if s2p_path.split("suite2p")[-1] == "":
                    found = True
                    print(f"Loading Cells from standard Suite2P folder {s2p_path}")
                    break
        if not found:
            print(f"No matching Suite2p folder found")
            return None
        
        cell_fname = str(0)+".npz"
        cell_npz_path = search_file(s2p_path, cell_fname)
        if cell_npz_path:
            corr_path = search_file(s2p_path, cell_fname).split(cell_fname)[0]
            cells = {}
            for cell_fname in get_files(corr_path):
                cell_id = int(cell_fname.split(".npz")[0])
                cells[cell_id] = Cell(self.animal_id, self.session_id, cell_id, s2p_path)
            self.cells = cells
        else:
            print(f"{cell_fname} not found. Assuming no correlation data is present.")
        return self.cells

    def load_corr_matrix(self, unit_id="merged", generate_corr=False, remove_geldrying=True):
        """
        Loads the correlation matrix for the specified unit ID.

        This function first checks if the correlation matrix file exists for the specified unit ID. If it does, it loads the correlation matrix and p-value matrix from the file. If it does not exist, it loads the correlation data from individual cell.npz files and saves the correlation matrix and p-value matrix to a file.

        :param unit_id: The unit ID for which to load the correlation matrix. Can be "merged" or a specific unit ID. Defaults to "merged".
        :type unit_id: str
        :return: A tuple containing the correlation matrix and p-value matrix.
        :rtype: tuple
        """
        corr_matrix, pval_matrix = None, None
        merged = True if unit_id == "merged" else False
        s2p_folder_ending = "merged" if merged else unit_id
        s2p_folder_ending = "" if s2p_folder_ending == "all" else s2p_folder_ending
        for path in self.s2p_folder_paths:
            if path.split("suite2p")[-1] == s2p_folder_ending or path.split("suite2p")[-1] == "_"+s2p_folder_ending:
                break
        corr_matrix_path = os.path.join(path, "plane0", f"allcell_correlation_array.npy")
        
        if not os.path.exists(corr_matrix_path):
            if generate_corr:
                print("Loading correlation data from individual cell.npz files...")
                corr_matrix, pval_matrix = None, None
                cells = self.get_cells(merged)
                if type(cells) == dict:
                    pearson_corrs = []
                    pvalue_pearson_corrs = []
                    num_cells = len(cells)
                    for cell_id, cell in cells.items():
                        pearson_corrs = np.concatenate([pearson_corrs, cell.get_correlation_properties()["pearson_corr"]])
                        pvalue_pearson_corrs = np.concatenate([pvalue_pearson_corrs, cell.get_correlation_properties()["pvalue_pearson_corr"]])
                    corr_matrix = pearson_corrs.reshape([num_cells, num_cells])
                    pval_matrix = pvalue_pearson_corrs.reshape([num_cells, num_cells])
                    
                    print("Saving correlation matrix")
                    np.save(corr_matrix_path, (corr_matrix, pval_matrix))
            else:
                print("No correlation data. Returning None, None")
        else:
            print(f"Loading {corr_matrix_path}")
            corr_matrix, pval_matrix = np.load(corr_matrix_path)

        if remove_geldrying and unit_id == "merged" and type(corr_matrix)==np.ndarray:
            # removes geldrying cells in matrix with shape (#cell x #cells)
            geldrying = self.load_geldrying()
            geldrying_indexes = np.argwhere(geldrying==True).flatten()
            corr_matrix = remove_rows_cols(corr_matrix, geldrying_indexes, geldrying_indexes)
            pval_matrix = remove_rows_cols(pval_matrix, geldrying_indexes, geldrying_indexes)
            print("removed gelddrying cells")
        return corr_matrix, pval_matrix

    def load_geldrying(self):
        self.cell_geldrying = None
        fname = "cell_drying.npy"
        for s2p_path in self.s2p_folder_paths:
            if "merged" in s2p_path:
                fpath = os.path.join(s2p_path, "plane0", fname)
        if os.path.exists(fpath):
            self.cell_geldrying = np.load(fpath)
        else:
            print(f"File not found: {fpath}")
        return self.cell_geldrying

    def get_units(self, get_geldrying=False):
        units = {}
        for part in self.session_parts:
            not_session_parts = np.array(self.session_parts)[np.array(self.session_parts)!=part]
            unit_id = int(part[1]) 
            #if unit != 2:
            #    continue
            for s2p_folder_path in self.s2p_folder_paths:
                if part in s2p_folder_path:
                    single_unit = True
                    for other_part in not_session_parts:
                        if other_part in s2p_folder_path:
                            single_unit = False
                    if single_unit:
                        break
            data_path = os.path.join(s2p_folder_path, "plane0")
            backup_s2p_files(data_path, restore=True)
            backup_s2p_files(data_path, restore=False)
            unit = Unit(s2p_folder_path, session=self, unit_id=unit_id)
            backup_s2p_files(data_path, restore=False)
            num_good_cells = unit.print_s2p_iscell()
            if num_good_cells < 100: #If less than 100 good cells
                #Print amount of cells vs good cells
                print(f"Skipping Unit {unit.unit_id} (<100 cells)")    
            else:
                units[unit_id] = unit
                #single cells sliding mean detector for gel detection
                if get_geldrying:
                    cell_drying = unit.get_geldrying_cells()
                    bad = sum(unit.cell_geldrying)
                    good = len(unit.cell_geldrying)-bad
                    print(f"Autodetection Cells: {good+bad}    Good: {good}   geldrying:{bad} ")

        self.units = units
        return units
    
    def get_Unit_all(self):
        #create Unit for whole session
        data_path = None
        for s2p_folder_path in self.s2p_folder_paths:
            if s2p_folder_path.split("suite2p")[-1] == "" :
                data_path = s2p_folder_path
        if data_path:
            backup_s2p_files(data_path, restore=True)
            unit_all = Unit(data_path, session=self, unit_id="all")
        else:
            print("No s2p folder found for full session")
        return unit_all

    def get_most_good_cell_unit(self):
        most_good_cells = 0
        for unit_id, unit in self.units.items():
            num_good_cells = unit.num_not_geldrying()
            if num_good_cells > most_good_cells:
                most_good_cells = num_good_cells 
                best_unit = unit
        print(f"Best Mask has {most_good_cells} cells and is from {best_unit.unit_id}")
        return best_unit

    def get_usefull_units(self, min_num_usefull_cells):
        """
        This method updates the 'usefull' attribute of each unit in the 'units' dictionary and returns a dictionary of units that have more than 'min_num_usefull_cells' number of good cells.

        :param min_num_usefull_cells: The minimum number of good cells required for a unit to be considered useful.
        :type min_num_usefull_cells: int
        :return: A dictionary of useful units where the keys are unit IDs and the values are unit objects.
        :rtype: dict
        """
        for unit_id, unit in self.units.items():
            num_good_cells = unit.num_not_geldrying()
            if num_good_cells > min_num_usefull_cells:
                unit.usefull = True
            else:
                unit.usefull = False
        return {unit_id:unit for unit_id, unit in self.units.items() if unit.usefull}
    
    def calc_unit_yx_shifts(self, best_unit, units):
        """
        S2P Registration (Footprint position shift determination)
        """
        # caly yx_shift
        refImg = best_unit.get_reference_image(n_frames_to_be_acquired = 1000)
        #refImg = get_reference_image(best_unit)
        refAndMasks = register.compute_reference_masks(refImg, best_unit.ops)
        #refAndMasks = register.compute_reference_masks(refImg, ops)
        for unit_id, unit in units.items():
            if unit_id == best_unit.unit_id:
                continue   
            #unit.yx_shift = calc_yx_shift(refAndMasks, unit, unit.ops, num_align_frames)
            if unit.usefull:
                unit.calc_yx_shift(refAndMasks, num_align_frames=1000)

    def merge_units(self, generate=True, regenerate=False, delete_used_subsessions=False, image_x_size=512, image_y_size=512):
        """
        Takes MUnits with #cells> #most_cells/3 based on best MUnit (cells withoug geldrying).
        1. stat files are merged (suite2p) + deduplicated(cabincorr algo)
        2. Individual MUnit Suite2p folders are updated based on new Stat file (suite2p)
        3. Updated MUnits are merged to create full session data saved in suite2p_merged
        4. Gel drying is calculated for merged suite2p files

        :param generate: Whether to generate a new merged unit if it does not already exist. Defaults to True.
        :type generate: bool
        :param regenerate: Whether to regenerate the merged unit even if it already exists. Defaults to False.
        :type regenerate: bool
        :param image_x_size: The x size of the image. Defaults to 512.
        :type image_x_size: int
        :param image_y_size: The y size of the image. Defaults to 512.
        :type image_y_size: int
        :return: A merged unit object.
        :rtype: Unit
        """
        generate = True if regenerate==True else generate
        merged_s2p_path = os.path.join(self.s2p_folder_paths[0].split("suite2p")[0], "suite2p_merged")
        if os.path.exists(merged_s2p_path):
            if regenerate:
                shutil.rmtree(merged_s2p_path)
            else:
                merged_unit = Unit(merged_s2p_path, self, f"Already_merged")
                return merged_unit

        if generate:
            if not self.units:
                self.get_units(get_geldrying=True)
            # get unit with the most good cells (after geldrying detection)
            best_unit = self.get_most_good_cell_unit()
            # get units with enough usefull cells (at least 1/3 of best MUnit cells)
            min_num_usefull_cells = best_unit.num_not_geldrying() / 3
            units = self.get_usefull_units(min_num_usefull_cells)
            
            self.calc_unit_yx_shifts(best_unit, units)

            # merge statistical information of units and deduplicate
            merger = Merger()
            merged_stat = merger.merge_stat(units, best_unit)
            print(f"Number of cells after merging: {merged_stat.shape[0]}")

            updated_units = {} 
            merged_unit_id = ""
            for unit_id, unit in units.items():
                # shift merged mask
                print(f"Updating Unit {unit_id}")
                merger.shift_update_unit_s2p_files(unit, merged_stat, image_x_size=image_x_size, image_y_size=image_y_size)
                updated_units[unit_id] = Unit(unit.suite2p_folder_path, session=self, unit_id=unit_id)
                merged_unit_id += str(unit_id)+"_"
            # concatenate S2P results
            ops = default_ops()
            #FIXME: remove removed cells
            merged_F, _, _, _ = merger.merge_s2p_files(updated_units, merged_stat, ops) #best_unit.c.ops)
            #merged_F, merged_Fneu, merged_spks, merged_iscell = merger.merge_s2p_files(updated_units, merged_stat, best_unit.c.ops)

            if delete_used_subsessions:
                for unit_id, unit in updated_units.items():
                    shutil.rmtree(unit.suite2p_folder_path)

            self.get_cabincorr_data_paths()
            self.get_s2p_folder_paths()
            merged_unit = Unit(merged_s2p_path, self, f"{merged_unit_id}_merged")
            merged_unit.get_geldrying_cells()
            self.merged_unit = merged_unit
        return merged_unit
    
    def merge_units_get_geldrying(self, generate=True, regenerate=False, delete_used_subsessions=False, binary_needed=True):
        print(f"-------------------------------Generating Initial Suite2P Files for individual units ---------------------")
        self.get_s2p_folder_paths(generate=generate, regenerate=regenerate, unit_ids="single")
        self.get_cabincorr_data_paths(generate=generate, regenerate=regenerate, unit_ids="single")
        if binary_needed:
            print(f"-----------------------------------Rerun Suite2P if data.bin is missing-----------------------------------")
            # Rerunning Suite2p if binary file is not present
            bin_fname = "data.bin"
            for s2p_path in self.s2p_folder_paths:
                binary_file_present = False
                part_to_rerun = False
                for part in self.session_parts:
                    if part in s2p_path:
                        binary_file_present = os.path.exists(os.path.join(s2p_path, "plane0", bin_fname))
                        if binary_file_present:
                            break
                        else:
                            part_to_rerun = part
                    if part_to_rerun:
                        print(f"binary file not present in {s2p_path}")
                        self.run_suite2p(regenerate=True, unit_ids=part_to_rerun)
        print(f"-----------------------------------Loading Units-----------------------------------")
        self.get_units(get_geldrying=True)
        print(f"-----------------------------------Merging Units-----------------------------------")
        merged_unit = self.merge_units(generate=False, regenerate=regenerate, delete_used_subsessions=delete_used_subsessions)
        merged_unit.get_geldrying_cells()
        return merged_unit

class Cell:#(Session):
    def __init__(self, animal_id, session_id, cell_id, s2p_path):
        #super().__init__(animal_id, session_id, unit_ids=unit_ids)
        self.animal_id = animal_id
        self.session_id = session_id
        self.cell_id = cell_id
        self.s2p_path = s2p_path
        self.corr_path = search_file(s2p_path, str(cell_id)+".npz")
        self.corr_props = None 
        self.geldrying = None
        self.fluorescence = None
        self.num_bursts = None

        
    def get_correlation_properties(self):
        if not self.corr_props:
            self.corr_props = {}
            with np.load(self.corr_path) as corr_props:
                for key, value in corr_props.items():
                    self.corr_props[key] = value
        return self.corr_props

    def get_corr_prop_names(self):
        return list(self.get_correlation_properties().keys())

    def get_corr_pval(self):
        return np.array(self.get_correlation_properties()["pearson_corr"]), np.array(self.get_correlation_properties()["pvalue_pearson_corr"])

    def is_geldrying(self):
        if type(self.geldrying) != bool:
            geldrying_path = search_file(self.s2p_path, "cell_drying.npy")
            if geldrying_path:
                self.geldrying = np.load(geldrying_path)[self.cell_id]
            else:
                print(f"No cell_drying.npy file present")
        return self.geldrying

    def get_fluorescence(self):
        if type(self.fluorescence) != np.ndarray:
            fluorescence_path = search_file(self.s2p_path, "F.npy")
            self.fluorescence = np.load(fluorescence_path)[self.cell_id]
        return self.fluorescence
    
    def get_number_bursts(self):
        #TODO: 
        #detect the upphase of boolean time series from 0 to 1
        #upphase_bin = 
        #detected upphase is parameterized --> keep track of duration of bursts
        #firering rate 
        #firering_rate = # bursts/second
        #num_bursts
        #if num_burtsts < flavio_number:
        #    asdf
        #distribution of number of bursts or firering rate per session should be shown
        if type(num_bursts) != int:
            pass
        num_bursts = 0
        return num_bursts


class Animal:
    root_dir = os.path.join("F:", "Steffen_Experiments")
    dir_ = r'002P-F'
    cabincorr_file_name = "binarized_traces.npz"
    
    def __init__(self, yaml_file_path) -> None:
        self.sessions = {}
        self.year, self.day_of_birth, self.animal_id, self.pdays, self.session_dates, self.session_names, self.sex = self.load_data(yaml_file_path)
        self.animal_dir = os.path.join(Animal.root_dir, self.animal_id)
        print(f"Loading animal: {self.animal_id}")

    def load_data(self, yaml_path):
        #TODO: integrate variable loading of properties
        with open(yaml_path, "r") as yaml_file:
            animal_metadata_dict = yaml.safe_load(yaml_file)
        cohort_year = int(animal_metadata_dict["cohort_year"])
        dob = animal_metadata_dict["dob"]
        animal_id = animal_metadata_dict["name"]
        pdays = [int(pday) for pday in animal_metadata_dict["pdays"]]
        session_dates = animal_metadata_dict["session_dates"]
        session_names = animal_metadata_dict["session_names"]
        sex = animal_metadata_dict["sex"]

        return cohort_year, dob, animal_id, pdays, session_dates, session_names, sex

    def get_session_data(self, session_id, generate=False, regenerate=False, unit_ids="all", delete=False):
        yaml_file_index = self.session_names.index(session_id)
        session = Session(self.animal_id, session_id, generate=generate, regenerate=regenerate, 
                        unit_ids=unit_ids, delete=delete, age=self.pdays[yaml_file_index], 
                        session_date=self.session_dates[yaml_file_index])
        self.sessions[session_id] = session
        return session
           
    def get_overview(self):
        print("-----------------------------------------------")
        print(f"{self.animal_id} born: {self.day_of_birth} sex: {self.sex}")
        overview_df = pd.DataFrame(columns = ['session_name', 'date', 'P', 'suite2p_folder_paths'])#, 'duration [min]'])
        for session_id, session in self.sessions.items():
            overview_df.loc[len(overview_df)] = {'session_name': session_id, 'date': session.session_date, 'P':session.age, 'suite2p_folder_paths':session.s2p_folder_paths}
        display(overview_df)
        print("-----------------------------------------------")
        return overview_df

class Vizualizer:
    def __init__(self, animals={}, save_dir=Animal.root_dir):
        self.animals = animals
        self.save_dir = os.path.join(save_dir, "figures")
        dir_exist_create(self.save_dir)
        self.max_color_number = 301
        # Collor pallet for plotting
        self.colors = mlp.colormaps["rainbow"](range(0, self.max_color_number))

    def add_animal(self, animal):
        self.animals[animal.animal_id] = animal

    def create_colorsteps(self, min_value, max_value, max_color_number=None):
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
        if not max_color_number:
            max_color_number = self.max_color_number
        value_diff = max_value-min_value if max_value-min_value != 0 else 1
        return math.floor(max_color_number/(value_diff))

    def plot_colorsteps_example(self):
        # Colorexample
        for num, c in enumerate(self.colors):
            plt.plot([num, num], color=c, linewidth=2)

        handles = []
        for age in [0, 15, 30, 50, 75, 100, 125, 150, 180, 200, 220, 240]:
                handles.append(Line2D([0], [0], color=self.colors[age], linewidth=2, linestyle='-', label=f"Age {age}"))

        plt.legend(handles=handles)
        #plt.show()

        #### save figures
    
    def bursts(self, animal_id, session_id, fluorescence_type="F_raw", num_cells="all", unit_id="all", remove_geldrying=True, dpi=300, fps="30"):

        #TODO: insert possibility to filter for good cells?
        #is_cells_ids = np.where(calcium_object.iscell==1)[0]
        #is_not_cells_ids = np.where(calcium_object.iscell==0)[0]
        #num_is_cells = is_cells_ids.shape[0] #get is cells
        #calcium_object.plot_traces(calcium_object.F_filtered, np.arange(num_is_cells))

        #for s2p_folder in self.animals[animal_id].sessions[session].s2p_folder_paths:
        session = self.animals[animal_id].sessions[session_id]
        bin_traces_zip = session.load_cabincorr_data(unit_id=unit_id)
        fluorescence = None
        if bin_traces_zip:
            if fluorescence_type in list(bin_traces_zip.keys()):
                fluorescence = bin_traces_zip[fluorescence_type]
            else:
                print(f"{animal_id} {session_id} No fluorescence data of type {fluorescence_type} in binarized_traces.npz")
        else:
            print(f"{animal_id} {session_id} no binarized_traces.npz found")

        if remove_geldrying and unit_id == "merged" and type(fluorescence)==np.ndarray:
            geldrying = session.load_geldrying()
            geldrying_indexes = np.argwhere(geldrying==True).flatten()
            fluorescence = np.delete(fluorescence, geldrying_indexes, 0)
        
        if type(fluorescence)==np.ndarray:
            self.traces(fluorescence, animal_id, session_id, unit_id, num_cells, fluorescence_type=fluorescence_type, dpi=dpi)
        return fluorescence

    def traces(self, fluorescence, animal_id, session_id, unit_id="all", 
               num_cells="all", fluorescence_type="", low_pass_filter=True, dpi=300):
        # plot fluorescence
        if low_pass_filter:
            fluorescence = butter_lowpass_filter(fluorescence, cutoff=0.5, fs=30, order=2)
        
        fluorescence = np.array(fluorescence)
        fluorescence = np.transpose(fluorescence) if len(fluorescence.shape)==2 else fluorescence
        plt.figure()
        plt.figure(figsize=(12, 7))
        if num_cells != "all":
            plt.plot(fluorescence[:, :int(num_cells)])
        else:
            plt.plot(fluorescence)

        if unit_id!="all":
            file_name = f"{animal_id} {session_id} Unit {unit_id}"
        else:
            file_name = f"{animal_id} {session_id}"

        seconds = 5
        fps=30
        num_frames = fps*seconds
        num_x_ticks = 50
        written_label_steps = 2

        x_time = [int(frame/num_frames)*seconds for frame in range(len(fluorescence)) if frame%num_frames==0] 
        steps = round(len(x_time)/(2*num_x_ticks))
        x_time_shortened = x_time[::steps]
        x_pos = np.arange(0, len(fluorescence), num_frames)[::steps] 
        
        title = f"Bursts from {file_name} {fluorescence_type}"
        xlabel=f"seconds"
        ylabel='fluorescence based on Ca in Cell'
        x_labels = [time if num%written_label_steps==0 else "" for num, time in enumerate(x_time_shortened)]
        plt.xticks(x_pos, x_labels, rotation=40, fontsize=8)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.savefig(os.path.join(self.save_dir, title.replace(" ", "_")+".png"), dpi=dpi)
        plt.show()
        plt.close()

    def save_rasters_fig(self, calcium_object, animal_id, session_id, unit_id="all"): 
        #TODO: Update to classes
        show_rasters_savelocation = os.path.join(calcium_object.data_dir, "figures")
        show_rasters_savelocation_name = os.path.join(show_rasters_savelocation, "rasters.png")
        own_location_name = os.path.join(self.save_dir, f"Rasters_{animal_id}_{session_id}_Unit_{unit_id}.png")

        dir_exist_create(os.path.join(calcium_object.data_dir, "figures"))
        del_present_file(own_location_name)
        del_present_file(show_rasters_savelocation_name)

        calcium_object.show_rasters(save_image=True)

        #change picture location
        os.rename(show_rasters_savelocation_name, own_location_name)    

    def pearson_hist(self, animal_id, session_id, unit_id="all", remove_geldrying=True, dpi=300, generate_corr=False, color_classify=False,
                                facecolor="tab:blue"):
        title_unit_text = "Suite2P" if unit_id == "all" else unit_id  
        title = f"Corr_Hist {animal_id} {session_id} {title_unit_text}"
        unit_id = "" if unit_id=="all" else unit_id
        # Create a figure and two subplots
        session = self.animals[animal_id].sessions[session_id]
        corr_matrix, pval_matrix = session.load_corr_matrix(unit_id, generate_corr=generate_corr, remove_geldrying=True)
        if type(corr_matrix) == np.ndarray:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
            # First subplot
            sns.heatmap(corr_matrix, annot=False, cmap='YlGnBu', ax=ax1)
            ax1.set_xlabel("Neuron id")
            ax1.set_ylabel("Neuron id")
            ax1.set_title('Pearson Correlation Matrix')
    
            # Second subplot
            hist_data = corr_matrix if isinstance(corr_matrix, np.ndarray) else corr_matrix.to_numpy()
            mean = np.nanmean(corr_matrix)
            std = np.nanstd(corr_matrix)
            if color_classify:
                anz = Analyzer()
                corr_mean_std_good = anz.good_mean_std(mean, std)
                facecolor = facecolor if corr_mean_std_good else "tab:red" 
            sns.histplot(data=hist_data.flatten(), binwidth=0.1, ax=ax2, facecolor=facecolor)
            ax2.set_title("Pearson Correlation Coefficient Histogram")
            hist_xlabel="Coefficients combined in 0.1 size bins"
            hist_ylabel="Number of coefficients in bin"
            ax2.set_xlabel(hist_xlabel)
            ax2.set_ylabel(hist_ylabel)
            ax2.legend()

            mean_text = f'Mean: {mean:.2}'
            std_text = f'Std: {std:.2}'
            
            extra = Rectangle((0, 0), 1, 1, fc=facecolor, fill=True, edgecolor='none', linewidth=0)
            extra = Rectangle((0, 0), 1, 1, fc=facecolor, fill=True, edgecolor='none', linewidth=0)
            plt.legend([extra, extra],[mean_text, std_text], loc='upper right')#, title='Legend')


            fig.suptitle(title)
            plt.savefig(os.path.join(self.save_dir, title.replace(" ", "_")),
                        dpi=dpi)
            plt.show()
        else:
            print(f"No correlation data to be plotted")
        return corr_matrix, pval_matrix

    def pearson_kde(self, filters=[], unit_id="all", x_axes_range=[-0.5, 0.5], generate_corr=False, remove_geldrying=True, average_by_pday=False, dpi=300):
        if type(filters) != list:
            filters = [filters]
        title_unit_text = "Suite2P" if unit_id == "all" else unit_id  
        title = f"All correlation coefficient KDE {filters} {title_unit_text} {x_axes_range}"
        unit_id = "" if unit_id=="all" else unit_id
        # Plot Kernel density Estimation
        filtered_animals = filter_animals(self.animals, filters)
        unique_sorted_ages, min_age, max_age = get_age_range(filtered_animals)
        colorsteps = self.create_colorsteps(min_age, max_age)
        
        plt.figure()
        plt.figure(figsize=(12, 7))
        
        
        sum_corrs_by_pday = {}
        num_corrs = {}
        for animal_id, animal in filtered_animals.items():
            for session_id, session in animal.sessions.items():
                age = session.age
                corr_matrix, pval_matrix = session.load_corr_matrix(unit_id, generate_corr=generate_corr, remove_geldrying=remove_geldrying)
                if type(corr_matrix) != np.ndarray:
                    continue
                if not average_by_pday:
                    sns.kdeplot(data=corr_matrix.flatten(), color=self.colors[(age-min_age)*colorsteps], linewidth=1)#, fill=True, alpha=.001,)#, hist_kws=dict(edgecolor="k", linewidth=2))
                else:
                    if age not in sum_corrs_by_pday:
                        sum_corrs_by_pday[age] = corr_matrix.flatten()
                        num_corrs[age] = 1
                    else:
                        sum_corrs_by_pday[age] = np.append(sum_corrs_by_pday[age], corr_matrix)
                        num_corrs[age] += 1
        if average_by_pday:
            for age, sum_corrs in sum_corrs_by_pday.items():
                corr_matrix = sum_corrs/num_corrs[age]
                sns.kdeplot(data=corr_matrix, color=self.colors[(age-min_age)*colorsteps], linewidth=1)
        handles = []
        line_plot_steps = 1
        if len(unique_sorted_ages) > 17:
            line_plot_steps = round(len(unique_sorted_ages)/17)

        for age in np.unique(unique_sorted_ages[::line_plot_steps]):
            handles.append(Line2D([0], [0], color=self.colors[(age-min_age)*colorsteps], linewidth=2, linestyle='-', label=f"Age {age}"))
        #handles=[Patch(color="tab:red", label="Bad=mean+sigma > 0.3"), Patch(color="tab:blue", label="Good=mean+sigma < 0.3")]
        plt.xlabel("Correlation")
        plt.ylabel("Frequency")
        plt.title(title)
        plt.xlim(left=x_axes_range[0], right=x_axes_range[1])
        plt.legend(handles=handles)
        plt.savefig(os.path.join(self.save_dir,title.replace(" ", "_")+".png"), dpi=300)
        plt.show()

    def plot_means_stds(self, filters=[], unit_id="", dpi=300, x_tick_jumps = 4, generate_corr=False, remove_geldrying=True):
        if filters != list:
            filters = [filters]
        
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
                corr_matrix, pval_matrix = session.load_corr_matrix(unit_id, generate_corr=generate_corr, remove_geldrying=remove_geldrying)
                if type(corr_matrix) != np.ndarray:
                    continue
                ages.append(session.age)
                means.append(np.nanmean(corr_matrix))
                stds.append(np.nanstd(corr_matrix))
                drawn_animal_ids.append(animal_id)
            if animal_id in drawn_animal_ids:
                ax1.plot(ages, means, color=self.colors[number*colorsteps], marker=".")
                ax2.plot(ages, stds, color=self.colors[number*colorsteps], marker=".")

        age_labels = [str(age) if num%x_tick_jumps==0 else "" for num, age in enumerate(unique_sorted_ages)]
        unique_draws_animal_ids = np.unique(drawn_animal_ids)
        lines = [Line2D([0], [0], color=self.colors[number*colorsteps], linewidth=3, linestyle='-', label=unique_draws_animal_ids[number]) for number in range(len(unique_draws_animal_ids))]
        unit_text = "Suite2P" if unit_id=="all" else unit_id
        title = f"{filters}{unit_text} Means and Standard Deviations"
        fig.suptitle(title)

        ax1.axhline(y = mean_threshold, color = 'r', linestyle = '--', label="Mean Threshold")
        ax1.set_xticks(unique_sorted_ages, age_labels, rotation=40, ha='right', rotation_mode='anchor')
        ax1.set_xlabel("pday")
        ax1.set_ylabel("Mean")
        ax1.set_title(f'Means of pearson correlations')
        mean_threshold_legend_object = Line2D([0], [0], color='r', linewidth=2, linestyle='--', label=f"Mean thr={mean_threshold}")
        ax1_handles= lines+[mean_threshold_legend_object]


        ax2.axhline(y=std_threshold, color = 'r', linestyle = '--', label="Std Threshold")
        ax2.set_xticks(unique_sorted_ages, age_labels, rotation=40, ha='right', rotation_mode='anchor')
        ax2.set_xlabel("pday")
        ax2.set_ylabel("Standard Deviation")
        ax2.set_title(f"Std of pearson correlations")
        std_threshold_legend_object = Line2D([0], [0], color='r', linewidth=2, linestyle='--', label=f"Std thr={std_threshold}")
        ax2_handles= lines+[std_threshold_legend_object]

        
        ax1.legend(handles=ax1_handles)
        ax2.legend(handles=ax2_handles)
        plt.savefig(os.path.join(self.save_dir, title.replace(" ", "_")+".png"), dpi=300)
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
        #plt.show()

    def unit_fluorescence_good_bad(self, unit, batch_size=10, starting=0, interactive=False, plot_duplicates=True):
        
        cell_geldrying = unit.get_geldrying_cells()
        fluorescence = unit.fluorescence

        title = f"{unit.animal_id}_{unit.session_id}_MUnit_{unit.unit_id}"

        if not plot_duplicates:
            if isinstance(unit.dedup_cell_ids, np.ndarray):
                cell_geldrying = cell_geldrying[unit.dedup_cell_ids]
                fluorescence = fluorescence[unit.dedup_cell_ids]

        cell_geldrying = cell_geldrying[starting:]
        fluorescence = fluorescence[starting:]
        cell_geldrying_batches = split_array(cell_geldrying, batch_size)
        fluorescence_batches = split_array(fluorescence, batch_size)
        num_batches = len(fluorescence_batches)

        for i, (cell_geldrying_batch, fluorescence_batch) in enumerate(zip(cell_geldrying_batches, fluorescence_batches)):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))

            for num, (cell_geldrying, neuron_data) in enumerate(zip(cell_geldrying_batch, fluorescence_batch)):
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
            #plt.show()
            dir_exist_create(os.path.join(self.save_dir,"html"))
            #interactive html
            
            if interactive:
                mpld3.save_html(fig, os.path.join(self.save_dir, "html", f"F_slide_{title}_{batch_title}.html"))

    def binary_frames(self, frames, num_images_x=2):
        num_frames = frames.shape[0]
        num_rows = round(num_frames/num_images_x)
        fig, ax = plt.subplots(num_rows, num_images_x, figsize =(5*num_images_x, 5*num_rows))
        fig.suptitle(f"{num_frames} Binary Frames", fontsize=20)
        for i, image in enumerate(frames):
            x = int(i/num_images_x)
            y = i%num_images_x
            ax[x, y].imshow(image)
            ax[x, y].invert_yaxis()
        #plt.show()
    
    def show_survived_cell_percentage(self, animals=None, pipeline_stats=None):
        if type(pipeline_stats) != pd.DataFrame:
            if not animals:
                animals = self.animals
            else:
                raise ValueError("No data was given.")
            cell_numbers_dict = extract_cell_numbers(animals)
            # Create table to show statistics for comparison of S2P vs Own Pipeline
            pipeline_stats = summary_df_s2p_vs_geldrying(cell_numbers_dict)
        fig, (ax1) = plt.subplots(1, 1, figsize=(18, 4))
        ax1.bar(pipeline_stats.index, pipeline_stats.survived_cells)

        #fig.suptitle('Survived cell percentages')
        ax1.set_ylabel("% Cells")
        ax1.set_xlabel("Animal")
        ax1.set_title(f'Survived cell: {np.mean(pipeline_stats.survived_cells):.2%}')
        for i, v in enumerate(pipeline_stats.survived_cells):
            plt.text(range(len(pipeline_stats.index))[i] - 0.2, v + 0.01, f"{v:.2%}")
        plt.savefig(os.path.join(self.save_dir, f"Survived_cells_after_removing_geldrying.png"), dpi=300)

    def show_survived_cell_numbers(self, animals=None, cell_numbers_dict=None, min_num_cells=200):
        if not cell_numbers_dict:
            if not animals:
                animals = self.animals
            else:
                raise ValueError("No data was given.")
            cell_numbers_dict = extract_cell_numbers(animals)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        for animal_id, animal in cell_numbers_dict.items():
            ages, iscells, notgeldrying, corrs, gel_corrs = get_sorted_cells_notgeldyring_lists(animal)
            usefull_iscells = iscells >= min_num_cells
            usefull_notgeldrying = notgeldrying >= min_num_cells
            if len(usefull_iscells) > 0:
                ax1.plot(ages[usefull_iscells], iscells[usefull_iscells], label=f"{animal_id}", marker=".")
            if len(usefull_notgeldrying) > 0:
                ax2.plot(ages[usefull_notgeldrying], notgeldrying[usefull_notgeldrying], label=f"{animal_id}", marker=".", )

        title = f'Compare Cell Numbers before, after Geldrying Detector with at least {min_num_cells} Cells'
        fig.suptitle(title)
        ax1.set_ylabel("# Cells")
        ax1.set_xlabel("pday")
        ax1.set_ylim(bottom=0, top=1300)
        ax1.set_title('Suite2P iscell')
        ax1.legend()
        ax2.set_ylabel("# Cells")
        ax2.set_xlabel("pday")
        ax2.set_ylim(bottom=0, top=1300)
        ax2.set_title('Not Geldrying Cells (Own Pipeline)')
        ax2.legend()
        plt.savefig(os.path.join(self.save_dir, title.replace(" ", "_")+".png"), dpi=300)

    def show_usefull_sessions_comparisson(self, animals=None, cell_numbers_dict=None, min_num_cells=200, dpi=300, facecolor="tab:blue"):
        if not cell_numbers_dict:
            if not animals:
                animals = self.animals
            else:
                raise ValueError("No data was given.")
            cell_numbers_dict = extract_cell_numbers(animals)

        pday_usefull_session_count = {}
        for animal_id, animal in cell_numbers_dict.items():
            ages, iscells, notgeldrying, corrs, gel_corrs = get_sorted_cells_notgeldyring_lists(animal)
            for age, num_iscells, num_notgeldrying in zip(ages, iscells, notgeldrying):
                if age not in pday_usefull_session_count:
                    pday_usefull_session_count[age] = {"s2p" : 0, "notgeldrying" : 0}
                pday_usefull_session_count[age]["s2p"] += 1 if num_iscells >= min_num_cells else 0
                pday_usefull_session_count[age]["notgeldrying"] += 1 if num_notgeldrying >= min_num_cells else 0
        pdays = sorted(list(pday_usefull_session_count.keys()))
        pday_usefull_session_count = sorted(pday_usefull_session_count.items())

        s2p_count_list = []
        notgeldrying_count_list = []
        for age, counts in pday_usefull_session_count:
            s2p_count_list.append(counts["s2p"])
            notgeldrying_count_list.append(counts["notgeldrying"])
        xticks = range(15, max(pdays)+5, 5)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 4))
        title = f"Number of Sessions distributed across pdays with cell numbers > {min_num_cells}"
        fig.suptitle(title)
        minx, maxx = min(pdays)-1, max(pdays)+1
        ax1.set_title(f'{sum(s2p_count_list)} Sessions after Suite2P')
        ax1.bar(pdays, s2p_count_list)
        ax1.set_xlim(minx, maxx)
        ax1.set_ylabel("# Sessions")
        ax1.set_xlabel("pday")
        ax1.grid(color='gray', linestyle='-', linewidth=0.3)
        ax1.set_xticks(xticks, labels=xticks, rotation=40, ha='right', rotation_mode='anchor')
        miny, maxy = ax1.get_ylim()

        ax2.bar(pdays, notgeldrying_count_list)
        ax2.set_title(f'{sum(notgeldrying_count_list)} Sessions after removing geldrying cells')
        ax2.set_xlim(minx, maxx)
        ax2.set_ylim(miny, maxy)
        ax2.set_ylabel("# Sessions")
        ax2.set_xlabel("pday")
        ax2.set_xticks(xticks, labels=xticks, rotation=40, ha='right', rotation_mode='anchor')
        ax2.grid(color='gray', linestyle='-', linewidth=0.3)
        plt.savefig(os.path.join(self.save_dir, title.replace(" ", "_").replace(">","bigger than")+".png"), dpi=300)


        
class Unit:
    def __init__(self, suite2p_folder_path, session, unit_id):
        self.suite2p_folder_path = suite2p_folder_path
        self.animal_id = session.animal_id
        self.session_id = session.session_id
        self.session_dir = session.session_dir
        self.unit_id = unit_id
        self.c, self.contours, self.footprints = self.get_c()
        self.dedup_cell_ids = None
        self.get_all_sliding_cell_stat = None
        self.fluorescence = butter_lowpass_filter(self.c.dff, cutoff=0.5, fs=30, order=2)
        self.cell_geldrying = None
        self.load_geldrying()
        self.cell_geldrying_reasons = None
        self.ops = self.define_ops()
        self.refImg = None
        self.yx_shift = [0, 0]
        self.usefull = None
        

    def get_c(self):
        #Merging cell footprints
        c = run_cabin_corr(Animal.root_dir, data_dir=os.path.join(self.suite2p_folder_path, "plane0"),
                            animal_id=self.animal_id, session_id=self.session_id)
        contours = c.contours
        footprints = c.footprints
        return c, contours, footprints

    def get_geldrying_cells(self, bad_minutes = 1.5, not_bad_minutes=0.5, mode="mean"):
        #detect gel_drying with sliding mean change. Too long increase of mean = bad
        #returns boolean list of cells, where True is a cell labeled as drying 
        if type(self.cell_geldrying) is np.ndarray:
            return self.cell_geldrying
        if type(self.get_all_sliding_cell_stat) is not np.ndarray:
            anz = Analyzer()
            self.get_all_sliding_cell_stat = anz.get_all_sliding_cell_stat(fluorescence=self.fluorescence, mode=mode)
        anz = Analyzer()
        self.cell_geldrying = np.full([len(self.get_all_sliding_cell_stat)], True)
        self.cell_geldrying_reasons = [""]*len(self.get_all_sliding_cell_stat)
        for i, mean_stds in enumerate(self.get_all_sliding_cell_stat):
            self.cell_geldrying[i], self.cell_geldrying_reasons[i] = anz.geldrying(mean_stds, bad_minutes=bad_minutes, not_bad_minutes=not_bad_minutes, mode=mode) 
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
        self.yx_shift = [round(np.mean(ymax)), round(np.mean(xmax))]
        return self.yx_shift

    def print_s2p_iscell(self):
        iscell_path = search_file(self.suite2p_folder_path, "iscell.npy")
        iscell = np.load(iscell_path)
        num_cells = len(iscell[:, 0])
        num_good_cells = sum(iscell[:, 0])
        num_bad_cells = num_cells-num_good_cells
        print(f"Suite2p: Cells: {num_cells}  Good: {num_good_cells}  Bad: {num_bad_cells}")
        return num_good_cells

    def num_not_geldrying(self):
        return len(self.cell_geldrying)-sum(self.cell_geldrying)

class Binary_loader:
    """
    A class for loading binary data and converting it into an animation.

    This class provides methods for loading binary data from a file and converting a sequence of binary frames into an animated GIF. The `load_binary` method loads binary data from a specified file and returns it as a NumPy array. The `binary_frames_to_animation` method takes a sequence of binary frames and converts them into an animated GIF, which is saved to the specified directory.

    Attributes:
        None
    """
    def load_binary(self, data_path, n_frames_to_be_acquired, fname="data.bin", image_x_size=512, image_y_size=512):
        """
        Loads binary data from a file.

        This method takes the path of a binary data file as input, along with the number of frames to be acquired and the dimensions of each frame. It loads the binary data from the specified file and returns it as a NumPy array.

        Args:
            data_path (str): The path of the binary data file.
            n_frames_to_be_acquired (int): The number of frames to be acquired from the binary data file.
            fname (str): The name of the binary data file. Defaults to "data.bin".
            image_x_size (int): The width of each frame in pixels. Defaults to 512.
            image_y_size (int): The height of each frame in pixels. Defaults to 512.

        Returns:
            np.ndarray: A NumPy array containing the loaded binary data.
        """
        # load binary file from suite2p_folder from unit
        image_size=image_x_size*image_y_size
        fpath = search_file(data_path, fname)
        binary = np.memmap(fpath,
                            dtype='uint16',
                            mode='r',
                            shape=(n_frames_to_be_acquired, image_x_size, image_y_size))
        return binary
    
    def binary_frames_to_animation(frames, frame_range=[0, -1], save_dir="animation"):
        """
        Converts a sequence of binary frames into an animated GIF.

        This method takes a sequence of binary frames as input, along with the range of frames to include in the animation and the directory in which to save the resulting GIF. It converts the specified frames into an animated GIF and saves it to the specified directory.

        Args:
            frames (np.ndarray): A NumPy array containing the sequence of binary frames.
            frame_range (List[int]): A list specifying the range of frames to include in the animation. Defaults to [0, -1], which includes all frames.
            save_dir (str): The directory in which to save the resulting GIF. Defaults to "animation".

        Returns:
            animation.ArtistAnimation: An instance of `animation.ArtistAnimation` representing the created animation.
        """
        import matplotlib.animation as animation

        range_start, range_end = frame_range
        gif_save_path = os.path.join(save_dir, f"{range_start}-{range_end}.gif")

        images = []
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, frame in enumerate(frames):
            if i%100 == 0:
                print(i)
            p1 = ax.text(512/2-50, 0, f"Frame {i}", animated=True)
            p2 = ax.imshow(frame, animated=True)
            images.append([p1, p2])
        ani = animation.ArtistAnimation(fig, images, interval=50, blit=True,
                                        repeat_delay=1000)
        ani.save(gif_save_path)
        return ani
    
class Merger:
    """
    Merges indiviual MUnits/Subsession of a Session
    """
    def shift_stat_cells(self, stat, yx_shift, image_x_size=512, image_y_size=512):
        # stat files first value ist y-value second is x-value
        new_stat = copy.deepcopy(stat)

        for num, cell_stat in enumerate(new_stat):
            y_shifted = []
            for y in cell_stat["ypix"]:
                y_shifted.append(y-yx_shift[0])
            cell_stat["ypix"] = np.array(y_shifted)
            
            x_shifted = []
            for x in cell_stat["xpix"]:
                x_shifted.append(x-yx_shift[1])
            cell_stat["xpix"] = np.array(x_shifted)

            #center of cell_stat
            med = cell_stat["med"]
            med_shifted = [med[0]-yx_shift[0], med[1]-yx_shift[1]]
            cell_stat["med"] = med_shifted
        return new_stat
    
    def merge_stat(self, units, best_unit, parallel=True, image_x_size=512, image_y_size=512):
        """
        shift and merge, deduplicate, stat files with best_unit as reference position
        """
        num_batches = get_num_batches_based_on_available_ram()
        
        shifted_unit_stat_no_abroad = self.remove_abroad_cells(best_unit.c.stat, units, image_x_size=image_x_size, image_y_size=image_y_size)
        merged_footprints = self.stat_to_footprints(shifted_unit_stat_no_abroad)
        merged_stat = shifted_unit_stat_no_abroad
        for unit_id, unit in units.items():
            if unit_id == best_unit.unit_id:
                continue    
            shifted_unit_stat = self.shift_stat_cells(unit.c.stat, yx_shift=unit.yx_shift, image_x_size=image_x_size, image_y_size=image_y_size)
            shifted_unit_stat_no_abroad = self.remove_abroad_cells(shifted_unit_stat, units, image_x_size=image_x_size, image_y_size=image_y_size)
            shifted_footprints = self.stat_to_footprints(shifted_unit_stat_no_abroad)
            clean_cell_ids, merged_footprints = self.merge_deduplicate_footprints(merged_footprints, shifted_footprints, parallel=parallel, num_batches=num_batches)
            merged_stat = np.concatenate([merged_stat, shifted_unit_stat_no_abroad])[clean_cell_ids]
        return merged_stat
    
    def remove_abroad_cells(self, stat, units, image_x_size=512, image_y_size=512):
        # removing out of bound cells 
        remove_cells = []
        for cell_num, cell in enumerate(stat):
            abroad = False
            #check for every shift 
            for unit_id, unit in units.items():
                if abroad:
                    break
                yx_shift = unit.yx_shift
                for axis in ["ypix", "xpix"]:
                    shift = yx_shift[0] if axis=="ypix" else yx_shift[1]
                    shifted = cell[axis]+shift

                    # check if cell is out of bound
                    max_location = image_y_size if axis=="ypix" else image_x_size
                    if sum(shifted>=max_location)>0 or sum(shifted<0)>0:
                        abroad = True
                        break    
            if abroad:
                remove_cells.append(cell_num)
                
        for abroad_cell in remove_cells[::-1]:
            stat = np.delete(stat, abroad_cell)
            print(f"removed cell {abroad_cell}")
        return stat

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

    def generate_batch_cell_overlaps(self, footprints, parallel=True, recompute_overlap=False, n_cores=16, num_batches=3):
        # this computes spatial overlaps between cells; doesn't take into account temporal correlations
            
        print ("... computing cell overlaps ...")
        
        num_footprints = footprints.shape[0]
        num_min_cells_per_process = 10
        num_parallel_processes = 30 if num_footprints/30>num_min_cells_per_process else int(num_footprints/num_min_cells_per_process)
        ids = np.array_split(np.arange(num_footprints, dtype="int64"), num_parallel_processes)

        if num_batches > num_parallel_processes:
            num_batches = num_parallel_processes

        #TODO: will results in an error, if np.array_split is used on inhomogeneouse data like ids on Scicore
        batches = np.array_split(ids, num_batches) if num_batches!=1 else [ids]
        results = np.array([])
        num_cells = 0
        for batch in batches:
            res = parmap.map(find_overlaps1,
                            batch,
                            footprints,
                            #c.footprints_bin,
                            pm_processes=16,
                            pm_pbar=True,
                            pm_parallel=parallel)
            for cell_batch in res:
                num_cells += len(cell_batch)
                for cell in cell_batch:
                    results = np.append(results, cell)
        results = results.reshape(num_cells, 5)
        res = [results]
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
        shift_to_unit = np.array([-1]) * unit.yx_shift
        shifted_unit_stat = self.shift_stat_cells(new_stat, yx_shift=shift_to_unit, image_x_size=image_x_size, image_y_size=image_y_size)

        backup_s2p_files(data_path, note="backup")
        update_s2p_files(data_path, shifted_unit_stat)


def load_all(root_dir, wanted_animal_ids=["all"], wanted_session_ids=["all"], generate=False, regenerate=False, unit_ids="single", delete=False):
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
    present_animal_ids = get_directories(root_dir)
    animals_dict = {}

    # Search for animal_ids
    for animal_id in present_animal_ids:
        if animal_id in wanted_animal_ids or "all" in wanted_animal_ids:
            sessions_path = os.path.join(root_dir, animal_id)
            present_sessions = get_directories(sessions_path)
            yaml_file_name = os.path.join(root_dir, animal_id, f"{animal_id}.yaml")
            animal = Animal(yaml_file_name)
            Animal.root_dir = root_dir
            # Search for 2P Sessions
            for session in present_sessions:
                if session in wanted_session_ids or "all" in wanted_session_ids:
                    animal.get_session_data(session, generate=generate, regenerate=regenerate, unit_ids=unit_ids, delete=delete)
            animals_dict[animal_id] = animal
    return animals_dict

def run_cabin_corr(root_dir, data_dir, animal_id, session_id):
    #TODO: update code to newest version of cabincorr
    #Init
    c = calcium.Calcium(root_dir, animal_id, session_name=session_id, data_dir=data_dir)
    print(c.data_dir) #TODO: remove when finished

    c.animal_id = animal_id 
    c.detrend_model_order = 1
    c.recompute_binarization = False
    c.remove_ends = False
    c.detrend_filter_threshold = 0.001
    c.mode_window = 30*30
    c.percentile_threshold = 0.000001
    c.dff_min = 0.02
    c.data_type = "2p"
    #
    c.load_suite2p()

    c.load_binarization()

    # getting contours and footprints
    c.load_footprints()
    return c