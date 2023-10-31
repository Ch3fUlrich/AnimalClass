# imports
# Statistics
import pandas as pd
import numpy as np

# Regular Expression searching
import re

# Used for Popups
import tkinter as tk

# for progress bar support
from tqdm import tqdm

# interact with system
import os
import sys
import shutil
import psutil
from datetime import datetime


# statistics
import scipy
import math

# Mesc file analysis
import pathlib
import h5py
# add root directory to be able to import packages
# todo: make all packages installable so they can be called/imported by environment
#module_path = os.path.abspath(os.path.join('../'))
#print(module_path)
#sys.path.append(module_path)
global old_stdout 

import time
from multiprocessing import Pool

def gif_to_mp4(path):
    """
    Converts a GIF file to an MP4 file.

    This function takes the path of a GIF file as input, converts it to an MP4 file, and saves the resulting MP4 file in the same directory as the input GIF file. The name of the output MP4 file is the same as the input GIF file, with the file extension changed from `.gif` to `.mp4`.

    Args:
        path (str): The path of the input GIF file.

    Returns:
        None
    """
    import moviepy.editor as mp
    clip = mp.VideoFileClip(path)
    save_path = path.replace('.gif', '.mp4')
    clip.write_videofile(save_path)

def show_mesc_units(path):
    h5 = h5py.File(path, 'r')
    for munit_id, MUnits in h5['MSession_0'].items():
        print(munit_id)
        print(MUnits['Channel_0'])

def yield_animal_session(animal_dict):
    """
    Yield animal_id, session_id, and session.

    Args:
        animal_dict (dict): A dictionary containing animal data.

    Yields:
        tuple: A tuple containing animal_id, session_id, and session.
    """
    for animal_id, animal in animal_dict.items():
        for session_id, session in animal.sessions.items():
            yield animal_id, session_id, session

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time}")
        return result
    return wrapper

def num_to_date(date_string):
    if type(date_string) != str:
        date_string = str(date_string)
    date = datetime.strptime(date_string, '%Y%m%d')
    return date

def get_num_batches_based_on_available_ram():
    stats = psutil.virtual_memory()  # returns a named tuple
    available = getattr(stats, 'available')
    byte_to_gb = 1/1000000000
    available_ram_gb = available*byte_to_gb
    print("Setting Number of Batches according to free RAM")
    num_batches = 16
    num_batches_range = [12, 8, 4, 2, 1]
    ram_range = [32, 16, 32, 64, 128]
    for batches, ram in zip(num_batches_range, ram_range):
        if available_ram_gb > ram:
            num_batches = batches
    print(f"Available RAM: {round(available_ram_gb)}GB setting number of batches to {num_batches}")
    return num_batches

def make_list_ifnot(string_or_list):
    return [string_or_list] if type(string_or_list) != list else string_or_list

def save_file_present(file_path):
    fname = file_path.split("\\")[-1]
    file_present = False
    if os.path.exists(file_path):
        print(f"File already present {file_path}")
        file_present = True
    else:
        print(f"Saving {fname} to {file_path}")
    return file_present

def find_binary_fpath(data_path, subdirectories=["data"], possible_binary_fnames=["data.bin", "Image_001_001.raw"]):
    """
    Searches for binary files in the specified data path and its subdirectories.

    Args:
        data_path (str): The path to the data directory.
        subdirectories (list, optional): A list of subdirectories to search for binary files. Defaults to ["data"].
        possible_binary_fnames (list, optional): A list of possible binary file names. Defaults to ["data.bin", "Image_001_001.raw"].

    Returns:
        str: The path to the binary file if found, else None.
    """
    subdirectories = make_list_ifnot(subdirectories)
    possible_binary_fnames = make_list_ifnot(possible_binary_fnames)
    binary_fpath = None
    possible_binary_data_paths = [data_path] + [os.path.join(data_path, subdirectory) for subdirectory in subdirectories]
    for possible_binary_data_path in possible_binary_data_paths:
        for possible_binary_fname in possible_binary_fnames:
            binary_file_path = os.path.join(possible_binary_data_path, possible_binary_fname)
            if os.path.exists(binary_file_path):
                binary_fpath = binary_file_path
                break
        if binary_fpath:
            break
    if not binary_fpath:
        print(f"No binary path to {possible_binary_fnames} found in {possible_binary_data_paths}")
    return binary_fpath

def remove_rows_cols(data, remove_rows, remove_cols):
    data = np.delete(data, remove_rows, 0)
    data = np.delete(data, remove_cols, 1)
    return data

def extract_cell_numbers(animals):
    """
    Extracts cell numbers from a dictionary of animals and their sessions.

    :param animals: A dictionary of animals, where the keys are animal IDs and the values are animal objects.
    :return: A dictionary where the keys are animal IDs and the values are dictionaries containing session IDs, ages, and cell numbers.
    """
    cell_numbers_dict = {}
    for animal_id, animal in animals.items():
        cell_numbers_dict[animal_id] = {}
        ages = []
        for session_id, session in animal.sessions.items():
            cell_numbers = {}
            cell_numbers["iscell"] = -1
            cell_numbers["not_geldrying"] = -1
            cell_numbers["corr"] = False
            cell_numbers["gel_corr"] = False
            if not session.suite2p_paths:
                print(f"No suite2p paths: {animal_id} {session_id}")
                continue
            for path in session.suite2p_paths:
                path_ending = path.split("suite2p")[-1]
                path = os.path.join(path, "plane0")
                iscell_path = os.path.join(path, "iscell.npy")
                cell_drying_path = os.path.join(path, "cell_drying.npy")
                corr_path = os.path.join(path, "allcell_corr_pval_zscore.npy")
                if path_ending=="":
                    cell_numbers["iscell"], cell_numbers["corr"] = get_cellnum_check_corr(iscell_path, corr_path)
                elif path_ending == "_merged":
                    cell_numbers["not_geldrying"], cell_numbers["gel_corr"] = get_cellnum_check_corr(cell_drying_path, corr_path, geldrying=True)

            ages.append(session.age)
            cell_numbers_dict[animal_id][session_id] = cell_numbers
        cell_numbers_dict[animal_id]["ages"] = ages
    return cell_numbers_dict

def get_cellnum_check_corr(cell_path, corr_path, geldrying=False):
    if os.path.exists(cell_path):
        iscell = np.load(cell_path)
        cell_number = sum(np.array(iscell==False, dtype="int32")) if geldrying else int(np.sum(iscell[:,0]))
    else:
        cell_number = -1
    corr_present = True if os.path.exists(corr_path) else False
    return cell_number, corr_present

def summary_df_s2p_vs_geldrying(cell_numbers_dict):
    """
    Generates a summary DataFrame comparing Suite2p and gel drying results.

    This function takes as input a dictionary containing cell numbers data for multiple animals. The dictionary keys are animal IDs and 
    the values are data structures containing information about the cells for each animal. The function processes this data to generate a 
    summary DataFrame comparing the results of Suite2p and gel drying for each animal. The resulting DataFrame has one row for each animal 
    and columns for various summary statistics, including the number of cells identified by Suite2p, the number of cells not affected by gel 
    drying, the proportion of cells that survived gel drying, and the proportion of sessions that failed for each method.

    Args:
        cell_numbers_dict (Dict[str, Any]): A dictionary containing cell numbers data for multiple animals. The keys are animal IDs and the 
        values are data structures containing information about the cells for each animal.

    Returns:
        pd.DataFrame: A DataFrame containing summary statistics comparing Suite2p and gel drying results for each animal.
    """
    data = []
    for animal_id, animal in cell_numbers_dict.items():
        sorted_ages, iscells, notgeldrying, corr, gel_corr = get_sorted_cells_notgeldyring_lists(animal)
        sumiscells = sum(iscells)
        sumnotgeldrying = sum(notgeldrying)
        survived_cells = sumnotgeldrying/sumiscells
        sess_count = len(iscells)
        failed_S2P = sum(np.array(iscells)==-1)/sess_count #f"{sum(np.array(iscells)==0)/sess_count:.2%}"
        failed_own = sum(np.array(notgeldrying)==-1)/sess_count #f"{sum(np.array(notgeldrying)==0)/sess_count:.2%}"
        failed_corr = sum(np.array(corr)==False)/sess_count 
        failed_gel_corr = sum(np.array(gel_corr)==False)/sess_count 
        data.append([animal_id, sumiscells, sumnotgeldrying, survived_cells, sess_count, failed_S2P, failed_own, failed_corr, failed_gel_corr])

    pipeline_stats = pd.DataFrame(data, columns=
                                  ["animal_id", "iscells", "notgeldrying", 
                                   "survived_cells", "sess count", 
                                   "Failed_S2P", "Failed_Own",
                                   "Failed_corr", "Failed_gel_corr"])
    pipeline_stats = pipeline_stats.set_index("animal_id")
    pipeline_stats = pipeline_stats.sort_index()
    return pipeline_stats

def get_cells_pdays_df(cell_numbers_dict, suite2p_cells = False):
    #show tabular visualization of usefull mice
    animal_ids = list(cell_numbers_dict.keys())

    ages = []
    for animal_id, animal in cell_numbers_dict.items():
        sorted_ages, iscells, notgeldrying, corr, gel_corr = get_sorted_cells_notgeldyring_lists(animal)
        ages += list(sorted_ages)
    ages = np.unique(ages)

    #create pday_cell_count_dict and set num_cells to -1 for all ages
    pday_cell_count_dict = {}
    for animal_id in animal_ids:
        pday_cell_count_dict[animal_id] = {}
        for age in ages:
            pday_cell_count_dict[animal_id][age] = -1

    #set pday_cell_count_dict[animal_id][age] to the number of not geldrdying cells
    for animal_id, animal in cell_numbers_dict.items():
        sorted_ages, iscells, notgeldrying, corr, gel_corr = get_sorted_cells_notgeldyring_lists(animal)
        cell_count = iscells if suite2p_cells else notgeldrying
        for age, session_cells in zip(sorted_ages, cell_count):
            pday_cell_count_dict[animal_id][age] = session_cells

    pday_cell_count_df = pd.DataFrame(pday_cell_count_dict).transpose()
    return pday_cell_count_df

def get_sorted_cells_notgeldyring_lists(cell_numbers_dict):
    """
    Sorts the cells in a dictionary of cell numbers by age.

    :param cell_numbers_dict: A dictionary of cell numbers where the keys are session IDs and the values are dictionaries containing ages and cell numbers.
    :return: A tuple containing three numpy arrays: sorted ages, sorted iscells, and sorted notgeldrying.
    """
    ages = np.array(cell_numbers_dict["ages"])
    sort_ages_ids = np.argsort(ages)
    sorted_ages = ages[sort_ages_ids]
    sessiondates = np.array(list(cell_numbers_dict.keys())[:-1])
    sorted_sessiondates = sessiondates[sort_ages_ids]
    iscells = []
    notgeldrying = []
    corrs = []
    gel_corrs = []
    for sessiondate in sorted_sessiondates:
        iscells.append(cell_numbers_dict[sessiondate]["iscell"])
        notgeldrying.append(cell_numbers_dict[sessiondate]["not_geldrying"])
        corrs.append(cell_numbers_dict[sessiondate]["corr"])
        gel_corrs.append(cell_numbers_dict[sessiondate]["gel_corr"])
    return np.array(sorted_ages), np.array(iscells), np.array(notgeldrying), np.array(corrs), np.array(gel_corrs)
    
def show_prints(show=True):
    if show:
        # Restore
        sys.stdout = old_stdout if old_stdout else sys.stdout
    else:
        # Disable
        if old_stdout not in globals(): 
            old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

def filter_animals(animal_dict, filters = []):
    """
    Filters the animal dictionary based on the specified filters.

    Parameters:
    - animal_dict (dict): A dictionary containing animal IDs as keys and corresponding Animal objects as values.
    - filters (list, optional): A list of filters to apply. Default is an empty list.

    Returns:
    - filtered_animal_dict (dict): A dictionary containing filtered animal IDs as keys and corresponding Animal objects as values.
    """
    filtered_animal_dict = animal_dict
    for filter in filters:
        tmp_animal_dict = {}
        for animal_id, animal in filtered_animal_dict.items():
            if filter == animal_id:
                tmp_animal_dict[animal_id] = animal
                continue
            if animal.cohort_year == filter: 
                tmp_animal_dict[animal_id] = animal
                continue
            if filter == "male" or filter == "female":
                tmp_animal_dict[animal_id] = animal
                continue
        filtered_animal_dict = tmp_animal_dict
    return filtered_animal_dict

def get_age_range(animal_dict):
    """
    Retrieves the minimum and maximum age range from the given animal dictionary.

    Parameters:
    - animal_dict (dict): A dictionary containing animal IDs as keys and corresponding Animal objects as values.

    Returns:
    - min_age (int): The minimum age value found among all animals.
    - max_age (int): The maximum age value found among all animals.
    """
    ages_list = []
    min_age = float('inf')
    max_age = 0
    for animal_id, animal in animal_dict.items():
        pdays = animal.get_pdays()
        ages_list += pdays
        min_age = min(pdays) if min(pdays) < min_age else min_age
        max_age = max(pdays) if max(pdays) > max_age else max_age
    unique_sorted_ages = np.unique(ages_list)
    unique_sorted_ages.sort() 
    return unique_sorted_ages, min_age, max_age

def split_array(arr, batch_size):
    """
    Splits an array into multiple arrays with batch size.

    Parameters:
        arr (list): The array to be split.
        batch_size (int): The batch size.

    Returns:
        list: A list of sub-arrays where each sub-array has the specified batch size.
    """
    if batch_size == "all":
        batch_size = len(arr)
    else:
        batch_size = int(batch_size)
    return [arr[i:i+batch_size] for i in range(0, len(arr), batch_size)]

### Popups
def yes_no_q(Question):
    """
    Displays a popup window with a yes/no question and returns the user's answer.

    Args:
        Question (str): The question to display in the popup window.

    Returns:
        bool: True if the user clicked "Yes", False if the user clicked "No".
    """
    def on_button_click(button_text):
        global result
        result = button_text
        root.destroy()

    root = tk.Tk()

    label = tk.Label(root, text=Question)
    label.pack()

    button1 = tk.Button(root, text="Yes", command=lambda: on_button_click("Yes"))
    button1.pack()

    button2 = tk.Button(root, text="No", command=lambda: on_button_click("No"))
    button2.pack()

    root.mainloop()

    return True if result=="Yes" else False

#### directory, file search
def dir_exist_create(directory):
    """
    Checks if a directory exists and creates it if it doesn't.

    Parameters:
    dir (str): Path of the directory to check and create.

    Returns:
    None
    """
    # Check if the directory exists
    if not os.path.exists(directory):
        # Create the directory
        os.makedirs(directory)

def file_exist_rename(data_path, fname, fname_new, reset=False):
    """
    Check if a file exists in the specified data path. If it doesn't exist, rename the old file to the new file name.

    Args:
        data_path (str): The path where the files are located.
        fname (str): The name of the file to check.
        fname_new (str): The new name to rename the file to.

    Returns:
        None

    Raises:
        FileNotFoundError: If the file specified by `fname` does not exist in `data_path`.

    Example:
        data_path = '/path/to/files/'
        fname = 'old_file.txt'
        fname_new = 'new_file.txt'
        file_exist_rename(data_path, fname, fname_new)
    """
    fpath_new = os.path.join(data_path, fname_new)
    fpath = os.path.join(data_path, fname)
    if not os.path.exists(fpath):
        print(f"{fname} not exists")
    if reset:
        if os.path.exists(fpath_new):
            del_file_dir(fpath)
            shutil.copyfile(fpath_new, fpath)
        else:
            print(f"{fname_new} not exists")
    else:
        if not os.path.exists(fpath_new):
            if os.path.exists(fpath):
                os.rename(fpath, fpath_new)
        else:
            del_file_dir(fpath)

def create_dirs(dirs):
    """
    Create a new directory hierarchy.

    Args:
        dirs (list): A list of strings representing the path to the new directory.

    Returns:
        str: The path to the newly created directory.
    """
    new_path = dirs[0]
    for path_part in dirs[1:]:
        new_path = os.path.join(new_path, path_part)
        dir_exist_create(new_path)
    return new_path

#reset files S2P files to original ones
def reset_s2p_files(data_path):
    data_path = os.path.join(data_path)
    file_exist_rename(data_path, "F.npy", 'F_old.npy', reset=True)
    file_exist_rename(data_path, "Fneu.npy", 'Fneu_old.npy', reset=True)
    file_exist_rename(data_path, "iscell.npy", 'iscell_old.npy', reset=True)
    file_exist_rename(data_path, "ops.npy", 'ops_old.npy', reset=True)
    file_exist_rename(data_path, "spks.npy", 'spks_old.npy', reset=True)
    file_exist_rename(data_path, "stat.npy", 'stat_old.npy', reset=True)

def backup_path_files(data_path, backup_folder_name="backup", 
                      redo_backup=False, restore=False):
    data_path = os.path.join(data_path)
    if os.path.exists(data_path):
        backup_path = os.path.join(data_path, backup_folder_name)
        if restore:
            shutil.copytree(backup_path, data_path, dirs_exist_ok=True)
            print("Files restored from Backup.")
        else:
            if not os.path.exists(backup_path):
                shutil.copytree(data_path, backup_path)
            else:
                if redo_backup:
                    if os.path.exists(backup_path):
                        shutil.rmtree(backup_path)
                    shutil.copytree(data_path, backup_path)
                else:
                    print("Backup path already exists. Skipping")
    else:
        print("Data path does not exist. Skipping Backup")

def del_file_dir(fpath):
    """
    Deletes a file or directory at the specified path.

    This function checks if the given path exists. If it does, it removes the file or directory at that path.
    If the path is a file, it uses `os.remove` to delete it.
    If the path is a directory, it uses `shutil.rmtree` to delete it.

    Parameters:
    fpath (str): The path of the file or directory to be deleted.

    Returns:
    None
    """
    # check if the file or path exists
    if os.path.exists(fpath):
        print(f"removing {fpath}")
        if os.path.isfile(fpath):
            os.remove(fpath)
        else:
            shutil.rmtree(fpath)

def get_directories(directory, regex_search=""):
    """
    This function returns a list of directories from the specified directory that match the regular expression search pattern.
    
    Parameters:
    directory (str): The directory path where to look for directories.
    regex_search (str, optional): The regular expression pattern to match. Default is an empty string, which means all directories are included.
    
    Returns:
    list: A list of directory names that match the regular expression search pattern.
    """
    directories = None
    if os.path.exists(directory):
        directories = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name)) and len(re.findall(regex_search, name))>0]
    else:
        print(f"Directory does not exist: {directory}")
    return directories

def get_files(directory, ending="", regex_search=""):
    """
    This function returns a list of files from the specified directory that match the regular expression search pattern and have the specified file ending.
    
    Parameters:
    directory (str): The directory path where to look for files.
    ending (str, optional): The file ending to match. Default is '', which means all file endings are included.
    regex_search (str, optional): The regular expression pattern to match. Default is an empty string, which means all files are included.
    
    Returns:
    list: A list of file names that match the regular expression search pattern and have the specified file ending.
    """
    files_list = None
    if os.path.exists(directory):
        files_list = [name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name)) and len(re.findall(regex_search, name))>0 and name.endswith(ending)]
    else:
        print(f"Directory does not exist: {directory}")
    return files_list

def search_file(directory, filename):
    """
    This function searches for a file with a given filename within a specified directory and its subdirectories.

    :param directory: The directory in which to search for the file.
    :type directory: str
    :param filename: The name of the file to search for.
    :type filename: str
    :return: The full path of the file if found, otherwise returns the string "Not found".
    :rtype: str
    """
    for root, dirs, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None