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

# Helper functions
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
            if animal.year == filter: # cohort_year
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
        ages_list.append(animal.pdays)
        min_age = min(animal.pdays) if min(animal.pdays) < min_age else min_age
        max_age = max(animal.pdays) if max(animal.pdays) > max_age else max_age
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
        if os.path.exists(fpath) and os.path.exists(fpath_new):
            os.remove(fpath)
        os.rename(fpath_new, fpath)
    else:
        if not os.path.exists(fpath_new):
            if os.path.exists(fpath):
                os.rename(fpath, fpath_new)

#reset files S2P files to original ones
def reset_s2p_files(data_path):
    file_exist_rename(data_path, "F.npy", 'F_old.npy', reset=True)
    file_exist_rename(data_path, "Fneu.npy", 'Fneu_old.npy', reset=True)
    file_exist_rename(data_path, "iscell.npy", 'iscell_old.npy', reset=True)
    file_exist_rename(data_path, "ops.npy", 'ops_old.npy', reset=True)
    file_exist_rename(data_path, "spks.npy", 'spks_old.npy', reset=True)
    file_exist_rename(data_path, "stat.npy", 'stat_old.npy', reset=True)

def del_present_file(directory):
    """
    Deletes a file if it exists.

    Parameters:
    file_location (str): Path of the file to delete.

    Returns:
    None
    """
    # check if the file exists
    if os.path.exists(directory):
        # delete the file
        os.remove(directory)

def get_directories(directory):
    """
    Returns a list of directories in the specified folder path.

    Args:
        folder_path (str): The path of the folder to get the directories from.

    Returns:
        list: A list of directory names.
    """
    # Get a list of directories in the specified folder
    # Filter the list to include only directories (excluding the "figures" directory)
    directories = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name)) and name!="figures"]
    return directories

def get_files(directory, ending="all"):
    """
    This function returns a list of files in a given directory. 
    If an ending is specified, it returns only the files that end with the specified ending.
    
    :param directory: The directory to search for files.
    :type directory: str
    :param ending: The file ending to filter by. Default value is "all", which returns all files.
    :type ending: str
    :return: A list of files in the given directory. If an ending is specified, only files that end with the specified ending are returned.
    :rtype: list
    """
    files_list = [name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))]
    if ending != "all":
        files_list_with_ending = []
        for file in files_list:
            if file.endswith(ending):
                files_list_with_ending.append(file)
        return files_list_with_ending
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