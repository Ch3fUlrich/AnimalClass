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
# module_path = os.path.abspath(os.path.join('../'))
# print(module_path)
# sys.path.append(module_path)
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
    save_path = path.replace(".gif", ".mp4")
    clip.write_videofile(save_path)


def xticks_frames_to_seconds(frames, fps=30.95):
    import matplotlib.pyplot as plt

    seconds = 5
    num_frames = fps * seconds
    num_x_ticks = 50
    written_label_steps = 2

    x_time = [
        int(frame / num_frames) * seconds
        for frame in range(frames)
        if frame % num_frames == 0
    ]
    steps = round(len(x_time) / (2 * num_x_ticks))
    x_time_shortened = x_time[::steps]
    x_pos = np.arange(0, frames, num_frames)[::steps]

    x_labels = [
        time if num % written_label_steps == 0 else ""
        for num, time in enumerate(x_time_shortened)
    ]
    plt.xticks(x_pos, x_labels, rotation=40, fontsize=8)


def show_mesc_units(path):
    h5 = h5py.File(path, "r")
    for munit_id, MUnits in h5["MSession_0"].items():
        print(munit_id)
        print(MUnits["Channel_0"])


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


def create_contours(stat_or_points):
    """
    Extracts and returns the contours of cells based on pixel coordinates.

    Parameters:
    - clean (bool, optional): If True, removes geldrying cells from the extracted contours.
                              Defaults to False.

    Returns:
    np.ndarray: An array containing the cell contours. Each contour is represented
                as an array of (x, y) coordinates.

    Note:
    The cell contours are calculated using Convex Hull on the pixel coordinates of
    cells obtained from the loaded statistics.

    If the `clean` parameter is set to True, the function removes geldrying cells
    from the extracted contours using the `remove_geldrying_cells` method.

    Example:
    >>> contours = obj.get_cell_contours(clean=True)
    """
    from scipy.spatial import ConvexHull

    contours = []
    for cell_stat in stat_or_points:
        if isinstance(cell_stat, np.ndarray):
            x_y_points = np.array([cell_stat[:, 1], cell_stat[:, 0]]).T
        else:
            x_y_points = np.array([cell_stat["xpix"], cell_stat["ypix"]]).T
        # find convex hull of temp
        hull = ConvexHull(x_y_points)
        contour_open = x_y_points[hull.vertices]
        # add the last point to close the contour
        contour = np.vstack([contour_open, contour_open[0]])
        contours.append(contour)
    cell_contours = np.array(contours)
    return cell_contours


def num_to_date(date_string):
    if type(date_string) != str:
        date_string = str(date_string)
    date = datetime.strptime(date_string, "%Y%m%d")
    return date


def get_num_batches_based_on_available_ram():
    stats = psutil.virtual_memory()  # returns a named tuple
    available = getattr(stats, "available")
    byte_to_gb = 1 / 1000000000
    available_ram_gb = available * byte_to_gb
    print("Setting Number of Batches according to free RAM")
    num_batches = 16
    num_batches_range = [12, 8, 4, 2, 1]
    ram_range = [32, 16, 32, 64, 128]
    for batches, ram in zip(num_batches_range, ram_range):
        if available_ram_gb > ram:
            num_batches = batches
    print(
        f"Available RAM: {round(available_ram_gb)}GB setting number of batches to {num_batches}"
    )
    return num_batches


def make_list_ifnot(string_or_list):
    return [string_or_list] if type(string_or_list) != list else string_or_list


def save_file_present(file_path):
    splitter = "\\" if "\\" in file_path else "/"
    fname = file_path.split(splitter)[-1]
    file_present = False
    if os.path.exists(file_path):
        print(f"File already present {file_path}")
        file_present = True
    else:
        print(f"Saving {fname} to {file_path}")
    return file_present


def find_binary_fpath(
    data_path,
    subdirectories=["data"],
    possible_binary_fnames=["data.bin", "Image_001_001.raw"],
):
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
    possible_binary_data_paths = [data_path] + [
        os.path.join(data_path, subdirectory) for subdirectory in subdirectories
    ]
    for possible_binary_data_path in possible_binary_data_paths:
        for possible_binary_fname in possible_binary_fnames:
            binary_file_path = os.path.join(
                possible_binary_data_path, possible_binary_fname
            )
            if os.path.exists(binary_file_path):
                binary_fpath = binary_file_path
                break
        if binary_fpath:
            break
    if not binary_fpath:
        print(
            f"No binary path to {possible_binary_fnames} found in {possible_binary_data_paths}"
        )
    return binary_fpath


def remove_rows_cols(data, remove_rows, remove_cols):
    data = np.delete(data, remove_rows, 0)
    data = np.delete(data, remove_cols, 1)
    return data


def extract_cell_numbers(animals):
    """
    Extracts cell numbers from a dictionary of animals and their sessions.

    :param animals: A dictionary of animals, where the keys are animal IDs and the values are animal objects.
    :return: A dictionary where the keys are animal IDs and the values are dictionaries containing session IDs, pdays, and cell numbers.
    """
    cell_numbers_dict = {}
    for animal_id, animal in animals.items():
        cell_numbers_dict[animal_id] = {}
        pdays = []
        for session_id, session in animal.sessions.items():
            cell_numbers = {}
            cell_numbers["iscell"] = -1
            cell_numbers["not_geldrying"] = -1
            cell_numbers["corr"] = False
            cell_numbers["gel_corr"] = False
            if not session.suite2p_dirs:
                print(f"No suite2p paths: {animal_id} {session_id}")
                continue
            for path in session.suite2p_dirs:
                path_ending = path.split("suite2p")[-1]
                path = os.path.join(path, "plane0")
                iscell_path = os.path.join(path, "iscell.npy")
                cell_drying_path = os.path.join(path, "cell_drying.npy")
                corr_path = os.path.join(path, "allcell_corr_pval_zscore.npy")
                if path_ending == "":
                    (
                        cell_numbers["iscell"],
                        cell_numbers["corr"],
                    ) = get_cellnum_check_corr(iscell_path, corr_path)
                elif path_ending == "_merged":
                    (
                        cell_numbers["not_geldrying"],
                        cell_numbers["gel_corr"],
                    ) = get_cellnum_check_corr(
                        cell_drying_path, corr_path, geldrying=True
                    )

            pdays.append(session.pday)
            cell_numbers_dict[animal_id][session_id] = cell_numbers
        cell_numbers_dict[animal_id]["pdays"] = pdays
    return cell_numbers_dict


def get_cellnum_check_corr(cell_path, corr_path, geldrying=False):
    if os.path.exists(cell_path):
        iscell = np.load(cell_path)
        cell_number = (
            sum(np.array(iscell == False, dtype="int32"))
            if geldrying
            else int(np.sum(iscell[:, 0]))
        )
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
        (
            sorted_pdays,
            iscells,
            notgeldrying,
            corr,
            gel_corr,
        ) = get_sorted_cells_notgeldyring_lists(animal)
        sumiscells = sum(iscells)
        sumnotgeldrying = sum(notgeldrying)
        survived_cells = sumnotgeldrying / sumiscells
        sess_count = len(iscells)
        failed_S2P = (
            sum(np.array(iscells) == -1) / sess_count
        )  # f"{sum(np.array(iscells)==0)/sess_count:.2%}"
        failed_own = (
            sum(np.array(notgeldrying) == -1) / sess_count
        )  # f"{sum(np.array(notgeldrying)==0)/sess_count:.2%}"
        failed_corr = sum(np.array(corr) == False) / sess_count
        failed_gel_corr = sum(np.array(gel_corr) == False) / sess_count
        data.append(
            [
                animal_id,
                sumiscells,
                sumnotgeldrying,
                survived_cells,
                sess_count,
                failed_S2P,
                failed_own,
                failed_corr,
                failed_gel_corr,
            ]
        )

    pipeline_stats = pd.DataFrame(
        data,
        columns=[
            "animal_id",
            "iscells",
            "notgeldrying",
            "survived_cells",
            "sess count",
            "Failed_S2P",
            "Failed_Own",
            "Failed_corr",
            "Failed_gel_corr",
        ],
    )
    pipeline_stats = pipeline_stats.set_index("animal_id")
    pipeline_stats = pipeline_stats.sort_index()
    return pipeline_stats


def get_cells_pdays_df(cell_numbers_dict, suite2p_cells=False):
    # show tabular visualization of usefull mice
    animal_ids = list(cell_numbers_dict.keys())

    pdays = []
    for animal_id, animal in cell_numbers_dict.items():
        (
            sorted_pdays,
            iscells,
            notgeldrying,
            corr,
            gel_corr,
        ) = get_sorted_cells_notgeldyring_lists(animal)
        pdays += list(sorted_pdays)
    pdays = np.unique(pdays)

    # create pday_cell_count_dict and set num_cells to -1 for all pdays
    pday_cell_count_dict = {}
    for animal_id in animal_ids:
        pday_cell_count_dict[animal_id] = {}
        for pday in pdays:
            pday_cell_count_dict[animal_id][pday] = -1

    # set pday_cell_count_dict[animal_id][pday] to the number of not geldrdying cells
    for animal_id, animal in cell_numbers_dict.items():
        (
            sorted_pdays,
            iscells,
            notgeldrying,
            corr,
            gel_corr,
        ) = get_sorted_cells_notgeldyring_lists(animal)
        cell_count = iscells if suite2p_cells else notgeldrying
        for pday, session_cells in zip(sorted_pdays, cell_count):
            pday_cell_count_dict[animal_id][pday] = session_cells

    pday_cell_count_df = pd.DataFrame(pday_cell_count_dict).transpose()
    return pday_cell_count_df


def get_sorted_cells_notgeldyring_lists(cell_numbers_dict):
    """
    Sorts the cells in a dictionary of cell numbers by pday.

    :param cell_numbers_dict: A dictionary of cell numbers where the keys are session IDs and the values are dictionaries containing pdays and cell numbers.
    :return: A tuple containing three numpy arrays: sorted pdays, sorted iscells, and sorted notgeldrying.
    """
    pdays = np.array(cell_numbers_dict["pdays"])
    sort_pdays_ids = np.argsort(pdays)
    sorted_pdays = pdays[sort_pdays_ids]
    sessiondates = np.array(list(cell_numbers_dict.keys())[:-1])
    sorted_sessiondates = sessiondates[sort_pdays_ids]
    iscells = []
    notgeldrying = []
    corrs = []
    gel_corrs = []
    for sessiondate in sorted_sessiondates:
        iscells.append(cell_numbers_dict[sessiondate]["iscell"])
        notgeldrying.append(cell_numbers_dict[sessiondate]["not_geldrying"])
        corrs.append(cell_numbers_dict[sessiondate]["corr"])
        gel_corrs.append(cell_numbers_dict[sessiondate]["gel_corr"])
    return (
        np.array(sorted_pdays),
        np.array(iscells),
        np.array(notgeldrying),
        np.array(corrs),
        np.array(gel_corrs),
    )


def show_prints(show=True):
    if show:
        # Restore
        sys.stdout = old_stdout if old_stdout else sys.stdout
    else:
        # Disable
        if old_stdout not in globals():
            old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")


def filter_animals(animal_dict, filters=[]):
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


def get_pday_range(animal_dict):
    """
    Retrieves the minimum and maximum pday range from the given animal dictionary.

    Parameters:
    - animal_dict (dict): A dictionary containing animal IDs as keys and corresponding Animal objects as values.

    Returns:
    - min_pday (int): The minimum pday value found among all animals.
    - max_pday (int): The maximum pday value found among all animals.
    """
    pdays_list = []
    min_pday = float("inf")
    max_pday = 0
    for animal_id, animal in animal_dict.items():
        pdays = animal.get_pdays()
        pdays_list += pdays
        min_pday = min(pdays) if min(pdays) < min_pday else min_pday
        max_pday = max(pdays) if max(pdays) > max_pday else max_pday
    unique_sorted_pdays = np.unique(pdays_list)
    unique_sorted_pdays.sort()
    return unique_sorted_pdays, min_pday, max_pday


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
    return [arr[i : i + batch_size] for i in range(0, len(arr), batch_size)]


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

    return True if result == "Yes" else False


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


# reset files S2P files to original ones
def reset_s2p_files(data_path):
    data_path = os.path.join(data_path)
    file_exist_rename(data_path, "F.npy", "F_old.npy", reset=True)
    file_exist_rename(data_path, "Fneu.npy", "Fneu_old.npy", reset=True)
    file_exist_rename(data_path, "iscell.npy", "iscell_old.npy", reset=True)
    file_exist_rename(data_path, "ops.npy", "ops_old.npy", reset=True)
    file_exist_rename(data_path, "spks.npy", "spks_old.npy", reset=True)
    file_exist_rename(data_path, "stat.npy", "stat_old.npy", reset=True)


def backup_path_files(
    data_path, backup_folder_name="backup", redo_backup=False, restore=False
):
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


def copy_object_attributes_to_object(
    propertie_name_list, set_object, get_object=None, propertie_values=None
):
    """
    Set attributes of a target object based on a list of property names and values.

    This function allows you to set attributes on a target object (the 'set_object') based on a list of
    property names provided in 'propertie_name_list' and corresponding values. You can specify these
    values directly through 'propertie_values' or retrieve them from another object ('get_object').
    If 'propertie_values' is not provided, this function will attempt to fetch the values from the
    'get_object' using the specified property names.

    Args:
        propertie_name_list (list): A list of property names to set on the 'set_object.'
        set_object (object): The target object for attribute assignment.
        get_object (object, optional): The source object to retrieve property values from. Default is None.
        propertie_values (list, optional): A list of values corresponding to the property names.
            Default is None.

    Returns:
        None

    Raises:
        ValueError: If the number of properties in 'propertie_name_list' does not match the number of values
            provided in 'propertie_values' (if 'propertie_values' is specified).

    Example Usage:
        # Example 1: Set attributes directly with values
        copy_object_attributes_to_object(["attr1", "attr2"], my_object, propertie_values=[value1, value2])

        # Example 2: Retrieve attribute values from another object
        copy_object_attributes_to_object(["attr1", "attr2"], my_object, get_object=source_object)
    """
    propertie_name_list = list(propertie_name_list)
    if propertie_values:
        propertie_values = list(propertie_values)
        if len(propertie_values) != len(propertie_name_list):
            raise ValueError(
                f"Number of properties does not match given propertie values: {len(propertie_name_list)} != {len(propertie_values)}"
            )
    elif get_object:
        propertie_values = []
        for propertie in propertie_name_list:
            if propertie in get_object.__dict__.keys():
                propertie_values.append(getattr(get_object, propertie))
            else:
                propertie_values.append(None)

    propertie_values = (
        propertie_values if propertie_values else [None] * len(propertie_name_list)
    )
    for propertie, value in zip(propertie_name_list, propertie_values):
        if value or not value and propertie not in set_object.__dict__.keys():
            setattr(set_object, propertie, value)


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
        directories = [
            name
            for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))
            and len(re.findall(regex_search, name)) > 0
        ]
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
        files_list = [
            name
            for name in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, name))
            and len(re.findall(regex_search, name)) > 0
            and name.endswith(ending)
        ]
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


##################################################################################################################
############################## Check for errors in the pipeline ##################################################
##################################################################################################################


def add_error_output(animals, output_directory, strings_in_error=["Error:", "error:"]):
    from pathlib import Path

    ofnames = get_files(directory=output_directory, ending=".o")
    outputs = {}
    for ofname in ofnames:
        with open(os.path.join(output_directory, ofname), "r") as f:
            output = f.read()
            efname = ofname.replace(".o", ".e")
            with open(os.path.join(output_directory, efname), "r") as f:
                error = f.read()
            outputs[output] = error

    efnames = [ofname.replace(".o", ".e") for ofname in ofnames]
    for animal_id, animal in animals.items():
        for session_id, session in animal.sessions.items():
            session.output = None
            session.error = None

    for animal_id, animal in animals.items():
        for session_id, session in animal.sessions.items():
            for output, error in outputs.items():
                for string_in_error in strings_in_error:
                    if (
                        animal_id in output
                        and session_id in output
                        and string_in_error in error
                    ):
                        session.output = output
                        session.error = error
                        break
    return animals


def print_error_outputs(
    animals, output_directory, num_last_lines=10, strings_in_error=["Error:", "error:"]
):
    animals = add_error_output(
        animals, output_directory, strings_in_error=strings_in_error
    )
    counter = 0
    for animal_id, session_id, session in yield_animal_session(animals):
        # if session.output:
        #    print(f"{animal_id} {session_id} {session.output}")
        if session.error:
            last_lines = "\n".join(session.error.split("\n")[-num_last_lines:])
            print(f"{animal_id} {session_id} {last_lines}")
            counter += 1
    return counter


def get_missing_merge_sessions(animals):
    missing_sessions = []
    for animal_id, animal in animals.items():
        for session_id, session in animal.sessions.items():
            missing = True
            s2p_plane0_paths = session.suite2p_plane0_paths
            if s2p_plane0_paths:
                for s2p_plane0 in s2p_plane0_paths:
                    if "merged" in s2p_plane0:
                        if os.path.exists(os.path.join(s2p_plane0, "F.npy")):
                            missing = False
            if missing:
                print(animal_id, session_id)
                missing_sessions.append(session)
    return missing_sessions
