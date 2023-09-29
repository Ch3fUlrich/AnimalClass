import yaml
import os
import shutil
from datetime import datetime
from openpyxl import load_workbook, Workbook
import re
import numpy as np
import h5py

root_animal_yaml_name = "animal_summary.yaml"

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
    directories = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    return directories

def get_animal_folder_names(directory):
    directories = get_directories(directory)
    animal_folder_names = [folder for folder in directories if folder[:3]=="DON"]
    return animal_folder_names

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

def row_to_list(sheet, row):
    result = []
    # Iterate over cells in the specified row
    for num, cell in enumerate(sheet[row][1:10000]):
        value = cell.value
        if cell.value != None:
            result.append(value)
        else:
            break 
    return result

def num_to_date(date_string):
    """
    :parameter add_20 add 20 to the number in front so the date has the format YYYYMMDD
    """
    if type(date_string) != str:
        date_string = str(date_string)
    date = datetime.strptime(date_string, '%Y%m%d')
    return date

def get_animal_dict_from_spreadsheet(fname):
    """
    read animal_id, dob, sex from spreadsheet
    """
    org_exp_workbook = load_workbook(filename=fname)
    sheet = org_exp_workbook.active
    animal_id_records = row_to_list(sheet, "C")
    dob_records = row_to_list(sheet, "K")
    sex_records = row_to_list(sheet, "D")
    animals = create_animal_dict(animal_id_records, dob_records, sex_records)
    return animals

def init_animal_dict(animal_id, cohort_year=None, dob=None, sex=None):
    animal_dict = {
            'cohort_year': cohort_year,
            'dob': dob,
            'name': animal_id,
            'pdays': [],
            'session_dates': [],
            'session_names': [],
            'sex': "male" if sex == "m" else "female" if sex else None
        }
    return animal_dict

def create_animal_dict(animal_id_records, dob_records, sex_records):
    animals = {}
    print(f"Warning cohort_year if defined by dob year.")
    print(f"Warning dob year 2020 is changed to 2021.")
    for animal_id, dob, sex in zip(animal_id_records, dob_records, sex_records):
        animal_id = "DON-00"+animal_id[3:] if len(animal_id) == 7 else "DON-0"+animal_id[3:]
        if animal_id in animals:
            continue

        dob = "20"+str(int(dob))
        dob_date = num_to_date(dob)
        #WARNING dob_date.year could be wrong for other 
        cohort_year = dob_date.year if dob_date.year != 2020 else 2021 
        animals[animal_id] = init_animal_dict(animal_id, cohort_year, dob, sex)
    return animals

def return_loaded_yaml_if_newer(used_path, may_newer_info_path):
    yaml_dict = None
    root_yaml_modification_date = os.path.getmtime(used_path) if os.path.exists(used_path) else 0
    if os.path.exists(may_newer_info_path):
        yaml_modification_date = os.path.getmtime(may_newer_info_path)
        if yaml_modification_date > root_yaml_modification_date:
            with open(may_newer_info_path, "r") as yaml_file:
                yaml_dict = yaml.safe_load(yaml_file)
    return yaml_dict

def get_animals_from_yaml(directory):
    root_dir = directory if directory else ""
    root_yaml_path = os.path.join(root_dir, root_animal_yaml_name)
    if os.path.exists(root_yaml_path):
        
        with open(root_yaml_path, "r") as yaml_file:
            animals = yaml.safe_load(yaml_file)
    else:
        animals = {}

    for animal_id in get_animal_folder_names(root_dir):
        animal_path = os.path.join(root_dir, animal_id)

        # update animals if a file has newer information changed
        animal_yaml_path = os.path.join(animal_path, animal_id+".yaml")
        animal = return_loaded_yaml_if_newer(root_yaml_path, animal_yaml_path)
        animals[animal_id] = animal if animal else animals[animal_id]
    return animals

def add_session_animal_folders(animals, animals_spreadsheet, directory=None):
    root_dir = directory if directory else ""
    for animal_id in get_animal_folder_names(root_dir):
        if animal_id not in animals:
            animals[animal_id] = animals_spreadsheet[animal_id]
            print(f"added animal from spreadsheet: {animal_id}")
        animal_path = os.path.join(root_dir, animal_id)
        
        for session_id in get_directories(animal_path):
            session_path = os.path.join(animal_path, session_id, "002P-F")

            mesc_fnames = get_files(session_path, ending=".mesc")
            mesc_munit_pairs = animals[animal_id]["UseMUnits"] if "UseMUnits" in animals[animal_id].keys() else []
            for fname in mesc_fnames:
                splitted_fname = fname.split("_")
                if animal_id != splitted_fname[0]:
                    continue

                # add session data based on file
                session_date = splitted_fname[1]
                if session_id not in animals[animal_id]["session_names"] and session_date not in animals[animal_id]["session_dates"]:
                    animals[animal_id]["session_names"].append(session_id)
                    animals[animal_id]["session_dates"].append(session_date)
                    dob_date = num_to_date(animals[animal_id]["dob"])
                    session_date = num_to_date(session_date)
                    pday = (session_date-dob_date).days
                    animals[animal_id]["pdays"].append(pday)
                    
                # get available MUnits
                last_fname_part = splitted_fname[-1].split(".")[0]
                session_parts = [int(part_number[-1])-1 for part_number in re.findall("S[0-9]", last_fname_part)]
                fpath = os.path.join(session_path, fname)
                munits_list = get_recording_munits(fpath, session_parts)
                if len(munits_list) <= len(session_parts):
                    usefull_munits = munits_list
                    file_naming = session_parts[:len(usefull_munits)]
                else:
                    add_mesc_munit_pair = True
                    if len(mesc_munit_pairs) > 0:
                        for mesc_munit_pair in mesc_munit_pairs:
                            if fname in mesc_munit_pair:
                                add_mesc_munit_pair = False
                    if add_mesc_munit_pair:
                        mesc_munit_pair = [fname, munits_list]
                        mesc_munit_pairs.append(mesc_munit_pair)

            if len(mesc_munit_pairs) > 0:
                animals[animal_id]["UseMUnits"] = mesc_munit_pairs
    return animals 

def get_recording_munits(mesc_fpath, session_parts, fps = 30, at_least_minutes_of_recording=5):
    # Get MUnit number list of first Mescfile session MSession_0
    with h5py.File(mesc_fpath, 'r') as file:
        munits = file[list(file.keys())[0]]
        recording_munits = []
        for name, unit in munits.items():
            # if recording has at least x minutes
            if unit["Channel_0"].shape[0] > fps*60*at_least_minutes_of_recording: 
                unit_number = name.split("_")[-1]
                recording_munits.append(int(unit_number))
    return recording_munits

def move_mesc_to_session_folder(directory=None):
    directory = None if directory == "" else directory
    for file in get_files(directory, ending=".mesc"):
        if os.path.isfile(file): #is file
            splitted_fname = file.split("_")
            if splitted_fname[0][:3] != "DON": #not animal file
                continue
            animal_id = splitted_fname[0]
            session_id = splitted_fname[1].split(".")[0]
            session_path = create_folder(str(animal_id), str(session_id), directory=directory)
            #move_file 
            shutil.move(file, session_path)

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

def create_folder(animal_id, session_id, directory=None):
    folder_dir = directory if directory else ""
    folder_names = [animal_id, session_id, "002P-F"]
    for folder_name in folder_names:
        folder_dir = os.path.join(folder_dir, folder_name)
        dir_exist_create(folder_dir)
    return folder_dir

def add_yaml_to_folders(animals, directory=None):
    directory = directory if directory else ""
    for animal_id, animal in animals.items():
        animal_path = os.path.join(directory, str(animal_id))
        yaml_path = os.path.join(animal_path, f"{animal_id}.yaml")
        if os.path.exists(animal_path):
            with open(yaml_path, "w") as f:
                yaml.dump(animal, f)

def main(directory = None):
    root_dir = directory if directory else ""
    fname = os.path.join("Intrinsic_CA3_database-September_7,_10_08_AM.xlsx")
    fpath = os.path.join(root_dir, fname)
    # load spreadsheet information
    animals_spreadsheed = get_animal_dict_from_spreadsheet(fpath)
    # move mesc in root directory to correct folder location
    move_mesc_to_session_folder(directory=root_dir)
    # load animal yaml files
    animals_yaml = get_animals_from_yaml(root_dir)
    # get animals based on folder structure
    animals = add_session_animal_folders(animals_yaml, animals_spreadsheed, directory=root_dir)
    add_yaml_to_folders(animals, directory=root_dir)
    with open(os.path.join(root_dir, root_animal_yaml_name), 'w') as file:
        yaml.dump(animals, file)

if __name__ == "__main__":
    root_dir = "/scicore/projects/donafl00-calcium/Users/Sergej/Steffen_Experiments"
    main()