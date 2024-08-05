import yaml
import os
import sys
import copy
import shutil
from datetime import datetime
from openpyxl import load_workbook, Workbook
import re
import numpy as np
import h5py

module_path = os.path.abspath(os.path.join("../"))
sys.path.append(module_path)
from Helper import *

manually_eddited_animals_yaml_fname = "animal_summary.yaml"


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
    if type(date_string) != str:
        date_string = str(date_string)
    date = datetime.strptime(date_string, "%Y%m%d")
    return date

def define_metadata_columns(sheet):
    metadata_columns = {}
    for column in range(1, sheet.max_column):
        cell = sheet.cell(row=1, column=column)
        if cell.value is not None:
            metadata_columns[cell.value] = column
    return metadata_columns

def get_animal_dict_from_spreadsheet(fname, sheet_title=None):
    """
    read animal_id, dob, sex from spreadsheet
    """
    org_exp_workbook = load_workbook(filename=fname)
    sheet_title = sheet_title if sheet_title else org_exp_workbook.sheetnames[0]
    if sheet_title not in org_exp_workbook.sheetnames:
        raise ValueError(f"sheet_title {sheet_title} not in sheetnames in {fname}")
    sheet = org_exp_workbook[sheet_title]
    print(f"Warning cohort_year if defined by dob year.")
    print(f"Warning dob year 2020 is changed to 2021.")
    metadata_columns = define_metadata_columns(sheet)
    animals = {}
    for row in range(2, sheet.max_row):
        # cell_obj = sheet.cell(row=row, column=j)
        date = sheet.cell(row=row, column=metadata_columns["date"]).value
        date = "20" + str(int(date)) if date else None
        animal_id = sheet.cell(row=row, column=metadata_columns["mouse ID"]).value
        if not animal_id:
            continue
        else:
            animal_id = (
                "DON-00" + animal_id[3:]
                if len(animal_id) == 7
                else "DON-0" + animal_id[3:]
            )
        sex = sheet.cell(row=row, column=metadata_columns["sex"]).value
        dob = sheet.cell(row=row, column=metadata_columns["DOB"]).value
        injected = sheet.cell(row=row, column=metadata_columns["injected"]).value
        injected = "20" + str(int(injected)) if injected else None
        implanted = sheet.cell(row=row, column=metadata_columns["implanted"]).value
        implanted = "20" + str(int(implanted)) if implanted else None
        duration = [sheet.cell(row=row, column=metadata_columns["duration [min]"]).value]
        duration = [
            (
                None
                if duration == "n/a" or duration == "" or duration == "?"
                else duration[0]
            )
        ]
        method = "2P"
        setup = sheet.cell(row=row, column=metadata_columns["setup"]).value
        cam_data = sheet.cell(row=row, column=metadata_columns["cam"]).value
        cam_data = [True if cam_data == "yes" else False]
        movement_data = sheet.cell(row=row, column=metadata_columns["behaviour"]).value
        movement_data = [True if movement_data == "yes" else False]
        light = sheet.cell(row=row, column=metadata_columns["[nm]"]).value
        laser_power = sheet.cell(row=row, column=metadata_columns["laser / LED power"]).value
        underground = [sheet.cell(row=row, column=metadata_columns["treadmill"]).value]
        pockel_cell_bias = sheet.cell(row=row, column=metadata_columns["PC bias"]).value
        n_channel = sheet.cell(row=row, column=metadata_columns["n Ch."]).value
        fucntional_channel = sheet.cell(row=row, column=metadata_columns["fct. channel"]).value
        #fucntional_channel = sheet.cell(row=row, column=metadata_columns["fucntional_channel"]).value
        ug_gain = sheet.cell(row=row, column=metadata_columns["UG gain"]).value
        ur_gain = sheet.cell(row=row, column=metadata_columns["UR gain"]).value
        pixels = sheet.cell(row=row, column=metadata_columns["pixels"]).value
        n_planes = sheet.cell(row=row, column=metadata_columns["n planes"]).value
        session = [sheet.cell(row=row, column=metadata_columns["session"]).value]
        weight = sheet.cell(row=row, column=metadata_columns["weight [g]"]).value
        comment = sheet.cell(row=row, column=metadata_columns["comment"]).value
        expt_pipeline = sheet.cell(row=row, column=metadata_columns["paradigm"]).value
        if "lens" in metadata_columns.keys():
            lens = sheet.cell(row=row, column=metadata_columns["lens"]).value

        convert_to_int = [n_channel, fucntional_channel, n_planes]
        for i, value in enumerate(convert_to_int):
            if value:
                if value == "n/a" or value == "" or value == "?":
                    convert_to_int[i] = None
                else:
                    convert_to_int[i] = int(value)
        n_channel, fucntional_channel, n_planes = convert_to_int

        animal_exists = False
        session_exists = False
        if len(animals.keys()) > 0:
            if animal_id in animals:
                animal_exists = True
                session_exists = (
                    True if date in animals[animal_id]["sessions"] else False
                )

        if not animal_exists:
            animals[animal_id] = create_animal_dict(
                animal_id=animal_id,
                sex=sex,
                dob=dob,
                injected=injected,
                implanted=implanted,
            )

        if session_exists:
            animals[animal_id]["sessions"][date]["duration"].append(duration[0])
            animals[animal_id]["sessions"][date]["underground"].append(underground[0])
            animals[animal_id]["sessions"][date]["cam_data"].append(cam_data[0])
            animals[animal_id]["sessions"][date]["movement_data"].append(
                movement_data[0]
            )
            animals[animal_id]["sessions"][date]["session_parts"].append(session[0])
        else:
            animals[animal_id]["sessions"][date] = create_session_dict(
                date=date,
                duration=duration,
                method=method,
                setup=setup,
                expt_pipeline=expt_pipeline,
                underground=underground,
                cam_data=cam_data,
                movement_data=movement_data,
                light=light,
                laser_power=laser_power,
                pockel_cell_bias=pockel_cell_bias,
                n_channel=n_channel,
                fucntional_channel=fucntional_channel,
                ug_gain=ug_gain,
                ur_gain=ur_gain,
                lens=lens,
                pixels=pixels,
                n_planes=n_planes,
                session_parts=session,
                weight=weight,
                comment=comment,
            )
    return animals


def init_animal_dict(animal_id, cohort_year=None, dob=None, sex=None):
    animal_dict = {
        "cohort_year": cohort_year,
        "dob": dob,
        "name": animal_id,
        "pdays": [],
        "session_dates": [],
        "session_names": [],
        "sex": sex,
    }
    return animal_dict


def create_session_dict(**kwargs):
    session_dict = kwargs
    for key, var in session_dict.items():
        session_dict[key] = None if var == "n/a" or var == "" or var == "?" else var
    # WARNING dob_date.year could be wrong for other
    return session_dict


def create_animal_dict(**kwargs):
    animal_dict = kwargs
    for key, var in animal_dict.items():
        animal_dict[key] = None if var == "n/a" or var == "" or var == "?" else var

    sex = animal_dict["sex"]
    animal_dict["sex"] = "male" if sex == "m" else "female" if sex else None
    dob = animal_dict["dob"]
    dob = "20" + str(int(dob))
    animal_dict["dob"] = dob
    dob_date = num_to_date(dob)
    cohort_year = dob_date.year if dob_date.year != 2020 else 2021
    animal_dict["cohort_year"] = cohort_year
    animal_dict["sessions"] = {}
    # WARNING dob_date.year could be wrong for other
    return animal_dict


def return_loaded_yaml_if_newer(used_path, may_newer_info_path):
    yaml_dict = None
    root_yaml_modification_date = (
        os.path.getmtime(used_path) if os.path.exists(used_path) else 0
    )
    if os.path.exists(may_newer_info_path):
        yaml_modification_date = os.path.getmtime(may_newer_info_path)
        if yaml_modification_date > root_yaml_modification_date:
            with open(may_newer_info_path, "r") as yaml_file:
                yaml_dict = yaml.safe_load(yaml_file)
    return yaml_dict


def get_animals_from_yaml(directory):
    root_dir = directory if directory else ""
    root_yaml_path = os.path.join(root_dir, manually_eddited_animals_yaml_fname)
    if os.path.exists(root_yaml_path):

        with open(root_yaml_path, "r") as yaml_file:
            animals = yaml.safe_load(yaml_file)
    else:
        animals = {}

    """
    for animal_id in get_directories(root_dir, regex_search="DON-"):
        animal_path = os.path.join(root_dir, animal_id)

        # update animals if a file has newer information changed
        animal_yaml_path = os.path.join(animal_path, animal_id+".yaml")
        animal = return_loaded_yaml_if_newer(root_yaml_path, animal_yaml_path)
        animal
        animals[animal_id] = animal if animal else animals[animal_id]
    
    to_delete_animals = []
    to_delete_keys = ["pdays", "cohort_year", "functional_channels", "sex", "session_dates", "session_names", "dob"]
    for animal_id, animal in animals.items():
        if 'UseMUnits' not in animal.keys():
            to_delete_animals.append(animal_id)
            continue
        for to_delete_key in to_delete_keys:
            if to_delete_key in animal.keys():# and 'UseMUnits' in animal.keys():
                del animal[to_delete_key]
    for to_delete_animal in to_delete_animals:
        del animals[to_delete_animal]"""
    return animals


def combine_spreadsheet_and_old_animal_summary_yaml(animals_spreadsheet, animals_yaml):
    import copy

    for yanimal_id, yanimal in animals_yaml.items():
        for date, ysession in yanimal["sessions"].items():

            if date not in animals_spreadsheet[yanimal_id]["sessions"].keys():

                animals_spreadsheet[yanimal_id]["sessions"][date] = create_session_dict(
                    date=date, method="2P", setup="femtonics"
                )

            if "UseMUnits" in ysession.keys():
                usemunits = ysession["UseMUnits"]
                if not usemunits:
                    continue
                else:
                    animals_spreadsheet[yanimal_id]["sessions"][date][
                        "UseMUnits"
                    ] = usemunits
    return animals_spreadsheet


def add_session_animal_folders(animals, animals_spreadsheet, directory=None):
    root_dir = directory if directory else ""
    for animal_id in get_directories(root_dir, regex_search="DON-"):
        if animal_id not in animals:
            animals[animal_id] = animals_spreadsheet[animal_id]
            print(f"added animal from spreadsheet: {animal_id}")
        animal_path = os.path.join(root_dir, animal_id)

        functional_channel_list = []
        for session_id in get_directories(animal_path):
            session_path = os.path.join(animal_path, session_id, "002P-F")

            mesc_fnames = get_files(session_path, ending=".mesc")
            mesc_munit_pairs = (
                animals[animal_id]["UseMUnits"]
                if "UseMUnits" in animals[animal_id].keys()
                else []
            )
            for fname in mesc_fnames:
                splitted_fname = fname.split("_")
                if animal_id != splitted_fname[0]:
                    continue

                # add session data based on file
                session_date = splitted_fname[1]
                if (
                    session_id not in animals[animal_id]["session_names"]
                    and session_date not in animals[animal_id]["session_dates"]
                ):
                    animals[animal_id]["session_names"].append(session_id)
                    animals[animal_id]["session_dates"].append(session_date)
                    dob_date = num_to_date(animals[animal_id]["dob"])
                    session_date = num_to_date(session_date)
                    pday = (session_date - dob_date).days
                    animals[animal_id]["pdays"].append(pday)

                # get available MUnits
                last_fname_part = splitted_fname[-1].split(".")[0]
                session_parts = [
                    int(part_number[-1]) - 1
                    for part_number in re.findall("S[0-9]", last_fname_part)
                ]
                fpath = os.path.join(session_path, fname)
                munits_list, number_channels = get_recording_munits(
                    fpath, session_parts
                )

                # Get MUnit number list of first Mescfile session MSession_0
                if len(munits_list) <= len(session_parts):
                    usefull_munits = munits_list
                    file_naming = session_parts[: len(usefull_munits)]
                else:
                    add_mesc_munit_pair = True
                    if mesc_munit_pairs:
                        if len(mesc_munit_pairs) > 0:
                            for mesc_munit_pair in mesc_munit_pairs:
                                if fname in mesc_munit_pair:
                                    add_mesc_munit_pair = False
                        if add_mesc_munit_pair:
                            mesc_munit_pair = [fname, munits_list]
                            mesc_munit_pairs.append(mesc_munit_pair)

                functional_channel = (
                    2
                    if 20210821 < int(session_date)
                    and int(session_date) < 20220422
                    and number_channels > 1
                    else 1
                )
                functional_channel_list.append(functional_channel)

            # TODO: integrate functional channels into individual yaml files
            animals[animal_id]["functional_channels"] = functional_channel_list

            if mesc_munit_pairs:
                if len(mesc_munit_pairs) > 0:
                    animals[animal_id]["UseMUnits"] = mesc_munit_pairs
    return animals


def get_recording_munits(
    mesc_fpath, session_parts, fps=30, at_least_minutes_of_recording=5
):
    # Get MUnit number list of first Mescfile session MSession_0
    with h5py.File(mesc_fpath, "r") as file:
        munits = file[list(file.keys())[0]]
        recording_munits = []
        for name, unit in munits.items():
            # if recording has at least x minutes
            if unit["Channel_0"].shape[0] > fps * 60 * at_least_minutes_of_recording:
                unit_number = name.split("_")[-1]
                recording_munits.append(int(unit_number))
                # get number of imaging channels
                number_channels = 0
                for key in unit.keys():
                    if "Channel" in key:
                        number_channels += 1
    return recording_munits, number_channels


def move_mesc_to_session_folder(directory=None):
    directory = None if directory == "" else directory
    for fname in get_files(directory, ending=".mesc"):
        splitted_fname = fname.split("_")
        if splitted_fname[0][:3] != "DON":  # not animal
            continue
        animal_id = splitted_fname[0]
        session_id = splitted_fname[1].split(".")[0]
        session_path = create_folder(
            str(animal_id), str(session_id), directory=directory
        )
        # move_file
        fpath = os.path.join(directory, fname)
        shutil.move(fpath, session_path)


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


def add_yaml_to_folders(animals, directory):
    # saving session yaml files
    for aid, animal in animals.items():
        for sid, sess in animal["sessions"].items():
            yaml_fname = f"{sid}.yaml"
            yaml_path = os.path.join(directory, aid, sid)
            if not os.path.exists(yaml_path):
                print(f"No path found. Skipping: {yaml_path}")
                continue
            save_path = os.path.join(yaml_path, yaml_fname)
            with open(save_path, "w") as file:
                yaml.dump(sess, file)
            print(save_path)

    # saving animal yaml files
    for aid, animal in animals.items():
        yaml_fname = f"{aid}.yaml"
        yaml_path = os.path.join(directory, aid)
        if not os.path.exists(yaml_path):
            print(f"No path found. Skipping: {yaml_path}")
            continue
        save_path = os.path.join(yaml_path, yaml_fname)
        animal_copy = copy.deepcopy(animal)
        del animal_copy["sessions"]
        with open(save_path, "w") as file:
            yaml.dump(animal_copy, file)
        print(save_path)


def update_excel_by_yaml(excel_animals, yaml_animals):
    pass
    # FIXME: continue


def main(directory=None):
    root_dir = directory if directory else ""
    fname = os.path.join("Intrinsic_CA3_database-September_7,_10_08_AM.xlsx")
    fpath = os.path.join(root_dir, fname)
    # load spreadsheet information
    animals_spreadsheet = get_animal_dict_from_spreadsheet(fpath)
    # move mesc in root directory to correct folder location
    move_mesc_to_session_folder(directory=root_dir)
    # load animal yaml files
    animals_yaml = get_animals_from_yaml(root_dir)
    animals = combine_spreadsheet_and_old_animal_summary_yaml(
        animals_spreadsheet, animals_yaml
    )
    # get animals based on folder structure
    # TODO: old stuff may create new version
    # print(ooold)
    # animals = add_session_animal_folders(
    #    animals_yaml, animals_spreadsheet, directory=root_dir
    # )
    add_yaml_to_folders(animals, directory=root_dir)
    # with open(os.path.join(root_dir, manually_eddited_animals_yaml_fname), "w") as file:
    #    yaml.dump(animals, file)
    # save yaml files in folders and root directory
    with open(os.path.join(root_dir, "animals.yaml"), "w") as file:
        yaml.dump(animals, file)


if __name__ == "__main__":
    root_dir = "/scicore/projects/donafl00-calcium/Users/Sergej/Steffen_Experiments"
    main()
