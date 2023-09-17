import yaml
import os
import shutil
from datetime import datetime
from openpyxl import load_workbook, Workbook

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

def create_animal_dict(animal_id_records, dob_records, sex_records):
    animals = {}
    for animal_id, dob, sex in zip(animal_id_records, dob_records, sex_records):
        animal_id = "DON-00"+animal_id[3:] if len(animal_id) == 7 else "DON-0"+animal_id[3:]
        if animal_id in animals:
            continue

        dob = "20"+str(int(dob))
        dob_date = num_to_date(dob)

        animals[animal_id] = {
            'cohort_year': dob_date.year,
            'dob': dob,
            'name': animal_id,
            'pdays': [],
            'session_dates': [],
            'session_names': [],
            'sex': "male" if sex == "m" else "female"
        }
    return animals

def add_session_info_from_file_move_file(animals, directory=None):
    for file in os.listdir(directory):
        if os.path.isfile(file): #is file
            splitted_fname = file.split("_")
            if splitted_fname[0][:3] != "DON": #not animal file
                continue
            animal_id = splitted_fname[0]
            session_id = splitted_fname[1].split(".")[0]
            if animal_id in animals:
                session_date = session_id
                animals[animal_id]["session_dates"].append(session_id)
                animals[animal_id]["session_names"].append(session_id)
                
                dob_date = num_to_date(animals[animal_id]["dob"])
                session_date = num_to_date(session_date)
                pday = (session_date-dob_date).days
                animals[animal_id]["pdays"].append(pday)

                session_path = create_folder(str(animal_id), str(session_id))
                #move_file 
                shutil.move(file, session_path)
    return animals

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

def create_folder(animal_id, session_id):
    dir_exist_create(animal_id)
    path = os.path.join(animal_id, session_id)
    dir_exist_create(path)
    path = os.path.join(path, "002P-F")
    dir_exist_create(path)
    return path

def add_yaml_to_folders(animals):
    for animal_id, animal in animals.items():
        animal_path = os.path.join(str(animal_id))
        yaml_path = os.path.join(animal_path, f"{animal_id}.yaml")
        if os.path.exists(animal_path):
            with open(yaml_path, "w") as f:
                yaml.dump(animal, f)

def main():
    fname = os.path.join("Intrinsic_CA3_database-September_7,_10_08_AM.xlsx")
    animals = get_animal_dict_from_spreadsheet(fname)
    animals = add_session_info_from_file_move_file(animals)
    add_yaml_to_folders(animals)
    with open('intrinsic_imaging.yaml', 'w') as file:
        yaml.dump(animals, file)

if __name__ == "__main__":
    main()