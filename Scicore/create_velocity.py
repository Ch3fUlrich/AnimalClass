# interact with system
import os
import sys

# add root directory to be able to import packages
# todo: make all packages installable so they can be called/imported by environment
module_path = os.path.abspath(os.path.join('../'))
sys.path.append(module_path)

from Classes import *
from Helper import *

# Init Directories and Notebook settings
#root_dir = "\\\\toucan-all.scicore.unibas.ch\\donafl00-calcium$\\Users\\Sergej\\Steffen_Experiments"  
root_dir = "/scicore/projects/donafl00-calcium/Users/Sergej/Steffen_Experiments"  
Animal.root_dir = root_dir

mice21 = ["DON-002865", "DON-003165", "DON-003343", "DON-006084", "DON-006085", "DON-006087"]
mice22 = ["DON-008497", "DON-008498", "DON-008499", "DON-009191", "DON-009192", "DON-010473", "DON-010477"]
mice23 = ["DON-014837", "DON-014838", "DON-014840", "DON-014847", "DON-014849", "DON-015078", "DON-015079"]

def main(wanted_animal_ids = ["all"], wanted_session_ids=["all"]):
    animals = load_all(root_dir, wanted_animal_ids=wanted_animal_ids, wanted_session_ids=wanted_session_ids) # Load all animals

    for animal_id, animal in animals.items():
        print(f"{animal_id}: {list(animal.sessions.keys())}")

    create_velo(animals)

def create_velo(animals):
    for animal_id, session_id, session in yield_animal_session(animals):
        min_usefull_cells = 100 if session.animal_id in mice21+mice22 else 80 if session.animal_id in mice23 else None
        if not min_usefull_cells:
            print(f"{animal_id} {session_id} No min_usefull_cells Skipping...")
            continue
        session.get_units(restore=True, get_geldrying=False, unit_type="summary", generate=False, regenerate=False)
        #session.convert_movement_data() # Already done in merge_movements
        session.get_units(restore=True, get_geldrying=False, unit_type="single", generate=False, regenerate=False)
        wheel, triggers, velocity = session.load_movements(merged=True, min_num_usefull_cells=80, regenerate=True,
                                                           movement_data_types=["wheel", "triggers", "velocity"])
        if "merged" not in session.units.keys():
            continue
        fluor_fpath = os.path.join(session.units["merged"].suite2p_dir, Session.fluoresence_fname)
        if os.path.exists(fluor_fpath) and type(velocity)==np.ndarray:
            fluoresence = np.load(fluor_fpath)
            print(f"Fluoresence shape: {fluoresence.shape}")
            print(f"velocity shape: {velocity.shape}")
        else:
            print(f"ERRRRRRRRRRRRRRRRRRRRROOOOOOOOOOOOOOOORRRRRRRRRRRRRRRR")


if __name__ == "__main__":
    arguments = sys.argv[1:]
    wanted_animal_ids = sys.argv[1:2] if len(arguments) >= 1 else ["all"]
    wanted_session_ids = sys.argv[2:3] if len(arguments) >= 2 else ["all"]
    if len(arguments) > 3:
        print("Command line usage: <animal_id> <session_id>")
        print("If an argument is not specified the corresponding argument is set to 'all'")
    print(f"Start Cleaning {wanted_animal_ids}, {wanted_session_ids}")
    main(wanted_animal_ids=wanted_animal_ids, wanted_session_ids=wanted_session_ids)#skip_animal=["DON-009191"], skip_session=["20220225"]
