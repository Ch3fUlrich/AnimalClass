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
mice23 = ["DON-014837", "DON-014838", "DON-014840", "DON-014847", "DON-014849", "DON-015078", "DON-015079",
          "DON-017115", "DON-017117", "DON-017118", "DON-019207", "DON-019210", "DON-019213", "DON-019542", 
          "DON-019545", "DON-019608", "DON-019609", "DON-019747", "DON-019748"]

def main(wanted_animal_ids = ["all"], wanted_session_ids=["all"]):
    animals = load_all(root_dir, wanted_animal_ids=wanted_animal_ids, wanted_session_ids=wanted_session_ids) # Load all animals

    for animal_id, animal in animals.items():
        print(f"{animal_id}: {list(animal.sessions.keys())}")

    create_is_wheel(animals)

def create_is_wheel(animals):
    for animal_id, session_id, session in yield_animal_session(animals):
        session.merged_underground_is_wheel()

if __name__ == "__main__":
    arguments = sys.argv[1:]
    wanted_animal_ids = sys.argv[1:2] if len(arguments) >= 1 else ["all"]
    wanted_session_ids = sys.argv[2:3] if len(arguments) >= 2 else ["all"]
    if len(arguments) > 3:
        print("Command line usage: <animal_id> <session_id>")
        print(
            "If an argument is not specified the corresponding argument is set to 'all'"
        )
    print(f"Start merging {wanted_animal_ids}, {wanted_session_ids}")
    main(wanted_animal_ids=wanted_animal_ids, wanted_session_ids=wanted_session_ids)
