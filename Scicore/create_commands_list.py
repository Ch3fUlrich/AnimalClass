# imports
# Statistics

import numpy as np
import os
import sys

# add root directory to be able to import packages
# todo: make all packages installable so they can be called/imported by environment
module_path = os.path.abspath(os.path.join("../"))
sys.path.append(module_path)

from Classes import Animal, load_all
from Helper import *

# set correct root directory for your animal data
# root_dir = "\\\\toucan-all.scicore.unibas.ch\\donafl00-calcium$\\Users\\Sergej\\Steffen_Experiments"
# root_dir = "/scicore/projects/donafl00-calcium/Users/Sergej/Steffen_Experiments"
mice_dict = {
    "mice21": [
        "DON-002865",
        "DON-003165",
        "DON-003343",
        "DON-006084",
        "DON-006085",
        "DON-006087",
    ],
}


def create_commands_file(
    commands_fname="commands.cmd",
    wanted_animal_ids=["all"],
    wanted_session_ids=["all"],
    skip_animal=[],
    skip_session=[],
    project_root_dir=None,
    mesc_to_tiff=True,
    suite2p=True,
    binarize=True,
    pairwise_correlate=False,
):
    Animal.root_dir = project_root_dir
    # TODO: skipping option is not integrated

    pipeline_settings = str([mesc_to_tiff, suite2p, binarize, pairwise_correlate])[
        1:-1
    ].replace(",", "")

    with open(commands_fname, "w") as f:
        animals = load_all(
            project_root_dir,
            wanted_animal_ids=wanted_animal_ids,
            wanted_session_ids=wanted_session_ids,
        )  # Load all animals
        for animal_id, animal in animals.items():
            if animal_id in wanted_animal_ids or "all" in wanted_animal_ids:
                for session_id, session in animal.sessions.items():
                    if session_id in wanted_session_ids or "all" in wanted_session_ids:
                        python_path = os.path.join(
                            os.getcwd(), "AnimalClass_command_line.py"
                        )
                        f.write(
                            f"python {python_path} {animal_id} {session_id} {pipeline_settings}\n"
                        )

    return commands_fname


def main(
    wanted_animal_ids=["all"],
    wanted_session_ids=["all"],
    skip_animal=[],
    skip_session=[],
    project_root_dir=None,
):
    create_commands_file(
        wanted_animal_ids=wanted_animal_ids,
        wanted_session_ids=wanted_session_ids,
        skip_animal=skip_animal,
        skip_session=skip_session,
        project_root_dir=project_root_dir,
    )


if __name__ == "__main__":
    root_dir = "/scicore/projects/donafl00-calcium/Users/Sergej/Steffen_Experiments"
    print(
        "Command line usage: <animal_id> <session_id>, multiple parameters can be used seperated by spaces"
    )
    print("Animal_id must start with ´DON´")
    print("If an argument is not specified the corresponding argument is set to 'all'")

    arguments = sys.argv[1:]
    wanted_animal_ids = []
    wanted_session_ids = []
    for argument in arguments:
        argument = str(argument)
        if argument != "all":
            if argument[:3] == "DON":
                wanted_animal_ids.append(argument)
            elif argument[:4] == "mice":
                wanted_animal_ids += mice_dict[argument]
            else:
                wanted_session_ids.append(argument)

    wanted_animal_ids = wanted_animal_ids if len(wanted_animal_ids) > 0 else ["all"]
    wanted_animal_ids = np.unique(wanted_animal_ids)
    print(wanted_animal_ids)
    wanted_session_ids = wanted_session_ids if len(wanted_session_ids) > 0 else ["all"]
    print(f"Creating commands.cmd for {wanted_animal_ids}, {wanted_session_ids}")
    main(
        wanted_animal_ids, wanted_session_ids, project_root_dir=root_dir
    )  # skip_animal=["DON-009191"], skip_session=["20220225"]
