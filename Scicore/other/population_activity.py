# interact with system
import os
import sys
from pathlib import Path
import numpy as np

root_path = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent
Loader_path = root_path.joinpath("Tools", "Loader")
print(Loader_path)
sys.path.append(str(root_path))
sys.path.append(str(Loader_path))

from Classes import population_similarity
from Loader import Animal, Session
from Helper import *

# Init Directories and Notebook settings
root_dir = Path(
    #"\\\\toucan-all.scicore.unibas.ch\\donafl00-calcium$\\Users\\Sergej\\Steffen_Experiments\\rodrigo"
    "/scicore/projects/donafl00-calcium/Users/Sergej/Steffen_Experiments/rodrigo"
)
# root_dir = 
Animal.root_dir = root_dir


def main(wanted_animal_id, wanted_session_id):
    print_loading = False
    animals = {}
    for animal_id, session_id in zip(wanted_animal_id, wanted_session_id):
        sessions_root_path = root_dir.joinpath(animal_id)
        if animal_id not in animals:
            yaml_file_name = sessions_root_path.joinpath(f"{animal_id}.yaml")
            animal = Animal(str(yaml_file_name), print_loading=print_loading)
            Animal.root_dir = root_dir
            animals[animal_id] = animal
            session_path = sessions_root_path.joinpath(session_id)
            animal.get_session_data(str(session_path), print_loading=print_loading)
        else:
            session_path = sessions_root_path.joinpath(session_id)
            animal.get_session_data(str(session_path), print_loading=print_loading)

    for animal_id, animal in animals.items():
        print(f"{animal_id}: {list(animal.sessions.keys())}")

    get_population_activity(animals)


def get_population_activity(animals):
    """
    Calculate the population activity for all animals and sessions.

    Get simultaneously active cells, cosine similarity and pearson correlation for all animals and sessions.
    Saves the data in the corresponding folders, matrices are compressed and saved as .npz files.

    Parameters
    ----------
    animals : dict
        Dictionary containing all animals

    Returns
    -------
    None
    """
    root_folder_path = Path(
        "/scicore/projects/donafl00-calcium/Users/Sergej/Steffen_Experiments/everton/population_activity/"
        #"\\\\toucan-all.scicore.unibas.ch\\donafl00-calcium$\\Users\\Sergej\\Steffen_Experiments\\everton\\population_activity"
    )
    for animal_id, animal in animals.items():
        for session_id, session in animal.sessions.items():
            frames = 5000
            # movement_types = ["all", "stationary", "moving"]
            movement_types = ["all"]  # , "stationary", "moving"]
            for movement_type in movement_types:
                print(f"Movement Type: {movement_type}")
                mtype = movement_type if movement_type != "all" else None
                num_coactive_cells = session.get_num_coactive_cells(
                    clean=True, movement_type=mtype
                )
                print("coactive cells", sum(num_coactive_cells))

                bin_fluorescence = session.load_binarized_traces(clean=True)
                upphase = session.filter_by_movement(bin_fluorescence, condition=mtype)
                upphase = upphase[:, :frames]
                print("Generating cosine similarities")
                cosine_similarity = population_similarity(
                    upphase, axis=1, metric="cosine", plot=False
                )
                print("Generating pearson correlations")
                pearson_correlations = population_similarity(
                    upphase, axis=1, metric="pearson", plot=False
                )

                # save data
                print("Saving data")
                folder_path = root_folder_path.joinpath("files", movement_type)
                cosine_path = folder_path.joinpath("cosine_similarity")
                pearson_path = folder_path.joinpath("pearson_correlations")
                num_coactive_path = folder_path.joinpath("num_coactive_cells")

                cosine_path.mkdir(parents=True, exist_ok=True)
                pearson_path.mkdir(parents=True, exist_ok=True)
                num_coactive_path.mkdir(parents=True, exist_ok=True)

                session_name = f"{animal_id}_{session_id}_{movement_type}"

                cosine_data_name = f"{session_name}_cosine_similarity"
                pearson_data_name = f"{session_name}_pearson_correlations"
                num_coactive_data_name = f"{session_name}_num_coactive_cells"

                np.savez_compressed(
                    str(cosine_path.joinpath(f"{cosine_data_name}")), cosine_similarity
                )
                np.savez_compressed(
                    str(pearson_path.joinpath(f"{pearson_data_name}")),
                    pearson_correlations,
                )
                np.savez_compressed(
                    str(num_coactive_path.joinpath(f"{num_coactive_data_name}")),
                    num_coactive_cells,
                )
                print("----------------------------------------------")


if __name__ == "__main__":
    arguments = sys.argv[1:]
    wanted_animal_id = sys.argv[1:2]
    wanted_session_id = sys.argv[2:3]
    if len(arguments) > 3:
        print("Command line usage: <animal_id> <session_id>")
        print(
            "If an argument is not specified the corresponding argument is set to 'all'"
        )
    print(
        f"Generating population activity for animal {wanted_animal_id} and session {wanted_session_id}"
    )
    main(wanted_animal_id=wanted_animal_id, wanted_session_id=wanted_session_id)
    # main(wanted_animal_id=["DON-015078"], wanted_session_id=["20230217"])
