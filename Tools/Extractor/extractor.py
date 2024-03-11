# Imports
import os
import sys
import numpy as np
import yaml
import shutil

# add root directory to be able to import packages
# todo: make all packages installable so they can be called/imported by environment
module_path = os.path.abspath(os.path.join("../"))
sys.path.append(module_path)
module_path = os.path.abspath(os.path.join("../../"))
sys.path.append(module_path)

from Classes import *
from Helper import *


class information_extractor:
    def __init__(
        self,
        root_dir,
        save_dir="extracted",
        wanted_animal_ids=["all"],
        wanted_session_ids=["all"],
        print_loading=True,
    ):
        Animal.root_dir = (
            root_dir  # TODO: could make problems if using multiple extractors
        )
        self.root_dir = root_dir
        self.save_path = os.path.join(root_dir, save_dir)
        dir_exist_create(self.save_path)
        self.animals = load_all(
            root_dir,
            wanted_animal_ids=wanted_animal_ids,
            wanted_session_ids=wanted_session_ids,
            print_loading=print_loading,
        )

    def enough_cells_session(self, session, min_number_cells=200):
        num_cells = 0
        for suite2p_dir in session.suite2p_dirs:
            if "merged" in suite2p_dir:
                fpath = search_file(suite2p_dir, "cell_drying.npy")
                if fpath:
                    geldrying = np.load(fpath)
                    num_cells = geldrying[geldrying == False].shape[0]
        return True if num_cells >= min_number_cells else False

    def cabincorr(self, folder_name_content="merged", animals=None):
        animals = animals or self.animals
        for animal_id, session_id, session in yield_animal_session(animals):
            if not self.enough_cells_session(session):
                continue
            for cabincorr_data_path in session.cabincorr_data_paths:
                if folder_name_content in cabincorr_data_path:
                    if os.path.exists(session.cabincorr_fpath):
                        create_dirs([self.save_path, animal_id, session_id])
                        new_path = create_dirs(
                            [self.save_path, session.animal_id, session.session_id]
                        )
                        save_file_path = os.path.join(new_path, Session.cabincorr_fname)
                        if not save_file_present(save_file_path):
                            shutil.copy(session.cabincorr_fpath, new_path)
                    else:
                        print(
                            f"No {Session.cabincorr_fname} found in {session.cabincorr_fpath}"
                        )

    def from_cabincorr(self, unit_id, data_types, animals=None, override=False):
        data_types = make_list_ifnot(data_types)
        animals = animals or self.animals
        for animal_id, session_id, session in yield_animal_session(animals):
            if not self.enough_cells_session(session):
                continue
            for data_type in data_types:
                save_file_path = os.path.join(
                    self.save_path, animal_id, session_id, data_type + ".npy"
                )
                if override or not save_file_present(save_file_path):
                    create_dirs([self.save_path, animal_id, session_id])
                    bin_traces_zip = session.load_cabincorr_data(unit_id=unit_id)
                    if not bin_traces_zip:
                        print(
                            f"No CaBincorrPath found for unit {unit_id} in {session.session_dir}"
                        )
                        break
                    data = bin_traces_zip[data_type]
                    np.save(save_file_path, data)

    def yaml(self, animals=None, override=False):
        animals = animals or self.animals
        for animal_id, animal in animals.items():
            yaml_fname = animal.animal_id + ".yaml"
            old_path = os.path.join(animal.animal_dir, yaml_fname)
            new_path = create_dirs([self.save_path, animal_id])
            new_file_path = os.path.join(new_path, yaml_fname)
            if override or not save_file_present(new_file_path):
                shutil.copy(old_path, new_path)
            for session_id, session in animal.sessions.items():
                if not self.enough_cells_session(session):
                    continue
                yaml_fname = session.date + ".yaml"
                old_path = os.path.join(animal.animal_dir, session.date, yaml_fname)
                new_path = create_dirs([self.save_path, animal_id, session.date])
                new_file_path = os.path.join(new_path, yaml_fname)
                if override or not save_file_present(new_file_path):
                    shutil.copy(old_path, new_path)

    def from_suite2p_folder(
        self, file_names, folder_name_content="merged", animals=None, override=False
    ):
        animals = animals or self.animals
        for file_name in file_names:
            for animal_id, session_id, session in yield_animal_session(animals):
                if not self.enough_cells_session(session):
                    continue
                if session.suite2p_dirs:
                    for suite2p_dir in session.suite2p_dirs:
                        if folder_name_content in suite2p_dir:
                            fpath = search_file(suite2p_dir, file_name)
                            if fpath:
                                save_path = create_dirs(
                                    [self.save_path, animal_id, session_id]
                                )
                                save_file_path = os.path.join(save_path, file_name)
                                if override or not save_file_present(save_file_path):
                                    shutil.copy(fpath, save_path)
                else:
                    print(f"No Suite2p paths found in {session.session_dir}")

    def fps_mesc(self):
        file_name = "fps.npy"
        for animal_id, session_id, session in yield_animal_session(self.animals):
            if not self.enough_cells_session(session):
                continue
            frame_rate = session.get_mesc_fps()
            create_dirs([self.save_path, animal_id, session_id])
            save_file_path = os.path.join(
                self.save_path, animal_id, session_id, file_name
            )
            if not save_file_present(save_file_path):
                # print(f"Saving FPS: {animal_id} {session_id}: {frame_rate:.10}")
                np.save(save_file_path, frame_rate)

    def from_movement(
        self, animals=None, merged=True, movement_types="velocity", override=False
    ):
        movement_types = make_list_ifnot(movement_types)
        animals = animals or self.animals
        for animal_id, session_id, session in yield_animal_session(animals):
            if not self.enough_cells_session(session):
                continue
            for movement_type in movement_types:
                fname = (
                    f"merged_{movement_type}.npy" if merged else f"{movement_type}.npy"
                )
                old_path = os.path.join(session.movement_dir, fname)
                if not os.path.exists(old_path):
                    print(
                        f"{animal_id} {session_id}: {movement_type} path does not exist {old_path}"
                    )
                    continue
                new_dir = create_dirs([self.save_path, animal_id, session.date])
                new_file_path = os.path.join(new_dir, fname)
                if override or not save_file_present(new_file_path):
                    shutil.copy(old_path, new_dir)

    def delete_session(self, animal_id, session_id):
        rmdir = None
        if animal_id in self.animals.keys():
            if session_id in self.animals[animal_id].sessions.keys():
                session = self.animals[animal_id].sessions[session_id]
                rmdir = session.session_dir
                del self.animals[animal_id].sessions[session_id]
        else:
            print(f"No session {session_id} found in animal {animal_id}")
            rmdir = os.path.join(self.root_dir, animal_id, session_id)
            if not os.path.exists(rmdir):
                print(f"No directory found {rmdir}")
        if rmdir:
            shutil.rmtree(rmdir)
