# work with files
import yaml
import copy
import shutil

# do Math stuff
import numpy as np
import re
import sys
import os

# Suite2p for TIFF file analysis
import suite2p
from suite2p.run_s2p import run_s2p, default_ops
from suite2p.registration import register

module_path = os.path.abspath(os.path.join("../"))
sys.path.append(module_path)
from Helper import *
from manifolds.donlabtools.utils.calcium import calcium
from manifolds.donlabtools.utils.calcium.calcium import *


def load_all(
    root_dir,
    wanted_animal_ids=["all"],
    wanted_session_ids=["all"],
    restore=False,
    regenerate=False,
    print_loading=True,
):
    """
    Loads animal data from the specified root directory for the given animal IDs.

    Parameters:
    - root_dir (string): The root directory path where the animal data is stored.
    - animal_ids (list, optional): A list of animal IDs to load. Default is ["all"].
    - generate (bool, optional): If True, generates new session data. Default is False.
    - regenerate (bool, optional): If True, regenerates existing session data. Default is False.
    - sessions (string, optional): Specifies the sessions. Default is "single".
    - delete (bool, optional): If True, deletes session data. Default is False.

    Returns:
    - animals_dict (dict): A dictionary containing animal IDs as keys and corresponding Animal objects as values.
    """
    present_animal_ids = get_directories(root_dir, regex_search="DON-")
    animals_dict = {}

    # Search for animal_ids
    for animal_id in present_animal_ids:
        if animal_id in wanted_animal_ids or "all" in wanted_animal_ids:
            sessions_root_path = os.path.join(root_dir, animal_id)
            present_sessions = get_directories(sessions_root_path)
            yaml_file_name = os.path.join(root_dir, animal_id, f"{animal_id}.yaml")
            animal = Animal(yaml_file_name, print_loading=print_loading)
            Animal.root_dir = root_dir
            # Search for 2P Sessions
            for session in present_sessions:
                if session in wanted_session_ids or "all" in wanted_session_ids:
                    session_path = os.path.join(sessions_root_path, session)
                    animal.get_session_data(
                        session_path,
                        restore=restore,
                        regenerate=regenerate,
                        print_loading=print_loading,
                    )
            animals_dict[animal_id] = animal
    animals_dict = {
        animal_id: animal for animal_id, animal in sorted(animals_dict.items())
    }
    return animals_dict


def run_cabin_corr(
    root_dir, data_dir, animal_id, session_id, regenerate=False, parallel=True
):
    # Init
    print(f"Getting cabincorr data from {data_dir}")
    cabincorr_path = os.path.join(data_dir, "binarized_traces.npz")
    if regenerate:
        del_file_dir(cabincorr_path)
    c = calcium.Calcium(root_dir, animal_id, session_name=session_id, data_dir=data_dir)

    c.parallel_flag = parallel
    c.animal_id = animal_id
    c.detrend_model_order = 1
    c.recompute_binarization = False
    c.remove_ends = False
    c.remove_bad_cells = True
    c.detrend_filter_threshold = 0.001
    c.mode_window = 30 * 30
    c.percentile_threshold = 0.000001
    c.dff_min = 0.02
    c.data_type = "2p"
    #
    c.load_suite2p()

    c.load_binarization()

    # getting contours and footprints
    c.load_footprints()
    return c


def backup_path_files(
    data_path, backup_folder_name="backup", redo_backup=False, restore=False
):
    data_path = os.path.join(data_path)
    backup_path = os.path.join(data_path, backup_folder_name)
    if restore:
        if os.path.exists(backup_path):
            shutil.copytree(backup_path, data_path, dirs_exist_ok=True)
            print(f"Restored original suite2p files")
        else:
            print(f"No backup found at {backup_path}")
    else:
        if not os.path.exists(backup_path):
            shutil.copytree(data_path, backup_path)
        else:
            if redo_backup:
                del_file_dir(backup_path)
                shutil.copytree(data_path, backup_path)
            else:
                print("Backup path already exists. Skipping")


def make_list_ifnot(string_or_list):
    return [string_or_list] if type(string_or_list) != list else string_or_list


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
    if not binary_fpath:
        print(
            f"No binary path to {possible_binary_fnames} found in {possible_binary_data_paths}"
        )
    return binary_fpath


# classes
class Animal:
    """
    This class represents an animal in an experiment.

    Attributes:
    root_dir (str): The root directory where the data is stored.
    sessions (dict): A dictionary to store session objects for this animal.
    cohort_year (int): The year of the cohort that the animal belongs to.
    dob (str): The date of birth of the animal.
    animal_id (str): The ID of the animal.
    sex (str): The sex of the animal.

    Methods:
    load_metadata(yaml_path): Loads metadata for the animal from a YAML file.
    get_session_data(session_id, print_loading=True): Loads data for a specific session.
    """

    root_dir = "D:\\Animals"

    def __init__(self, yaml_file_path, print_loading=True) -> None:
        self.sessions = {}
        self.cohort_year = None
        self.dob = None
        self.animal_id = None
        self.sex = None
        self.load_metadata(yaml_file_path)
        self.animal_dir = os.path.join(Animal.root_dir, self.animal_id)
        if print_loading:
            print(f"Added animal: {self.animal_id}")

    def load_metadata(self, yaml_path):
        with open(yaml_path, "r") as yaml_file:
            animal_metadata_dict = yaml.safe_load(yaml_file)

        # Load any additional metadata into session object
        for variable_name, value in animal_metadata_dict.items():
            setattr(self, variable_name, value)

        cohort_year = animal_metadata_dict["cohort_year"]
        self.cohort_year = (
            int(cohort_year)
            if type(cohort_year) == str
            else int(cohort_year[0])
            if type(cohort_year) == list
            else cohort_year
        )
        self.dob = animal_metadata_dict["dob"]
        self.animal_id = animal_metadata_dict["name"]
        self.sex = animal_metadata_dict["sex"]
        return animal_metadata_dict

    def get_session_data(
        self,
        path,
        restore=False,
        generate=False,
        regenerate=False,
        delete=False,
        print_loading=True,
    ):
        session_yaml_fnames = get_files(path, ending=".yaml")
        match = None
        if session_yaml_fnames:
            for session_yaml_fname in session_yaml_fnames:
                match = True
                session_yaml_path = os.path.join(path, session_yaml_fname)
                session = Session(
                    session_yaml_path,
                    animal_id=self.animal_id,
                    print_loading=print_loading,
                )
                if (
                    str(session.date) not in session_yaml_fname
                    and not session.session_type == "merged"
                ):
                    print(
                        f"Yaml file naming does not match session date: {session_yaml_fname} != {session.date}"
                    )
                    match = False
                if match:
                    session.pday = (
                        (num_to_date(session.date) - num_to_date(self.dob)).days
                        if session.session_type != "merged"
                        else None
                    )
                    session.load_data(
                        restore=restore,
                        generate=generate,
                        regenerate=regenerate,
                        delete=delete,
                    )
                    break
                else:
                    print(f"Reading next yaml file")
        if match:
            self.sessions[session.session_id] = session
            self.sessions = {
                session_id: session
                for session_id, session in sorted(self.sessions.items())
            }
        else:
            print(f"No matching yaml file found. Skipping session path {path}")

    def merge_sessions(
        self,
        reference_session_id="day0",
        restore=False,
        regenerate=False,
        n_frames_to_be_acquired=1000,
        num_align_frames=1000,
    ):
        reference_session = self.sessions[reference_session_id]
        sessions = self.sessions
        yaml_fname = f"{reference_session.animal_id}_merged.yaml"
        merged_session_dir = os.path.join(self.animal_dir, "merged")
        merged_yaml_path = os.path.join(merged_session_dir, yaml_fname)

        # check if already all, merged, suite2p files are present
        suite2p_files_list = [
            "F.npy",
            "Fneu.npy",
            "iscell.npy",
            "ops.npy",
            "spks.npy",
            "stat.npy",
        ]

        merged_s2p_path = os.path.join(merged_session_dir, "plane0")
        for s2p_file in suite2p_files_list:
            s2p_file_path = search_file(merged_s2p_path, s2p_file)
            if not s2p_file_path:
                merged_s2p_path = None
                break

        print(f"Generating yx-shifts based on reference session {reference_session_id}")
        for session_id, session in sessions.items():
            if session_id == reference_session.session_id:
                continue
            session.set_yx_shift(
                reference_session=reference_session,
                n_frames_to_be_acquired=n_frames_to_be_acquired,
                num_align_frames=num_align_frames,
            )

        # merge masks, updated sessions based on merged masks
        if regenerate or not merged_s2p_path:
            merger = Merger()
            if merged_s2p_path:
                print(f"removing old merged suite2p files from {merged_s2p_path}")
                shutil.rmtree(merged_s2p_path)

            # reload original session files
            for session_id, session in sessions.items():
                if restore:
                    if session.updated:
                        backup_path_files(session.suite2p_path, restore=True)
                        session.updated = False
                        session.change_yaml_file("updated", False)
                        (
                            session.c,
                            session.contours,
                            session.footprints,
                        ) = session.get_c_contours_footprints()

            # TODO: filter for usefull sessions?
            # get units with enough usefull cells (at least 1/3 of best MUnit cells)
            # min_num_usefull_cells = best_unit.num_not_geldrying() / 3
            # units = self.get_usefull_units(min_num_usefull_cells, unit_type=unit_type)

            # create a master mask by
            # merging masks of every session, remove abroad cells and deduplicate
            print("Creating master mask...")
            merged_stat = merger.merge_stat(sessions, reference_session, parallel=True)
            print(f"Number of cells after merging: {merged_stat.shape[0]}")

            # Update all sessions based on merged mask
            updated_sessions = {}
            for session_id, session in sessions.items():
                # shift merged mask and redo Suite2P analysis
                if session_id in updated_sessions or session.session_type == "merged":
                    continue
                print(f"Updating session {session_id}")
                # shift and rotate merged mask to correct cell locations
                # and update suite2p files in session
                merger.shift_update_session_s2p_files(session, merged_stat)
                session.change_yaml_file("updated", True)
                updated_sessions[session_id] = Session(
                    session.yaml_file_path,
                    animal_id=session.animal_id,
                    print_loading=True,
                )
                updated_sessions[session_id].pday = session.pday
                updated_sessions[session_id].load_data(restore=False, regenerate=True)

            merger.merge_s2p_files(updated_sessions, merged_stat)
            self.sessions = updated_sessions
            merged_yaml_path = merger.merge_yaml_metadata(
                updated_sessions, reference_session
            )

        # create Session entrie in dictionary
        self.get_session_data(
            merged_yaml_path, regenerate=regenerate, print_loading=True
        )
        # get_geldrying_cells()
        return self.sessions["merged"]


class Session:
    corr_fname = "allcell_clean_corr_pval_zscore.npy"
    cabincorr_fname = "binarized_traces.npz"

    def __init__(self, yaml_file_path, animal_id, print_loading=True):
        # Animal.root_dir = yaml_file_path.split("DON-")[0]
        self.animal_id = animal_id
        self.yaml_file_path = yaml_file_path
        self.image_x_size = 512
        self.image_y_size = 512
        self.session_id = None  # = session_name
        self.date = None
        self.method = None
        self.pday = None
        self.yx_shift = None
        self.rot_center_yx = None
        self.rot_angle = None
        self.session_type = None
        self.water_deprivation = None
        self.session_dir = None
        self.suite2p_path = None
        self.binary_path = None
        self.refImg = None
        self.refAndMasks = None
        self.bin_traces_zip = None
        self.ops = None
        self.c, self.contours, self.footprints = None, None, None
        print(yaml_file_path)
        self.load_metadata(yaml_file_path)
        self.updated = None
        updated_txt = "updated " if self.updated else ""
        if print_loading:
            print(f"Initialized {updated_txt}session: {animal_id} {self.session_id}")

    def load_metadata(self, yaml_path):
        """
        Load session metadata from a YAML file and update the session object's attributes.

        Parameters:
        - yaml_path (str): Path to the YAML file containing session metadata.

        Raises:
        - NameError: If any of the required metadata variables are not defined in the YAML file.

        This function loads session metadata from a YAML file and assigns the values to the session object's attributes.
        It also performs some conditional checks and updates specific attributes based on the loaded metadata.

        """
        with open(yaml_path, "r") as yaml_file:
            session_metadata_dict = yaml.safe_load(yaml_file)

        # Load any additional metadata into session object
        for variable_name, value in session_metadata_dict.items():
            setattr(self, variable_name, value)

        if self.session_type:
            if self.session_type == "pretraining" or self.session_type == "merged":
                self.yx_shift = [0, 0]
                self.rot_angle = 0
                self.rot_center_yx = [0, 0]
                self.session_id = (
                    self.session_type if self.session_type != "pretraining" else "day0"
                )
                self.date = "99999999" if self.session_type == "merged" else self.date
        if not self.session_id:
            self.session_id = str(self.date)

        needed_variables = ["date", "yx_shift", "rot_center_yx", "rot_angle"]
        for needed_variable in needed_variables:
            defined_variable = getattr(self, needed_variable)
            if defined_variable == None:
                raise NameError(
                    f"Variable {needed_variable} is not defined in yaml file {yaml_path}"
                )

    def update_paths(self):
        """
        Update file paths within the session object based on session metadata.

        This function updates file paths within the session object based on session metadata and default values.
        It constructs paths for the session directory, Suite2P output, and binary data, as well as checks for old backup files.

        """
        self.session_dir = (
            os.path.join(Animal.root_dir, self.animal_id, str(self.session_id))
            if not self.session_dir
            else self.session_dir
        )
        self.suite2p_path = (
            os.path.join(self.session_dir, "plane0")
            if not self.suite2p_path
            else self.suite2p_path
        )
        self.binary_path = (
            find_binary_fpath(self.session_dir)
            if not self.binary_path
            else self.binary_path
        )
        self.updated = (
            self.old_backup_files(self.suite2p_path)
            if self.updated == None
            else self.updated
        )

    def old_backup_files(self, path):
        """
        Check if there are old backup files in the specified path.

        Parameters:
        - path (str): Path to the directory to check for old backup files.

        Returns:
        - bool: True if there are old backup files, otherwise False.

        This function checks if there are old backup files in the specified directory path and returns a boolean indicating their presence.

        """
        old_backup = False
        backup_path = os.path.join(path, "backup")
        if os.path.exists(backup_path):
            suite2p_folder_files_size = 0
            backup_files_size = 0
            for file_path in os.listdir(backup_path):
                if os.path.isfile(file_path):
                    backup_files_size += os.path.getsize(file_path)
            for file_path in os.listdir(path):
                if os.path.isfile(file_path):
                    suite2p_folder_files_size += os.path.getsize(file_path)
            if backup_files_size != suite2p_folder_files_size:
                old_backup = True
        return old_backup

    def load_data(self, restore=True, generate=False, regenerate=False, delete=False):
        self.update_paths()
        self.ops = self.set_ops()
        if restore and self.updated:
            backup_path = os.path.join(self.suite2p_path, "backup")
            backup_path_files(self.suite2p_path, restore=restore)
            if os.path.exists(backup_path):
                self.change_yaml_file("updated", False)
        self.c, self.contours, self.footprints = self.get_c_contours_footprints(
            regenerate=regenerate
        )

    def set_ops(self, ops=None):
        """
        Set Suite2P processing options (ops) for the session.

        Parameters:
        - ops (dict, optional): Suite2P processing options. If not provided, default options are used.

        Returns:
        - dict: The Suite2P processing options.

        This function sets Suite2P processing options for the session. If 'ops' is not provided, it uses default options. It also enforces 'nonrigid' to be False in the processing options.

        """
        if not ops:
            if not self.ops:
                ops_path = os.path.join(self.suite2p_path, "ops.npy")
                if os.path.exists(ops_path):
                    ops = np.load(ops_path, allow_pickle=True).item()
                if ops == None:
                    ops = register.default_ops()
                self.ops = ops
        else:
            self.ops = ops
        self.ops["nonrigid"] = False
        return self.ops

    def get_reference_image(self, n_frames_to_be_acquired=1000):
        """
        This function gets the reference image. If the reference image is not already set,
        it loads a binary file and computes the reference image.

        Parameters:
        n_frames_to_be_acquired (int): The number of frames to be acquired. Default is 1000.

        Returns:
        numpy.ndarray: The reference image.
        """
        if not self.refImg:
            b_loader = Binary_loader()
            frames = b_loader.load_binary(
                self.binary_path,
                n_frames_to_be_acquired=n_frames_to_be_acquired,
                image_x_size=self.image_x_size,
                image_y_size=self.image_y_size,
            )
            self.refImg = register.compute_reference(frames, ops=self.ops)
        return self.refImg

    def get_refAndMasks(self, n_frames_to_be_acquired=1000):
        """
        Retrieve or compute reference image and masks for the session.

        Parameters:
        - n_frames_to_be_acquired (int, optional): Number of frames to be acquired for reference image calculation (default is 1000).

        Returns:
        - tuple: A tuple containing the reference image and associated masks.

        This function either retrieves pre-computed reference image and masks or computes them if not available. The reference image is essential for various processing tasks in the session.

        """
        if not self.refAndMasks:
            refImg = self.get_reference_image(
                n_frames_to_be_acquired=n_frames_to_be_acquired
            )
            ops = self.set_ops()
            refAndMasks = register.compute_reference_masks(refImg, ops)
            self.refAndMasks = refAndMasks
        return self.refAndMasks

    def set_yx_shift(
        self,
        reference_session=None,
        num_align_frames=1000,
        yx_shift=None,
        n_frames_to_be_acquired=1000,
    ):
        """
        This function sets the yx_shift attribute. If yx_shift is not provided or not already set,
        it loads a binary file and registers the frames to compute yx_shift.

        Parameters:
        refAndMasks (numpy.ndarray): The reference and masks used for registering frames.
        num_align_frames (int): The number of frames to align. Default is 1000.
        yx_shift (list): The shift in y and x directions. Default is None.

        Returns:
        list: The computed or provided yx_shift.
        """
        if yx_shift and not self.yx_shift:
            self.yx_shift = yx_shift
        if not self.yx_shift:
            if reference_session:
                if reference_session.session_id == self.session_id:
                    self.yx_shift = [0, 0]
                else:
                    refAndMasks = reference_session.get_refAndMasks(
                        n_frames_to_be_acquired=n_frames_to_be_acquired
                    )
                    b_loader = Binary_loader()
                    frames = b_loader.load_binary(
                        self.binary_path,
                        n_frames_to_be_acquired=num_align_frames,
                        image_x_size=self.image_x_size,
                        image_y_size=self.image_y_size,
                    )
                    (
                        frames,
                        ymax,
                        xmax,
                        cmax,
                        ymax1,
                        xmax1,
                        cmax1,
                        _,
                    ) = register.register_frames(refAndMasks, frames, ops=self.ops)
                    self.yx_shift = [round(np.mean(ymax)), round(np.mean(xmax))]
            else:
                print(f"A reference sessions need to be provided to compute yx_shift")
        return self.yx_shift

    def get_c_contours_footprints(self, regenerate=False):
        """
        Retrieve or regenerate cell contours and footprints for the session using CaBincorr.

        Parameters:
        - regenerate (bool, optional): Whether to regenerate the data (default is False).

        Returns:
        - tuple: A tuple containing 'c' (CaBincorr data), 'contours' (cell contours), and 'footprints' (cell footprints).

        This function retrieves or regenerates cell contours and footprints for the session using CaBincorr. The 'regenerate' option allows you to force data regeneration.

        """
        # Merging cell footprints
        c = run_cabin_corr(
            Animal.root_dir,
            data_dir=self.suite2p_path,
            animal_id=self.animal_id,
            session_id=self.session_id,
            regenerate=regenerate,
        )

        contours = c.contours
        footprints = c.footprints
        return c, contours, footprints

    def load_cabincorr_data(self):
        """
        Load CaBincorr data if not already loaded.

        Returns:
        - np.ndarray: CaBincorr data in the form of a NumPy array.

        This function loads CaBincorr data in the form of a NumPy array if it has not been loaded before. The data is stored in 'bin_traces_zip' in the session object.

        """
        if type(self.bin_traces_zip) != np.ndarray:
            path = os.path.join(self.suite2p_path, Session.cabincorr_fname)
            if os.path.exists(path):
                self.bin_traces_zip = np.load(path, allow_pickle=True)
            else:
                print("No CaBincorrPath found")
        return self.bin_traces_zip

    def update_s2p_files(self, stat):
        # Read in existing data from a suite2p run. We will use the "ops" and registered binary.
        suite2_data_path = self.suite2p_path
        binary_file_path = self.binary_path

        ops = np.load(
            os.path.join(suite2_data_path, "ops.npy"), allow_pickle=True
        ).item()
        Lx = ops["Lx"]
        Ly = ops["Ly"]
        f_reg = suite2p.io.BinaryFile(Ly, Lx, binary_file_path)

        """# Using these inputs, we will first mimic the stat array made by suite2p
        masks = cellpose_masks['masks']
        stat = []
        for u_ix, u in enumerate(np.unique(masks)[1:]):
            ypix,xpix = np.nonzero(masks==u)
            npix = len(ypix)
            stat.append({'ypix': ypix, 'xpix': xpix, 'npix': npix, 'lam': np.ones(npix, np.float32), 'med': [np.mean(ypix), np.mean(xpix)]})
        stat = np.array(stat)
        stat = roi_stats(stat, Ly, Lx)  # This function fills in remaining roi properties to make it compatible with the rest of the suite2p pipeline/GUI
        """
        # Feed these values into the wrapper functions
        (
            stat_after_extraction,
            F,
            Fneu,
            F_chan2,
            Fneu_chan2,
        ) = suite2p.extraction_wrapper(stat, f_reg, f_reg_chan2=None, ops=ops)
        # Do cell classification
        classfile = suite2p.classification.builtin_classfile
        iscell = suite2p.classify(stat=stat_after_extraction, classfile=classfile)
        # Apply preprocessing step for deconvolution
        dF = F.copy() - ops["neucoeff"] * Fneu
        dF = suite2p.extraction.preprocess(
            F=dF,
            baseline=ops["baseline"],
            win_baseline=ops["win_baseline"],
            sig_baseline=ops["sig_baseline"],
            fs=ops["fs"],
            prctile_baseline=ops["prctile_baseline"],
        )
        # Identify spikes
        spks = suite2p.extraction.oasis(
            F=dF, batch_size=ops["batch_size"], tau=ops["tau"], fs=ops["fs"]
        )

        # backing up original suite2p files first
        backup_path_files(suite2_data_path)

        old_files = ["binarized_traces.mat", "binarized_traces.npz", "Fall.mat"]
        old_folders = ["correlations", "figures"]
        for old_folder in old_folders:
            fpath = os.path.join(suite2_data_path, old_folder)
            del_file_dir(fpath)
        for old_file in old_files:
            fpath = os.path.join(suite2_data_path, old_file)
            del_file_dir(fpath)

        np.save(os.path.join(suite2_data_path, "F.npy"), F)
        np.save(os.path.join(suite2_data_path, "Fneu.npy"), Fneu)
        np.save(os.path.join(suite2_data_path, "iscell.npy"), iscell)
        np.save(os.path.join(suite2_data_path, "ops.npy"), ops)
        np.save(os.path.join(suite2_data_path, "spks.npy"), spks)
        np.save(os.path.join(suite2_data_path, "stat.npy"), stat)

    def change_yaml_file(self, key, value):
        """
        Modify a specific key-value pair in the session's YAML metadata file.

        Parameters:
        - key (str): The key to modify or add in the YAML metadata file.
        - value: The new value to assign to the specified key.

        This function updates or adds a key-value pair in the session's YAML metadata file. It opens the YAML file, makes the modification, and saves the changes back to the file.

        """
        yaml_path = self.yaml_file_path
        with open(yaml_path, "r") as yaml_file:
            animal_metadata_dict = yaml.safe_load(yaml_file)
        animal_metadata_dict[key] = value
        with open(yaml_path, "w") as file:
            yaml.dump(animal_metadata_dict, file)


class Vizualizer:
    def __init__(self, animals={}, save_dir=Animal.root_dir):
        self.animals = animals
        self.save_dir = os.path.join(save_dir, "figures")
        dir_exist_create(self.save_dir)
        # Collor pallet for plotting
        self.max_color_number = 301
        self.colors = mlp.colormaps["rainbow"](range(0, self.max_color_number))

    def session_footprints(self, session, rot90_times=1, figsize=(10, 10), cmap=None):
        # plot footprints of a session
        plt.figure(figsize=figsize)
        title = f"{session.animal_id}_{session.session_id}"
        footprints = session.footprints
        plt.title(f"{len(footprints)} footprints {title}")
        self.footprints(footprints, rot90_times=rot90_times, cmap=cmap)
        plt.savefig(os.path.join(self.save_dir, f"Footprints_{title}.png"), dpi=300)

    def footprints(self, footprints, rot90_times=1, cmap=None):
        # plot all footprints
        for footprint in footprints:
            idx = np.where(footprint == 0)
            footprint[idx] = np.nan
            plt.imshow(np.rot90(footprint, k=rot90_times), cmap=cmap)

    def session_contours(
        self, session, figsize=(10, 10), color=None, plot_center=False, comment=""
    ):
        # Plot Contours
        plt.figure(figsize=figsize)
        title = f"{session.animal_id}_{session.session_id}"
        contours = session.contours
        self.contours(contours, color, plot_center, comment)
        plt.title(f"{len(contours)} contours {title}")
        plt.savefig(os.path.join(self.save_dir, f"Contours_{title}.png"), dpi=300)

    def contour_to_point(self, contour):
        x_mean = np.mean(contour[:, 0])
        y_mean = np.mean(contour[:, 1])
        return np.array([x_mean, y_mean])

    def contours(
        self, contours, color=None, alpha=1, plot_center=False, comment=""
    ):  # plot_contours_points
        for contour in contours:
            y_corr = contour[:, 0]
            x_corr = contour[:, 1]
            plt.plot(x_corr, y_corr, color=color, alpha=alpha)
            if plot_center:
                xy_mean = self.contour_to_point(contour)
                plt.plot(xy_mean[1], xy_mean[0], ".", color=color, alpha=alpha)
        plt.title(f"{len(contours)} Contours{comment}")

    def multi_contours(
        self,
        multi_contours,
        plot_center=False,
        colors=[
            "white",
            "red",
            "green",
            "blue",
            "yellow",
            "purple",
            "orange",
            "cyan",
            "deeppink",
        ],
        transparent=False,
    ):
        alpha = 0.5 if transparent else 1
        for contours, col in zip(multi_contours, colors):
            self.contours(contours, color=col, alpha=alpha, plot_center=plot_center)

    def multi_session_contours(
        self,
        sessions,
        combination=None,
        colors=[
            "white",
            "red",
            "green",
            "blue",
            "yellow",
            "purple",
            "orange",
            "cyan",
            "deeppink",
        ],
        plot_center=False,
        transparent=False,
        shift=False,
        yx_shift=None,
        rot_angle=None,
        rot_center_yx=None,
        figsize=(20, 20),
        comment="",
    ):
        """
        sessions : dict
        combination : list of dict keys
        """
        plt.figure(figsize=figsize)
        if shift != False:
            shift_type = shift
            shift = True
        handles = []
        plot_contours = []
        plot_colors = []
        combination = list(sessions.keys()) if combination == None else combination

        merger = Merger()
        for (session_id, session), col in zip(sessions.items(), colors):
            if session_id not in combination:
                continue
            contours = session.contours
            shift_label = ""

            # shift, rotate contours
            all_shifted_rotated_contour_points = []
            if shift and session_id != "day0" and session.session_type != "merged":
                session_yx_shift = session.yx_shift if yx_shift == None else yx_shift
                session_rot_angle = (
                    session.rot_angle if rot_angle == None else rot_angle
                )
                session_rot_center_yx = (
                    session.rot_center_yx if rot_center_yx == None else rot_center_yx
                )
                for points_yx in contours:
                    shifted_rotated_contour_points = merger.shift_rotate_yx_points(
                        points_yx,
                        yx_shift=session_yx_shift,
                        rot_angle=session_rot_angle,
                        rot_center_yx=session_rot_center_yx,
                    )
                    all_shifted_rotated_contour_points.append(
                        shifted_rotated_contour_points
                    )
                contours = all_shifted_rotated_contour_points
                shift_label = f" yx_shift: {session_yx_shift}  yx_center: {session_rot_center_yx}  angle: {session_rot_angle}"

            plot_colors.append(col)
            plot_contours.append(contours)
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=col,
                    marker="o",
                    label=f"Session {session_id}: {shift_label}",
                )
            )

        self.multi_contours(
            plot_contours,
            colors=plot_colors,
            plot_center=plot_center,
            transparent=transparent,
        )
        plt.title(f"Contours for : {combination} {comment}")
        plt.legend(handles=handles, fontsize=20)
        shift_label = f"_shifted" if shift else ""
        plt.savefig(
            os.path.join(
                self.save_dir, f"Contours_{combination}{shift_label}{comment}.png"
            ),
            dpi=300,
        )
        plt.show()
        plt.close()

    def multi_session_refImg(self, sessions, num_images_x=2):
        num_sessions = len(sessions)
        num_rows = round(num_sessions / num_images_x)
        fig, ax = plt.subplots(
            num_rows, num_images_x, figsize=(5 * num_images_x, 5 * num_rows)
        )
        for i, (session_id, session) in enumerate(sessions.items()):
            title = f"Reference Images of {session.animal_id}"
            x = int(i / num_images_x)
            y = i % num_images_x
            if len(ax.shape) == 2:
                ax[x, y].imshow(session.refImg)
                ax[x, y].invert_yaxis()
                ax[x, y].set_title(f"{session_id}")
            else:
                ax[i].imshow(session.refImg)
                ax[i].invert_yaxis()
                ax[i].set_title(f"{session_id}")
        fig.suptitle(title, fontsize=20)
        plt.savefig(
            os.path.join(self.save_dir, title.replace(" ", "_") + ".png"), dpi=300
        )
        plt.show()

    def bursts(
        self,
        animal_id,
        session_id,
        fluorescence_type="F_raw",
        num_cells="all",
        dpi=300,
        fps="30",
    ):
        # TODO: good cells filter can be added from complete animal class

        session = self.animals[animal_id].sessions[session_id]
        bin_traces_zip = session.load_cabincorr_data()
        fluorescence = None
        if bin_traces_zip:
            if fluorescence_type in list(bin_traces_zip.keys()):
                fluorescence = bin_traces_zip[fluorescence_type]
            else:
                print(
                    f"{animal_id} {session_id} No fluorescence data of type {fluorescence_type} in binarized_traces.npz"
                )
        else:
            print(f"{animal_id} {session_id} no binarized_traces.npz found")

        if type(fluorescence) == np.ndarray:
            self.traces(
                fluorescence,
                animal_id,
                session_id,
                num_cells,
                fluorescence_type=fluorescence_type,
                dpi=dpi,
            )
        return fluorescence

    def traces(
        self,
        fluorescence,
        animal_id,
        session_id,
        num_cells="all",
        fluorescence_type="",
        low_pass_filter=True,
        dpi=300,
    ):
        # plot fluorescence
        if low_pass_filter:
            fluorescence = butter_lowpass_filter(
                fluorescence, cutoff=0.5, fs=30, order=2
            )

        fluorescence = np.array(fluorescence)
        fluorescence = (
            np.transpose(fluorescence) if len(fluorescence.shape) == 2 else fluorescence
        )
        plt.figure()
        plt.figure(figsize=(12, 7))
        if num_cells != "all":
            plt.plot(fluorescence[:, : int(num_cells)])
        else:
            plt.plot(fluorescence)

        file_name = f"{animal_id} {session_id}"
        seconds = 5
        fps = 30
        num_frames = fps * seconds
        num_x_ticks = 50
        written_label_steps = 2

        x_time = [
            int(frame / num_frames) * seconds
            for frame in range(len(fluorescence))
            if frame % num_frames == 0
        ]
        steps = round(len(x_time) / (2 * num_x_ticks))
        x_time_shortened = x_time[::steps]
        x_pos = np.arange(0, len(fluorescence), num_frames)[::steps]

        title = f"Bursts from {file_name} {fluorescence_type}"
        xlabel = f"seconds"
        ylabel = "fluorescence based on Ca in Cell"
        x_labels = [
            time if num % written_label_steps == 0 else ""
            for num, time in enumerate(x_time_shortened)
        ]
        plt.xticks(x_pos, x_labels, rotation=40, fontsize=8)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.savefig(
            os.path.join(self.save_dir, title.replace(" ", "_") + ".png"), dpi=dpi
        )
        plt.show()
        plt.close()


class Binary_loader:
    """
    A class for loading binary data and converting it into an animation.

    This class provides methods for loading binary data from a file and converting a sequence of binary frames into an animated GIF. The `load_binary`
    method loads binary data from a specified file and returns it as a NumPy array. The `binary_frames_to_gif` method takes a sequence of binary frames and converts them into an animated GIF, which is saved to the specified directory.

    Attributes:
        None
    """

    def load_binary(
        self,
        fpath,
        n_frames_to_be_acquired,
        fname="Image_001_001.raw",
        image_x_size=512,
        image_y_size=512,
    ):
        """
        Loads binary data from a file.

        This method takes the path of a binary data file as input, along with the number of frames to be acquired and the dimensions of each frame. It loads the binary data from the specified file and returns it as a NumPy array.

        Args:
            data_path (str): The path of the binary data file.
            n_frames_to_be_acquired (int): The number of frames to be acquired from the binary data file.
            fname (str): The name of the binary data file. Defaults to "data.bin".
            image_x_size (int): The width of each frame in pixels. Defaults to 512.
            image_y_size (int): The height of each frame in pixels. Defaults to 512.

        Returns:
            np.ndarray: A NumPy array containing the loaded binary data.
        """
        # load binary file from suite2p_folder from session
        binary = np.memmap(
            fpath,
            dtype="uint16",
            mode="r",
            shape=(n_frames_to_be_acquired, image_x_size, image_y_size),
        )
        binary_frames = copy.deepcopy(binary)
        return binary_frames

    def binary_frames_to_gif(
        self, frames, frame_range=[0, -1], fps=30, save_dir="animation", comment=""
    ):
        """
        Converts a sequence of binary frames into an animated GIF.

        This method takes a sequence of binary frames as input, along with the range of frames to include in the animation and the directory in which to save the resulting GIF. It converts the specified frames into an animated GIF and saves it to the specified directory.

        Args:
            frames (np.ndarray): A NumPy array containing the sequence of binary frames.
            frame_range (List[int]): A list specifying the range of frames to include in the animation. Defaults to [0, -1], which includes all frames.
            save_dir (str): The directory in which to save the resulting GIF. Defaults to "animation".

        Returns:
            animation.ArtistAnimation: An instance of `animation.ArtistAnimation` representing the created animation.
        """
        import matplotlib.animation as animation

        range_start, range_end = frame_range
        comment = comment + "_" if comment != "" else comment
        save_dir = os.path.join(save_dir, "animation")
        gif_save_path = os.path.join(
            save_dir, f"{comment}{range_start}-{range_end}.gif"
        )

        delay_between_frames = int(1000 / fps)  # ms
        images = []
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, frame in enumerate(frames):
            if i % 1000 == 0:
                print(i)
            p1 = ax.text(512 / 2 - 50, 0, f"Frame {i}", animated=True)
            p2 = ax.imshow(frame, animated=True)
            images.append([p1, p2])
        ani = animation.ArtistAnimation(
            fig, images, interval=delay_between_frames, blit=True, repeat_delay=1000
        )
        ani.save(gif_save_path)
        return ani


class Merger:
    def create_points_from_stat(self, cell_stat: np.ndarray):
        """
        Create an array of (y, x) points from a cell's statistical data.

        Parameters:
        - cell_stat (np.ndarray): Statistical data for a single cell.

        Returns:
        - np.ndarray: An array of (y, x) points representing the cell's position.

        This function extracts the (y, x) positions from the statistical data of a single cell and returns them as an array.
        """
        points_yx = np.array([cell_stat["ypix"], cell_stat["xpix"]]).transpose()
        return points_yx

    def rotate_points(self, points: [[int, int]], cx: float, cy: float, theta: float):
        """
        Rotate a set of points around a specified center.

        Parameters:
        - points (np.ndarray): An array of (y, x) points to be rotated.
        - cx (float): X-coordinate of the rotation center.
        - cy (float): Y-coordinate of the rotation center.
        - theta (float): Angle of rotation in degrees.

        Returns:
        - np.ndarray: An array of rotated (y, x) points.

        This function rotates a set of (y, x) points around a specified center (cx, cy) by a given angle (theta).

        """
        # Convert the angle to radians
        theta = np.radians(theta)

        # Create a rotation matrix
        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )

        # Create a matrix of points
        points_matrix = np.column_stack((points[:, 0] - cx, points[:, 1] - cy))

        # Apply the rotation matrix to the points
        rotated_points_matrix = np.dot(points_matrix, rotation_matrix.T)

        # Translate the points back to the original coordinate system
        rotated_points = rotated_points_matrix + np.array([cx, cy])
        rotated_points_int = rotated_points.astype(int)
        return rotated_points_int

    def shift_rotate_yx_points(
        self,
        points_yx,
        yx_shift: [int, int],
        rot_angle: float,
        rot_center_yx: [float, float] = None,
        roation_first=False,
    ):
        """
        Shift and rotate a set of (y, x) points.

        Parameters:
        - points_yx (np.ndarray): An array of (y, x) points to be shifted and rotated.
        - yx_shift (list): A list containing the (y, x) shift values.
        - rot_angle (float): Angle of rotation in degrees.
        - rot_center_yx (list, optional): The (y, x) coordinates for the rotation center.
        - rotation_first (bool, optional): Whether to apply rotation before shifting (default is False).

        Returns:
        - np.ndarray: An array of (y, x) points after shifting and rotating.

        This function shifts and rotates a set of (y, x) points. The order of shifting and rotating can be controlled by the 'rotation_first' parameter.

        """
        if roation_first:
            if rot_angle != 0:
                # rotate points at center if no rot_center_yx is provided
                rot_center_yx = (
                    np.mean(points_yx, axis=0) if not rot_center_yx else rot_center_yx
                )
                rot_y, rot_x = rot_center_yx

                # swapping rotation center and negating rotation angle to ensure correct rotation based on swapped x-, y-coordinates from suite2p
                rotated_contour_points = self.rotate_points(
                    points_yx, rot_y, rot_x, -rot_angle
                )
            else:
                rotated_contour_points = points_yx
            # shift
            shifted_rotated_contour_points = rotated_contour_points + np.array(yx_shift)
        else:
            # shift
            shifted_points_yx = points_yx + np.array(yx_shift)
            if rot_angle != 0:
                # rotate points at center if no rot_center_yx is provided
                rot_center_yx = (
                    np.mean(shifted_points_yx, axis=0)
                    if not rot_center_yx
                    else rot_center_yx
                )
                rot_y, rot_x = rot_center_yx

                # swapping rotation center and negating rotation angle to ensure correct rotation based on swapped x-, y-coordinates from suite2p
                shifted_rotated_contour_points = self.rotate_points(
                    shifted_points_yx, rot_y, rot_x, -rot_angle
                )
            else:
                shifted_rotated_contour_points = shifted_points_yx
        return shifted_rotated_contour_points

    def shift_rotate_contour_cloud(
        self,
        stat,
        yx_shift: [int, int],
        rot_angle: float,
        rot_center_yx: [float, float],
        roation_first=False,
    ):
        """
        Shift and rotate the cell contour cloud.

        Parameters:
        - stat (np.ndarray): Statistical data for multiple cells.
        - yx_shift (list): A list containing the (y, x) shift values.
        - rot_angle (float): Angle of rotation in degrees.
        - rot_center_yx (list): The (y, x) coordinates for the rotation center.
        - rotation_first (bool, optional): Whether to apply rotation before shifting (default is False).

        Returns:
        - tuple: A tuple containing shifted and rotated center points and shifted and rotated cell contour points.

        This function shifts and rotates the cell contour points of multiple cells based on the provided shift and rotation parameters.

        """
        # shift center of cells
        center_points_yx = np.array([cell_stat["med"] for cell_stat in stat])
        shifted_rotated_center_points_yx = self.shift_rotate_yx_points(
            center_points_yx,
            yx_shift=yx_shift,
            rot_angle=rot_angle,
            rot_center_yx=rot_center_yx,
            roation_first=roation_first,
        )
        # shift, rotate cell contour pixels
        all_shifted_rotated_contour_points = []

        # calculate corrected yxshift
        # code below can be used if roation is not affine
        # corrected_yxshifts = shifted_rotated_center_points_yx - center_points_yx
        # for num, (cell_stat, corrected_yxshift) in enumerate(zip(stat, corrected_yxshifts)):

        for num, cell_stat in enumerate(stat):
            points_yx = self.create_points_from_stat(cell_stat)
            # shift, rotate cell contour pixels
            shifted_rotated_contour_points = self.shift_rotate_yx_points(
                points_yx,
                yx_shift=yx_shift,
                rot_angle=rot_angle,
                rot_center_yx=rot_center_yx,
                roation_first=roation_first,
            )
            all_shifted_rotated_contour_points.append(shifted_rotated_contour_points)

        return shifted_rotated_center_points_yx, all_shifted_rotated_contour_points

    def shift_rotate_stat_cells(
        self,
        session: Session = None,
        stat: np.ndarray = None,
        yx_shift: [int, int] = None,
        rot_angle: float = None,
        rot_center_yx: [float, float] = None,
        roation_first=False,
    ):
        """
        Shift and rotate the statistical data of cells within a session.

        Parameters:
        - session (Session, optional): The session object containing session-specific parameters.
        - stat (np.ndarray, optional): Statistical data for multiple cells.
        - yx_shift (list, optional): A list containing the (y, x) shift values.
        - rot_angle (float, optional): Angle of rotation in degrees.
        - rot_center_yx (list, optional): The (y, x) coordinates for the rotation center.
        - rotation_first (bool, optional): Whether to apply rotation before shifting (default is False).

        Returns:
        - np.ndarray: Modified statistical data with shifted and rotated cell information.

        This function shifts and rotates the statistical data of cells within a session, taking into account various parameters, including shift, rotation, and rotation center.

        """
        # stat files first value ist y-value second is x-value
        stat = session.c.stat if type(stat) != np.ndarray else stat
        yx_shift = session.yx_shift if not yx_shift else yx_shift
        rot_angle = session.rot_angle if rot_angle == None else rot_angle
        rot_center_yx = session.rot_center_yx if not rot_center_yx else rot_center_yx
        new_stat = copy.deepcopy(stat)

        (
            shifted_rotated_center_points_yx,
            all_shifted_rotated_contour_points,
        ) = self.shift_rotate_contour_cloud(
            new_stat,
            yx_shift=yx_shift,
            rot_angle=rot_angle,
            rot_center_yx=rot_center_yx,
            roation_first=roation_first,
        )
        for cell_stat, center_point, shifted_rotated_contour_points in zip(
            new_stat,
            shifted_rotated_center_points_yx,
            all_shifted_rotated_contour_points,
        ):
            cell_stat["med"] = center_point
            # set new cell contour pixels
            cell_stat["ypix"] = shifted_rotated_contour_points[:, 0]
            cell_stat["xpix"] = shifted_rotated_contour_points[:, 1]
        return new_stat

    def remove_abroad_cells(
        self, stat: np.ndarray, sessions: dict, image_x_size=512, image_y_size=512
    ):
        """
        Removes cells that are out of bounds.

        Parameters:
        stat (list): List of cell statistics.
        sessions (dict): A dictionary of sessions keyed by session_id.
        image_x_size (int): Size of the image in x direction. Default is 512.
        image_y_size (int): Size of the image in y direction. Default is 512.

        Returns:
        stat (list): List of cell statistics after removing out of bound cells.
        """
        remove_cells = []
        # check for every shift and rotation combination if cell is in bounds
        for session_id, session in sessions.items():
            # negate shift and rotation angle to cover all possible cell positions
            yx_shift = list(-np.array(session.yx_shift))
            rot_angle = -session.rot_angle
            rot_center_yx = session.rot_center_yx
            if rot_angle == 0 and sum(abs(np.array(yx_shift))) == 0:
                continue
            _, all_shifted_rotated_contour_points = self.shift_rotate_contour_cloud(
                stat=stat,
                yx_shift=yx_shift,
                rot_angle=rot_angle,
                rot_center_yx=rot_center_yx,
                roation_first=True,
            )

            for cell_num, shifted_rotated_contour_points in enumerate(
                all_shifted_rotated_contour_points
            ):
                for point in shifted_rotated_contour_points:
                    # check if point is out of bounds
                    if (
                        point[0] >= image_y_size
                        or point[0] < 0
                        or point[1] >= image_x_size
                        or point[1] < 0
                    ):
                        remove_cells.append(cell_num)
                        break

        # removing out of bound cells
        unique_remove_cells = np.unique(remove_cells)
        unique_remove_cells.sort()
        for abroad_cell in unique_remove_cells[::-1]:
            stat = np.delete(stat, abroad_cell)
        if len(unique_remove_cells) > 0:
            print(f"removed abroad cells: {unique_remove_cells}")
        return stat

    def stat_to_footprints(self, stat: np.ndarray, dims=[512, 512]):
        """
        Converts cell statistics to footprints.

        Parameters:
        stat (list): List of cell statistics.
        dims (list): List containing dimensions of the footprints. Default is [512, 512].

        Returns:
        footprints (numpy array): Array of footprints.
        """
        imgs = []
        for k in range(len(stat)):
            x = stat[k]["xpix"]
            y = stat[k]["ypix"]

            # save footprint
            img_temp = np.zeros((dims[0], dims[1]))
            img_temp[x, y] = stat[k]["lam"]

            img_temp_norm = (img_temp - np.min(img_temp)) / (
                np.max(img_temp) - np.min(img_temp)
            )
            imgs.append(img_temp_norm)

        imgs = np.array(imgs)

        footprints = imgs
        return footprints

    def merge_stat(self, sessions: dict, reference_session: Session, parallel=True):
        """
        Shifts and merges stat files with reference_session as reference position.
        It also deduplicates the stat files.

        Parameters:
        sessions (dict): A dictionary of sessions keyed by session_id.
        reference_session (Session): The session to be used as reference.
        parallel (bool): If True, use parallel processing. Default is True.

        Returns:
        merged_stat (numpy array): Array of merged and deduplicated stat files.
        """
        image_x_size = reference_session.image_x_size
        image_y_size = reference_session.image_y_size
        num_batches = get_num_batches_based_on_available_ram()

        moved_session_stat_no_abroad = self.remove_abroad_cells(
            reference_session.c.stat,
            sessions,
            image_x_size=image_x_size,
            image_y_size=image_y_size,
        )
        merged_footprints = self.stat_to_footprints(moved_session_stat_no_abroad)
        merged_stat = moved_session_stat_no_abroad
        for session_id, session in sessions.items():
            if (
                session_id == reference_session.session_id
                or session.session_type == "merged"
            ):
                continue
            print(f"Working on session {session_id}...")
            moved_session_stat = self.shift_rotate_stat_cells(session)
            moved_session_stat_no_abroad = self.remove_abroad_cells(
                moved_session_stat,
                sessions,
                image_x_size=image_x_size,
                image_y_size=image_y_size,
            )
            moved_footprints = self.stat_to_footprints(moved_session_stat_no_abroad)
            clean_cell_ids, merged_footprints = self.merge_deduplicate_footprints(
                merged_footprints,
                moved_footprints,
                parallel=parallel,
                num_batches=num_batches,
            )
            merged_stat = np.concatenate([merged_stat, moved_session_stat_no_abroad])[
                clean_cell_ids
            ]
        return merged_stat

    def find_overlaps1(self, ids, footprints):
        """
        Finds overlaps between footprints.

        Parameters:
        ids : Array of IDs.
        footprints : Array of footprints.

        Returns:
        intersections (list): List of intersections between footprints.
        """
        intersections = []
        for k in ids:
            temp1 = footprints[k]
            idx1 = np.vstack(np.where(temp1 > 0)).T

            #
            for p in range(k + 1, footprints.shape[0], 1):
                temp2 = footprints[p]
                idx2 = np.vstack(np.where(temp2 > 0)).T
                res = array_row_intersection(idx1, idx2)

                #
                if len(res) > 0:
                    percent1 = res.shape[0] / idx1.shape[0]
                    percent2 = res.shape[0] / idx2.shape[0]
                    intersections.append([k, p, res.shape[0], percent1, percent2])
        #
        return intersections

    def generate_batch_cell_overlaps(
        self,
        footprints,
        parallel=True,
        recompute_overlap=False,
        n_cores=16,
        num_batches=3,
    ):
        # this computes spatial overlaps between cells; doesn't take into account temporal correlations
        """
        Computes spatial overlaps between cells. It doesn't take into account temporal correlations.

        Parameters:
        footprints : Array of footprints.
        parallel (bool): If True, use parallel processing. Default is True.
        recompute_overlap (bool): If True, recompute overlap. Default is False.
        n_cores (int): Number of cores to use for parallel processing. Default is 16.
        num_batches (int): Number of batches for parallel processing. Default is 3.

        Returns:
        df (DataFrame): DataFrame containing overlap information.
        """
        print("... computing cell overlaps ...")

        num_footprints = footprints.shape[0]
        num_min_cells_per_process = 10
        num_parallel_processes = (
            30
            if num_footprints / 30 > num_min_cells_per_process
            else int(num_footprints / num_min_cells_per_process)
        )
        ids = np.array_split(
            np.arange(num_footprints, dtype="int64"), num_parallel_processes
        )

        if num_batches > num_parallel_processes:
            num_batches = num_parallel_processes

        # TODO: will results in an error, if np.array_split is used on inhomogeneouse data like ids on Scicore
        batches = np.array_split(ids, num_batches) if num_batches != 1 else [ids]
        results = np.array([])
        num_cells = 0
        for batch in batches:
            res = parmap.map(
                find_overlaps1,
                batch,
                footprints,
                # c.footprints_bin,
                pm_processes=16,
                pm_pbar=True,
                pm_parallel=parallel,
            )
            for cell_batch in res:
                num_cells += len(cell_batch)
                for cell in cell_batch:
                    results = np.append(results, cell)
        results = results.reshape(num_cells, 5)
        res = [results]
        df = make_overlap_database(res)
        return df

    def find_candidate_neurons_overlaps(
        self,
        df_overlaps: pd.DataFrame,
        corr_array=None,
        deduplication_use_correlations=False,
        corr_max_percent_overlap=0.25,
        corr_threshold=0.3,
    ):
        """
        This function finds candidate neurons based on overlaps and correlations.

        Parameters:
        df_overlaps (DataFrame): DataFrame containing overlap information.
        corr_array (numpy array): Array containing correlation information. Default is None.
        deduplication_use_correlations (bool): If True, use correlations for deduplication. Default is False.
        corr_max_percent_overlap (float): Maximum percent overlap for correlation. Default is 0.25.
        corr_threshold (float): Threshold for correlation. Default is 0.3.

        Returns:
        candidate_neurons (numpy array): Array of candidate neurons based on overlaps and correlations.
        """
        dist_corr_matrix = []
        for index, row in df_overlaps.iterrows():
            cell1 = int(row["cell1"])
            cell2 = int(row["cell2"])
            percent1 = row["percent_cell1"]
            percent2 = row["percent_cell2"]

            if deduplication_use_correlations:
                if cell1 < cell2:
                    corr = corr_array[cell1, cell2, 0]
                else:
                    corr = corr_array[cell2, cell1, 0]
            else:
                corr = 0
            dist_corr_matrix.append([cell1, cell2, corr, max(percent1, percent2)])
        dist_corr_matrix = np.vstack(dist_corr_matrix)
        #####################################################
        # check max overlap
        idx1 = np.where(dist_corr_matrix[:, 3] >= corr_max_percent_overlap)[0]

        # skipping correlations is not a good idea
        #   but is a requirement for computing deduplications when correlations data cannot be computed first
        if deduplication_use_correlations:
            idx2 = np.where(dist_corr_matrix[idx1, 2] >= corr_threshold)[
                0
            ]  # note these are zscore thresholds for zscore method
            idx3 = idx1[idx2]
        else:
            idx3 = idx1
        #
        candidate_neurons = dist_corr_matrix[idx3][:, :2]
        return candidate_neurons

    def make_correlated_neuron_graph(
        self, num_cells: int, candidate_neurons: np.ndarray
    ):
        """
        This function creates a graph of correlated neurons.

        Parameters:
        num_cells (int): Number of cells.
        candidate_neurons (numpy array): Array of candidate neurons.

        Returns:
        G (networkx.Graph): Graph of correlated neurons.
        """
        adjacency = np.zeros((num_cells, num_cells))
        for i in candidate_neurons:
            adjacency[int(i[0]), int(i[1])] = 1

        G = nx.Graph(adjacency)
        G.remove_nodes_from(list(nx.isolates(G)))
        return G

    def delete_duplicate_cells(
        self, num_cells: int, G, corr_delete_method="highest_connected_no_corr"
    ):
        """
        This function deletes duplicate cells from the graph.

        Parameters:
        num_cells (int): Number of cells.
        G (networkx.Graph): Graph of correlated neurons.
        corr_delete_method (str): Method to delete duplicate cells. Default is 'highest_connected_no_corr'.

        Returns:
        clean_cell_ids (numpy array): Array of clean cell IDs after deleting duplicates.
        """
        # delete multi node networks
        #
        if corr_delete_method == "highest_connected_no_corr":
            connected_cells, removed_cells = del_highest_connected_nodes_without_corr(G)
        #
        print("Removed duplicated cells: ", len(removed_cells))
        clean_cells = np.delete(np.arange(num_cells), removed_cells)

        #
        clean_cell_ids = clean_cells
        removed_cell_ids = removed_cells
        connected_cell_ids = connected_cells
        return clean_cell_ids

    def merge_deduplicate_footprints(
        self,
        footprints1: np.ndarray,
        footprints2: np.ndarray,
        parallel=True,
        num_batches=4,
    ):
        """
        This function merges and deduplicates footprints.

        Parameters:
        footprints1, footprints2 (numpy arrays): Arrays of footprints to be merged and deduplicated.
        parallel (bool): If True, use parallel processing. Default is True.
        num_batches (int): Number of batches for parallel processing. Default is 4.

        Returns:
        clean_cell_ids (numpy array): Array of clean cell IDs after merging and deduplicating footprints.
        cleaned_merged_footprints (numpy array): Array of cleaned merged footprints.
        """
        merged_footprints = np.concatenate([footprints1, footprints2])
        num_cells = len(merged_footprints)

        df_overlaps = self.generate_batch_cell_overlaps(
            merged_footprints,
            recompute_overlap=True,
            parallel=parallel,
            num_batches=num_batches,
        )
        candidate_neurons = self.find_candidate_neurons_overlaps(
            df_overlaps,
            corr_array=None,
            deduplication_use_correlations=False,
            corr_max_percent_overlap=0.25,
            corr_threshold=0.3,
        )
        G = self.make_correlated_neuron_graph(num_cells, candidate_neurons)
        clean_cell_ids = self.delete_duplicate_cells(num_cells, G)
        cleaned_merged_footprints = merged_footprints[clean_cell_ids]
        return clean_cell_ids, cleaned_merged_footprints

    def shift_update_session_s2p_files(self, session: Session, new_stat: np.ndarray):
        """
        This function shifts and updates session files.

        Parameters:
        session (object): Session object containing session information.
        new_stat (numpy array): Array containing new statistics.

        Returns:
        None
        """
        suite2p_data_path = session.suite2p_path
        # shift merged mask
        shift_to_session = list(-np.array(session.yx_shift))
        rotate_to_angle = -session.rot_angle
        rot_center_yx = session.rot_center_yx
        shifted_rotated_session_stat = self.shift_rotate_stat_cells(
            stat=new_stat,
            yx_shift=shift_to_session,
            rot_angle=rotate_to_angle,
            rot_center_yx=rot_center_yx,
            roation_first=True,
        )
        backup_path_files(suite2p_data_path)
        session.update_s2p_files(shifted_rotated_session_stat)

    def merge_s2p_files(self, sessions, stat, first_session="day0"):
        """
        Merges F, Fneu, spks, iscell from individual sessions
        Does not merge the individual corrected stat files
        Does not merge ops
        """
        first_session_object = sessions[first_session]
        ops = first_session_object.ops
        path = first_session_object.suite2p_path
        merged_F = np.load(os.path.join(path, "F.npy"))
        merged_Fneu = np.load(os.path.join(path, "Fneu.npy"))
        merged_spks = np.load(os.path.join(path, "spks.npy"))
        merged_iscell = np.load(os.path.join(path, "iscell.npy"))
        for session_id, session in sessions.items():
            if session_id == first_session_object.session_id:
                continue
            path = session.suite2p_path
            F = np.load(os.path.join(path, "F.npy"))
            merged_F = np.concatenate([merged_F, F], axis=1)
            Fneu = np.load(os.path.join(path, "Fneu.npy"))
            merged_Fneu = np.concatenate([merged_Fneu, Fneu], axis=1)
            spks = np.load(os.path.join(path, "spks.npy"))
            merged_spks = np.concatenate([merged_spks, spks], axis=1)
            # sum iscells
            is_cell = np.load(os.path.join(path, "iscell.npy"))
            merged_iscell += is_cell

        # let cells life if one of the cells is detected as cell. Average probabilities for ifcell
        merged_iscell /= len(list(sessions.keys()))
        merged_iscell[:, 0] = np.ceil(merged_iscell[:, 0])

        animal_folder = os.path.join(Animal.root_dir, session.animal_id)
        merged_s2p_path = create_dirs([animal_folder, "merged", "plane0"])

        np.save(os.path.join(merged_s2p_path, "F.npy"), merged_F)
        np.save(os.path.join(merged_s2p_path, "Fneu.npy"), merged_Fneu)
        np.save(os.path.join(merged_s2p_path, "spks.npy"), merged_spks)
        np.save(os.path.join(merged_s2p_path, "iscell.npy"), merged_iscell)
        np.save(os.path.join(merged_s2p_path, "stat.npy"), stat)
        np.save(os.path.join(merged_s2p_path, "ops.npy"), ops)
        return merged_s2p_path

    def merge_yaml_metadata(self, sessions, reference_session):
        """
        Merge and save session metadata into a new YAML file for a merged session.

        Parameters:
        - sessions (dict): A dictionary containing session objects with their respective metadata.
        - reference_session (Session): The reference session to extract essential metadata.

        Returns:
        - str: The path to the merged YAML metadata file.

        This function combines metadata from multiple sessions, including the reference session, and saves it to a new YAML file for a merged session. It includes key information such as 'animal_id,' image dimensions, 'merged' status, and a list of session names included in the merge.

        """
        yaml_fname = f"merged.yaml"
        yaml_path = os.path.join(
            Animal.root_dir, reference_session.animal_id, "merged", yaml_fname
        )

        merged_metadata = {
            "image_x_size": reference_session.image_x_size,
            "image_y_size": reference_session.image_y_size,
            "session_type": "merged",
            "sessions": list(sessions.keys()),
        }

        # save metadata to yaml
        with open(yaml_path, "w") as file:
            yaml.dump(merged_metadata, file)
        return yaml_path
