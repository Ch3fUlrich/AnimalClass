# imports
# Statistics
import seaborn as sns
import pandas as pd
import numpy as np

# Plotting
import matplotlib as mlp
import matplotlib.pyplot as plt, mpld3  # plotting and html plots

plt.style.use("dark_background")
# plt.style.use('default')
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle

# Regular Expression searching
import re

# Suite2p for TIFF file analysis
import suite2p
from suite2p.run_s2p import run_s2p, default_ops
from suite2p.registration import register

# Used for Popups
import tkinter as tk

import nest_asyncio

import parmap

# for progress bar support
from tqdm import tqdm

# interact with system
import os
import sys
import copy
import shutil
import psutil

# statistics
import scipy
import math

# Mesc file analysis
import h5py
from tifffile import tifffile, imread
import pathlib

# run matlab code
# octave needs to be installed in your PATH environment variable https://octave.org/download
# from oct2py import octave

# add root directory to be able to import packages
# todo: make all packages installable so they can be called/imported by environment
# module_path = os.path.abspath(os.path.join('../'))
# print(module_path)
# sys.path.append(module_path)
from Helper import *
from manifolds.donlabtools.utils.calcium import calcium
from manifolds.donlabtools.utils.calcium.calcium import *

import logging


class GlobalLogger:
    def __init__(self, save_dir=""):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.configure_logger(save_dir=save_dir)

    def configure_logger(self, save_dir=""):
        self.logger.setLevel(logging.DEBUG)  # Set the desired level here

        # Create a file handler which logs even debug messages.
        log_file_path = os.path.join(save_dir, "Global_log.log")
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(logging.DEBUG)

        # Create a console handler with a higher log level.
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)

        # Create a formatter and add it to the handlers.
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add the handlers to the logger.
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def set_save_dir(self, save_dir):
        # Get the old handler
        old_handler = self.logger.handlers[0]

        # Create a new handler with the updated filename
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        log_save_location = os.path.join(save_dir, "Global_log.log")
        new_handler = logging.FileHandler(log_save_location)
        new_handler.setLevel(old_handler.level)
        new_handler.setFormatter(old_handler.formatter)

        # Remove the old handler and add the new one
        self.logger.removeHandler(old_handler)
        self.logger.addHandler(new_handler)


global_logger_object = GlobalLogger()
global_logger = global_logger_object.logger

# global_logger_object.set_save_dir(os.path.join(os.getcwd(), "logs"))
# global_logger = global_logger_object.logger
# global_logger.info('check if log file is created in new working directory')
# print(global_logger_object.logger.handlers)
# global_logger_object.logger.handlers[0].baseFilename


class Animal:
    root_dir = os.path.join("F:", "Steffen_Experiments")

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
            global_logger.info(f"Added animal: {self.animal_id}")

    def load_metadata(self, yaml_path):
        with open(yaml_path, "r") as yaml_file:
            animal_metadata_dict = yaml.safe_load(yaml_file)

        # Load any additional metadata into session object
        copy_object_attributes_to_object(
            propertie_name_list=animal_metadata_dict.keys(),
            set_object=self,
            propertie_values=animal_metadata_dict.values(),
        )

        cohort_year = animal_metadata_dict["cohort_year"]
        self.cohort_year = (
            int(cohort_year)
            if type(cohort_year) == str
            else int(cohort_year[0]) if type(cohort_year) == list else cohort_year
        )
        self.dob = animal_metadata_dict["dob"]
        for animal_id_key in ["name", "animal_id"]:
            if animal_id_key in animal_metadata_dict.keys():
                self.animal_id = animal_metadata_dict["animal_id"]
        self.sex = animal_metadata_dict["sex"]
        return animal_metadata_dict

    def get_session_data(
        self,
        path,
        generate=False,
        regenerate=False,
        restore=False,
        unit_ids="all",
        delete=False,
        print_loading=True,
    ):
        session_yaml_fnames = get_files(path, ending=".yaml")
        match = None
        session = None
        if session_yaml_fnames:
            for session_yaml_fname in session_yaml_fnames:
                match = True
                session_yaml_path = os.path.join(path, session_yaml_fname)
                session = Session(
                    session_yaml_path,
                    animal_id=self.animal_id,
                    unit_ids=unit_ids,
                    print_loading=print_loading,
                )
                # checking if sessin is correct
                if str(session.date) not in session_yaml_fname:
                    print(
                        f"Yaml file naming does not match session date: {session_yaml_fname} != {session.date}"
                    )
                    global_logger.warning(
                        f"Yaml file naming does not match session date: {session_yaml_fname} != {session.date}"
                    )
                    match = False
                if match:
                    session.pday = (
                        num_to_date(session.date) - num_to_date(self.dob)
                    ).days
                    session.load_data(
                        restore=restore,
                        generate=generate,
                        regenerate=regenerate,
                        delete=delete,
                    )
                    break
                else:
                    print(f"Reading next yaml file")
                    global_logger.info(f"Reading next yaml file")

        if match:
            self.sessions[session.session_id] = session
            self.sessions = {
                session_id: session
                for session_id, session in sorted(self.sessions.items())
            }
        else:
            # print(f"No matching yaml file found. Skipping session path {path}")
            global_logger.error(
                f"No matching yaml file found. Skipping session path {path}"
            )
        return session

    def get_pdays(self):
        pdays = []
        for session_id, session in self.sessions.items():
            pdays.append(session.pday)
        return pdays

    def get_overview(self):
        print("-----------------------------------------------")
        print(f"{self.animal_id} born: {self.dob} sex: {self.sex}")
        # TODO: would be usefull for others
        # overview_df = pd.DataFrame(columns = ['session_name', 'date', 'P', 'suite2p_dirs'])#, 'duration [min]'])
        # for session_id, session in self.sessions.items():
        #    overview_df.loc[len(overview_df)] = {'session_name': session_id, 'date': session.session_date, 'P':session.age, 'suite2p_dirs':session.suite2p_dirs}
        # display(overview_df)
        print("-----------------------------------------------")


class Session:
    fluoresence_fname = "F.npy"
    cabincorr_fname = "binarized_traces.npz"
    cell_geldrying_fname = "cell_drying.npy"
    iscell_fname = "iscell.npy"
    binary_fname = "data.bin"
    raw_fname = "Image_001_001.raw"
    vr_root_folder = "0000VR"  # VR
    cam_root_folder = "0000MC"  # Cam Data
    cam_top_root_folder = "000BSM"  # Top View Mousecam
    movement_root_folder = "TRD-2P"  # Treadmil 2P
    movement_1p_root_folder = "TRD-1P"  # Treadmil 1P
    movement_train_root_folder = "TRD-TR"  # Treadmil training
    inscopix_root_folder = "001P-I"  # 1P Inscopix
    s2p_root_folder = "002P-F"  # 2P Femtonics
    s2p_Thorlabs_root_folder = "002P-T"  # 2P Torlabs
    s2p_Bscope_data_folder = "data"  # 2P Bscope raw data folder

    def __init__(
        self,
        yaml_file_path,
        animal_id=None,
        generate=False,
        regenerate=False,
        unit_ids="all",
        delete=False,
        print_loading=True,
    ) -> None:
        self.animal_id = animal_id
        self.yaml_file_path = yaml_file_path
        self.image_x_size = 512
        self.image_y_size = 512
        self.fucntional_channel = 1
        self.ops = None
        # Initiate session properties
        properties = [
            "session_id",
            "fps",
            "date",
            "method",
            "pday",
            "mesc_munit_pairs",
            "session_parts",  # session attributes
            "yx_shift",
            "rot_center_yx",
            "rot_angle",
            "session_type",
            "water_deprivation",  # BMI
            "units",
            "session_dir",
            "suite2p_root_dir",
            "raw_data_dir",
            "movement_dir",
            "binary_path",
            "refImg",
            "refAndMasks",
            "c",
            "contours",
            "footprints",
            "ops",
            "merged_unit",
            "cell_geldrying",
            "cells",
        ]

        copy_object_attributes_to_object(properties, self)

        self.load_metadata(yaml_file_path)
        updated_txt = "updated " if self.session_type else ""
        if print_loading:
            # print(f"Initialized {updated_txt}session: {animal_id} {self.session_id}")
            print(f"Initialized session: {animal_id} {self.session_id}")
            global_logger.info(f"Initialized session: {animal_id} {self.session_id}")

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
        copy_object_attributes_to_object(
            propertie_name_list=session_metadata_dict.keys(),
            set_object=self,
            propertie_values=session_metadata_dict.values(),
        )

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

        needed_variables = ["date"]
        for needed_variable in needed_variables:
            defined_variable = getattr(self, needed_variable)
            if defined_variable == None:
                global_logger.critical(
                    f"Variable {needed_variable} is not defined in yaml file {yaml_path}"
                )
                raise NameError(
                    f"Variable {needed_variable} is not defined in yaml file {yaml_path}"
                )

    def update_paths(self):
        """
        Update file paths within the session object based on session metadata.

        This function updates file paths within the session object based on session metadata and default values.
        It constructs paths for the session directory, Suite2P output, and binary data, as well as checks for old backup files.

        """
        # load session data paths
        self.session_dir = (
            os.path.join(Animal.root_dir, self.animal_id, str(self.session_id))
            if not self.session_dir
            else self.session_dir
        )

        self.suite2p_root_dir = (
            os.path.join(
                Animal.root_dir,
                self.animal_id,
                str(self.session_id),
                Session.s2p_root_folder,
            )
            if not self.suite2p_root_dir
            else self.suite2p_root_dir
        )

        self.raw_data_dir = (
            os.path.join(
                Animal.root_dir,
                self.animal_id,
                str(self.session_id),
                Session.s2p_Bscope_data_folder,
            )
            if not self.raw_data_dir
            else self.raw_data_dir
        )

        self.movement_dir = (
            os.path.join(
                Animal.root_dir,
                self.animal_id,
                str(self.session_id),
                Session.movement_root_folder,
            )
            if not self.movement_dir
            else self.movement_dir
        )

        # TODO: Change below for multiple raw files for single day
        self.raw_data_paths = self.get_data_paths(
            directories=self.raw_data_dir, ending=".raw"
        )

        # setting suite2p root dir to raw data dir if there are raw data paths (Bscope setting)
        if self.raw_data_paths:
            self.suite2p_root_dir = self.raw_data_dir
            self.session_parts = self.get_session_parts(self.raw_data_paths)

        self.s2p_output_folder = os.path.join(self.suite2p_root_dir, "tif")
        self.mesc_data_paths = self.get_data_paths(
            directories=self.suite2p_root_dir, ending=".mesc"
        )
        self.mesc_munit_pairs = self.define_mesc_munit_pairs()
        self.tiff_data_paths = self.get_data_paths(
            directories=self.suite2p_root_dir, ending=".tiff"
        )
        self.session_parts = self.get_session_parts()
        self.suite2p_dirs = self.get_data_paths(
            directories=self.s2p_output_folder, regex_search="suite2p", folder=True
        )
        self.suite2p_plane0_paths = (
            [os.path.join(s2p_fpath, "plane0") for s2p_fpath in self.suite2p_dirs]
            if self.suite2p_dirs
            else None
        )
        self.cabincorr_data_paths = self.get_data_paths(
            directories=self.suite2p_plane0_paths, regex_search=Session.cabincorr_fname
        )

    def get_data_paths(
        self, directories=None, ending="", regex_search=None, folder=False
    ):
        # Search for file names with specific ending and naming content
        directories = make_list_ifnot(directories) if directories else []
        fpaths = None
        for directory in directories:
            if not regex_search:
                if ending == ".mesc":
                    regex_search = "S[0-9]"
                elif ending == ".tiff":
                    regex_search = "MUnit"
                elif ending == ".suite2p":
                    regex_search = "suite2p"
                elif ending == ".raw":
                    regex_search = "Image"
                else:
                    regex_search = ""
            if folder:
                fnames = get_directories(directory, regex_search=regex_search)
            else:
                fnames = get_files(directory, ending=ending, regex_search=regex_search)

            directory_fpaths = None
            if fnames:
                directory_fpaths = []
                fpaths = [] if not fpaths else fpaths
                for fname in fnames:
                    usefull = True
                    fpath = os.path.join(directory, fname)

                    if (
                        regex_search == "suite2p" and folder
                    ):  # auto filter for usefull suite2p folders
                        fluorescence_path = search_file(
                            fpath, Session.fluoresence_fname
                        )
                        usefull = True if fluorescence_path else False

                    if usefull:
                        directory_fpaths.append(fpath)
                fpaths += directory_fpaths
        return fpaths

    def get_session_parts(self, file_names=None):
        # get session parts from MESC file name if available
        if not self.session_parts:
            file_names = file_names if file_names else self.mesc_data_paths

            file_names = make_list_ifnot(file_names) if file_names else []
            session_parts = []
            for file_name in file_names:
                splitter = "\\" if "\\" in file_name else "/"
                last_fname_part = (
                    file_name.split(splitter)[-1].split("_")[-1].split(".")[0]
                )
                # get session parts from mesc file
                session_parts += re.findall("S[0-9]", last_fname_part)
                # get session parts from raw
                session_parts += [last_fname_part]

            self.session_parts = np.unique(session_parts).tolist()
        return self.session_parts

    def get_mesc_fps(self, mesc_fpath=None):
        if self.fps:
            return self.fps
        self.fps = None
        if not mesc_fpath:
            mesc_fpath = self.mesc_data_paths[0]
        if mesc_fpath:
            with h5py.File(mesc_fpath, "r") as file:
                msessions = [
                    msession_data
                    for name, msession_data in file.items()
                    if "MSession" in name
                ]
                msession = msessions[0]
                # msession_attribute_names = list(msession.attrs.keys())
                munits = (
                    [
                        munit_data
                        for name, munit_data in msession.items()
                        if "MUnit" in name
                    ]
                    if len(msessions) > 0
                    else []
                )
                # munit_attribute_names = list(munit.attrs.keys())
                frTimes = (
                    [
                        munit.attrs["ZAxisConversionConversionLinearScale"]
                        for munit in munits
                    ]
                    if len(munits) > 0
                    else None
                )
                if frTimes:
                    frTime = max(frTimes)  # in milliseconds
                    self.fps = 1000 / frTime
        else:
            # print(f"No mesc path found in {self.suite2p_root_dir}")
            global_logger.error(f"No mesc path found in {self.suite2p_root_dir}")
        return self.fps

    def get_recording_munits(self, mesc_fpath, fps=30, at_least_minutes_of_recording=5):
        # Get MUnit number list of first Mescfile session MSession_0
        with h5py.File(mesc_fpath, "r") as file:
            munits = file[list(file.keys())[0]]
            recording_munits = []
            for name, unit in munits.items():
                # if recording has at least x minutes
                if (
                    unit["Channel_0"].shape[0]
                    > fps * 60 * at_least_minutes_of_recording
                ):
                    unit_number = name.split("_")[-1]
                    recording_munits.append(int(unit_number))
                    # get number of imaging channels
                    number_channels = 0
                    for key in unit.keys():
                        if "Channel" in key:
                            number_channels += 1
        return recording_munits, number_channels

    def define_mesc_munit_pairs(self):
        predefined_pairs = []
        if "UseMUnits" in self.__dict__:
            predefined_pairs = self.UseMUnits
        mesc_munit_pairs = []
        if self.mesc_data_paths:
            for mesc_data_path in self.mesc_data_paths:
                splitter = "\\" if "\\" in mesc_data_path else "/"
                mesc_fnames = mesc_data_path.split(splitter)[-1]
                undefined_mesc_munit_pair = True
                for mesc_munit_pair in predefined_pairs:
                    predef_mesc_name = mesc_munit_pair[0]
                    # skip if mesc file name munits are already defined
                    if mesc_fnames in predef_mesc_name:
                        undefined_mesc_munit_pair = False
                        break
                if undefined_mesc_munit_pair:
                    # define usefull munit to merge
                    usefull_munits, number_channels = self.get_recording_munits(
                        mesc_data_path
                    )
                    mesc_munit_pairs.append([mesc_fnames, usefull_munits])
        if predefined_pairs:
            mesc_munit_pairs += predefined_pairs
        return mesc_munit_pairs  # , number_channels

    def get_all_unique_mesc_munit_combinations(self, mesc_munit_pairs=None):
        # get all possible tiff file names
        unique_combinations = []
        if not mesc_munit_pairs:
            mesc_munit_pairs = self.mesc_munit_pairs
        for mesc_fname, munits in mesc_munit_pairs:
            mesc_fname = mesc_fname.split(".mesc")[0]
            for munit in munits:
                unique_combination = f"{mesc_fname}_MUnit_{munit}"
                unique_combinations.append(unique_combination)
        return unique_combinations

    def load_data(
        self,
        unit_ids="all",
        restore=True,
        generate=False,
        regenerate=False,
        regenerate_merged=False,
        delete=False,
        topdown=False,
    ):
        self.update_paths()
        # self.ops = self.set_ops()
        if topdown:
            # generate top down pricipal
            self.generate_cabincorr(
                generate=generate, regenerate=regenerate, unit_ids=unit_ids
            )
            units = self.get_units(
                restore=restore,
                get_geldrying=True,
                unit_type="single",
                # unit_type="summary",
                generate=generate,
                regenerate=regenerate,
            )
        else:
            # Merging, generating cabincorr. suite2p, tiff from mesc
            self.generate_tiff(generate=generate, regenerate=regenerate, delete=delete)
            self.generate_suite2p(
                generate=generate,
                regenerate=regenerate,
                unit_ids=unit_ids,
                delete=delete,
            )
            self.generate_cabincorr(
                generate=generate,
                regenerate=regenerate,
                regenerate_merged=regenerate_merged,
                unit_ids=unit_ids,
            )
            # units = self.get_units(restore=restore,
            #                       get_geldrying=True,
            #                       unit_type="single",
            #                       generate=generate,
            #                       regenerate=regenerate)
            # merged_unit = self.merge_units(generate=generate, regenerate=regenerate,
            #                      compute_corrs=True, delete_used_subsessions=False)
            # session.load_corr_matrix(generate=True, regenerate=False, unit_id="merged")

    def get_unique_fout_names(self, wanted_combination=None):
        """
        Raw data is used, then the raw data paths present
        Decides if all unique mesc_munit_combinations or specific combination is used for generating tiff files.
        Or if raw data is used, then the raw data paths are used.
        """
        unique_fout_names = None
        if wanted_combination:
            unique_fout_names = wanted_combination

        elif self.raw_data_dir == self.suite2p_root_dir:  # raw data is used
            unique_fout_names = [
                raw_data_path.split(".")[0] + f"_MUnit_{i}"
                for i, raw_data_path in enumerate(self.raw_data_paths)
            ]
        else:  # mesc data is used
            mesc_munit_combinations = self.get_all_unique_mesc_munit_combinations()
            unique_fout_names = mesc_munit_combinations
        return unique_fout_names

    def generate_tiff_from_raw(self, raw_data_path, tiff_out_path):
        data = np.fromfile(raw_data_path, dtype=np.uint16).reshape(-1, 512, 512)
        print("Tiff data shape:", data.shape)
        tifffile.imwrite(tiff_out_path, data)

    def mesc_to_tiff(self, mesc_path, tiff_out_path, munit_naming):
        mesc_fucntional_channel = (
            self.fucntional_channel - 1
        )  # mesc starts with 0, suite2p with 1

        print("Mesc to Tiff...")
        global_logger.info("Mesc to Tiff...")
        data = []
        with h5py.File(mesc_path, "r") as file:
            temp = file["MSession_0"][munit_naming][
                f"Channel_{mesc_fucntional_channel}"
            ][()]
            print("    data loaded size: ", temp.shape)
            global_logger.info(f"    data loaded size: {temp.shape}")
            data.append(temp)
        data = np.vstack(data)
        print("Tiff data shape:", data.shape)

        tifffile.imwrite(tiff_out_path, data)

    def generate_tiff(
        self, wanted_combination=None, generate=False, regenerate=False, delete=False
    ):
        delete = False  # Raw is probably always usefull.
        self.tiff_data_paths = [] if generate and regenerate else self.tiff_data_paths
        self.tiff_data_paths = [] if not self.tiff_data_paths else self.tiff_data_paths

        if generate:
            unique_fout_names = self.get_unique_fout_names(wanted_combination)
            data_format = (
                ".raw" if self.raw_data_dir == self.suite2p_root_dir else ".mesc"
            )

            for tiff_fout_name in unique_fout_names:
                # if tiff file name is not in tiff_data_paths, generate it
                fname_session_parts, munit = tiff_fout_name.split("_MUnit_")
                tiff_path = os.path.join(
                    self.suite2p_root_dir, tiff_fout_name + ".tiff"
                )
                if tiff_path not in self.tiff_data_paths:

                    # define data path
                    splitter = "\\" if "\\" in tiff_path else "/"
                    data_path = os.path.join(
                        splitter.join(tiff_path.split(splitter)[:-1]),
                        fname_session_parts + data_format,
                    )

                    munit_naming = f"MUnit_{munit}"
                    print(f"{fname_session_parts} converting {munit_naming} to tiff")
                    if data_format == ".mesc":
                        self.mesc_to_tiff(data_path, tiff_path, munit_naming)
                    else:
                        self.generate_tiff_from_raw(data_path, tiff_path)

                    if delete:
                        os.remove(data_path)
                    print("Finished generating TIFF from MESC data.")
                    global_logger.info("Finished generating TIFF from MESC data.")

                else:
                    print(f" -> .tiff file already done")
                    global_logger.info(f" -> .tiff file already done")
                    print(f"{tiff_path}")
                    print(f"... skipping conversion...")
                    global_logger.info(f"... skipping conversion...")

        self.tiff_data_paths = self.get_data_paths(
            directories=self.suite2p_root_dir, ending="tiff"
        )
        return tiff_path if wanted_combination else self.tiff_data_paths

    def fname_extract_sessparts_munits(
        self,
        fname: str,
        return_string=True,
        session_regex="S[0-9]",
        munit_regex="MUnit_[0-9]",
    ):
        session_parts = re.findall(
            session_regex, fname
        )  # find corresponding session parts
        munit_parts = re.findall(munit_regex, fname)  # find MUnit naming
        if return_string:
            if len(session_parts) == 0 or len(munit_parts) != 0:
                unique_name = f"{fname}"
            else:
                unique_name = f"{'-'.join(session_parts)}_{munit_parts[0]}"
            return unique_name
        return session_parts, munit_parts

    def generate_suite2p(
        self,
        wanted_combination=None,
        generate=False,
        regenerate=False,
        unit_ids="all",
        delete=False,
        delete_bin=True,
    ):
        self.suite2p_dirs = [] if generate and regenerate else self.suite2p_dirs

        if generate:
            if not self.suite2p_dirs:
                dir_exist_create(self.s2p_output_folder)
                self.suite2p_dirs = []
            standard_s2p_path_naming = os.path.join(self.s2p_output_folder, "suite2p")

            # create specific tiff file or all combinations + empty empty string to get suite2p_dir for standard suite2p analysis
            if wanted_combination:
                unique_fout_names = [wanted_combination]
            elif unit_ids == "single":
                unique_fout_names = self.get_unique_fout_names(wanted_combination)
            elif unit_ids == "all":
                unique_fout_names = [""]
            else:
                global_logger.critical(
                    f"Only options single or all are allowed for unit_ids"
                )
                raise ValueError("Only options single or all are allowed for unit_ids")

            for unique_fout_name in unique_fout_names:
                # if s2p_path is not in suite2p_dirs, generate it
                unique_s2p_folder_ending = (
                    "_" + self.fname_extract_sessparts_munits(unique_fout_name)
                    if unique_fout_name != ""
                    else ""
                )
                suite2p_dir = standard_s2p_path_naming + unique_s2p_folder_ending
                if suite2p_dir not in self.suite2p_dirs:
                    dir_exist_create(suite2p_dir)
                    # standard suite2p run
                    if unique_fout_name == "":
                        print(
                            f"Generating all possible and missing tiff files for {suite2p_dir}"
                        )
                        global_logger.info(
                            f"Generating all possible and missing tiff files for {suite2p_dir}"
                        )
                        tiff_data_paths = self.generate_tiff(
                            generate=generate, delete=delete
                        )
                        tiff_fnames = []
                        for tiff_data_path in tiff_data_paths:
                            splitter = "\\" if "\\" in tiff_data_path else "/"
                            tiff_fname = tiff_data_path.split(splitter)[-1]
                            tiff_fnames.append(tiff_fname)

                        self.run_suite2p(
                            tiff_fnames,
                            save_folder=suite2p_dir,
                            reuse_bin=False,
                            delete_bin=delete_bin,
                            move_bin=False,
                        )
                    else:
                        # Single unit mesc runmerge_units
                        tiff_fname = None
                        self.tiff_data_paths = (
                            [] if not self.tiff_data_paths else self.tiff_data_paths
                        )
                        for tiff_data_path in self.tiff_data_paths:
                            if unique_fout_name in tiff_data_path:
                                splitter = "\\" if "\\" in tiff_data_path else "/"
                                tiff_fname = tiff_data_path.split(splitter)[-1]
                                break
                        if not tiff_fname:
                            print(
                                f"Generating missing tiff file for {unique_fout_name}"
                            )
                            global_logger.info(
                                f"Generating missing tiff file for {unique_fout_name}"
                            )
                            tiff_data_path = self.generate_tiff(
                                wanted_combination=unique_fout_name,
                                generate=generate,
                                delete=delete,
                            )
                            splitter = "\\" if "\\" in tiff_data_path else "/"
                            tiff_fname = tiff_data_path.split(splitter)[-1]
                        if tiff_fname:
                            self.run_suite2p(
                                tiff_fname,
                                save_folder=suite2p_dir,
                                delete_bin=delete_bin,
                            )
                else:
                    print(f".tiff -> suite2p folder already done")
                    global_logger.info(f".tiff -> suite2p folder already done")
                    print(f"{suite2p_dir}")
                    print(f"... skipping conversion...")
                    global_logger.info(f"... skipping conversion...")

        self.suite2p_dirs = self.get_data_paths(
            directories=self.s2p_output_folder, regex_search="suite2p", folder=True
        )

        if delete:
            print("Removing Tiff...")
            global_logger.info("Removing Tiff...")
            for data_path in self.tiff_data_paths:
                os.remove(data_path)
        self.tiff_data_paths = self.get_data_paths(
            directories=self.suite2p_root_dir, ending="tiff"
        )
        return self.suite2p_dirs

    def set_ops(self, ops=None):
        """
        update ops with default options if "ops.npy" is not provided in
            - animal_id/session/002P-F or
            - animal_id/session
        """
        if not ops:
            if not self.ops:
                ops_path = os.path.join(self.suite2p_root_dir, "ops.npy")
                if os.path.exists(ops_path):
                    ops_path = os.path.join(self.session_dir, "ops.npy")
                    if os.path.exists(ops_path):
                        ops = np.load(ops_path, allow_pickle=True).item()
                if ops == None:
                    ops = default_ops()  # populates ops with the default options
                self.ops = ops
        else:
            self.ops = ops
        # self.ops["nonrigid"] = False
        return self.ops

    def run_suite2p(
        self,
        tiff_fnames,
        save_folder,
        reuse_bin=True,
        delete_old_temp_bin=True,
        delete_bin=False,
        move_bin=True,
        fps=30,
        batch_size=500,
    ):
        print(f"Starting Suite2p... saving in {save_folder}")
        global_logger.info(f"Starting Suite2p... saving in {save_folder}")

        if type(tiff_fnames) == str:
            tiff_fnames = [tiff_fnames]

        # set your options for running
        ops = self.set_ops()

        # deleting binary file from old s2p run
        if delete_old_temp_bin:
            s2p_temp_binary_location = os.path.join(
                self.suite2p_root_dir, "suite2p", "plane0", Session.binary_fname
            )
            print(f"Deleting old binary file from {s2p_temp_binary_location}")
            global_logger.info(
                f"Deleting old binary file from {s2p_temp_binary_location}"
            )
            del_file_dir(s2p_temp_binary_location)

        # reusing binary file generated in the past, if present
        if reuse_bin:
            s2p_binary_file = os.path.join(save_folder, "plane0", Session.binary_fname)
            if os.path.exists(s2p_binary_file):
                shutil.copy(s2p_binary_file, s2p_temp_binary_location)
                print(f"Reusing binary file {s2p_binary_file}")
                global_logger.info(f"Reusing binary file {s2p_binary_file}")

        # provide an h5 path in 'h5py' or a tiff path in 'data_path'
        # db overwrites any ops (allows for experiment specific settings)
        db = {
            "batch_size": batch_size,  # we will decrease the batch_size in case low RAM on computer
            "fs": fps,  # sampling rate of recording, determines binning for cell detection
            "look_one_level_down": False,  # whether to look in ALL subfolders when searching for tiffs
            "data_path": [self.suite2p_root_dir],  # a list of folders with tiffs
            #'fucntional_channel': self.fucntional_channel,
            # (or folder of folders with tiffs if look_one_level_down is True, or subfolders is not empty)
            "save_folder": save_folder,
            #'threshold_scaling': 2.0, # we are increasing the threshold for finding ROIs to limit the number of non-cell ROIs found (sometimes useful in gcamp injections)
            #'tau': 1.25,           # timescale of gcamp to use for deconvolution
            #'nimg_init': 500,      # Can create errors... how many frames to use to compute reference image for registration
            "tiff_list": tiff_fnames,
            "allow_overlap": False,  # extract signals from pixels which belong to two ROIs. By default, any pixels which belong to two ROIs (overlapping pixels) are excluded from the computation of the ROI trace.
            "delete_bin": delete_bin,  # delete binary files afterwards
            "keep_movie_raw": False,  # keep the binary file of the non-registered frames
            #'reg_tif': True,        # write the registered binary to tiff files
            "move_bin": move_bin,  # If True and ops['fast_disk'] is different from ops[save_disk], the created binary file is moved to ops['save_disk']
            "save_disk": os.path.join(
                self.suite2p_root_dir, save_folder
            ),  # Move the bin files to this location afterwards
            #'combined': False      # combine results across planes in separate folder “combined” at end of processing.
        }

        # run one experiment
        opsEnd = run_s2p(ops=ops, db=db)
        print("Finished Suite2p.")
        global_logger.info("Finished Suite2p.")

    def generate_cabincorr(
        self,
        wanted_combination=None,
        generate=False,
        regenerate=False,
        regenerate_merged=False,
        unit_ids="all",
        delete=False,
        compute_corrs=False,
        parallel=True,
    ):
        self.cabincorr_data_paths = (
            [] if not self.cabincorr_data_paths else self.cabincorr_data_paths
        )

        if generate:
            standard_s2p_path_naming = os.path.join(self.s2p_output_folder, "suite2p")
            if wanted_combination:
                wanted_fname = self.fname_extract_sessparts_munits(wanted_combination)
                s2p_path = standard_s2p_path_naming + "_" + wanted_fname
                if not self.suite2p_dirs or s2p_path not in self.suite2p_dirs:
                    self.generate_suite2p(
                        wanted_combination=wanted_combination,
                        generate=generate,
                        delete=delete,
                    )
            elif unit_ids == "all":
                s2p_path = standard_s2p_path_naming
                if not self.suite2p_dirs or s2p_path not in self.suite2p_dirs:
                    self.generate_suite2p(
                        unit_ids="all", generate=generate, delete=delete
                    )
            elif unit_ids == "merged":
                s2p_path = standard_s2p_path_naming + "_merged"
                if (
                    not self.suite2p_dirs
                    or s2p_path not in self.suite2p_dirs
                    or regenerate_merged
                ):
                    self.merge_units(
                        unit_type="single",
                        regenerate=regenerate_merged,
                        delete_used_subsessions=delete,
                        parallel=parallel,
                    )
            elif unit_ids == "single":
                suite2p_dirs = []
                unique_fout_names = self.get_unique_fout_names()
                for unique_fout_name in unique_fout_names:
                    # if s2p_path is not in suite2p_dirs, generate it
                    unique_s2p_folder_ending = self.fname_extract_sessparts_munits(
                        unique_fout_name
                    )
                    suite2p_dir = (
                        standard_s2p_path_naming + "_" + unique_s2p_folder_ending
                    )
                    if suite2p_dir not in self.suite2p_dirs:
                        self.generate_suite2p(
                            wanted_combination=unique_fout_name,
                            generate=generate,
                            delete=delete,
                        )
                        suite2p_dirs.append(suite2p_dir)
                s2p_path = suite2p_dirs
            else:
                global_logger.critical(
                    f"Only options [single, all, merged] are allowed for unit_ids"
                )
                raise ValueError(
                    "Only options [single, all, merged] are allowed for unit_ids"
                )

            s2p_paths_to_look_at = [s2p_path] if type(s2p_path) != list else s2p_path
            for s2p_path in s2p_paths_to_look_at:
                data_dir = os.path.join(s2p_path, "plane0")
                c = run_cabin_corr(
                    Animal.root_dir,
                    data_dir=data_dir,
                    regenerate=regenerate,
                    animal_id=self.animal_id,
                    session_id=self.session_id,
                    compute_corrs=compute_corrs,
                    parallel=parallel,
                )

        self.update_paths()
        return self.cabincorr_data_paths

    def load_cabincorr_data(self, unit_id="all"):
        bin_traces_zip = None
        if self.cabincorr_data_paths:
            for path in self.cabincorr_data_paths:
                splitter = "\\" if "\\" in path else "/"
                path_unit = path.split("suite2p")[-1].split(splitter)[0]
                if (
                    path_unit == "_" + unit_id
                    or unit_id == "all"
                    and len(path_unit) == 0
                ):
                    if os.path.exists(path):  # pathnames changed
                        bin_traces_zip = np.load(path, allow_pickle=True)
                    else:
                        # print("No CaBincorrPath found")
                        global_logger.error("No CaBincorrPath found")
        return bin_traces_zip

    def load_fluoresence_from_cabincorr(
        self, unit_id="all", fluorescence_type="F_detrended"
    ):
        fluorescence = None
        bin_traces_zip = self.load_cabincorr_data(unit_id=unit_id)
        if bin_traces_zip:
            if fluorescence_type in list(bin_traces_zip.keys()):
                fluorescence = bin_traces_zip[fluorescence_type]
            else:
                print(
                    f"{self.animal_id} {self.session_id} No fluorescence data of type {fluorescence_type} in binarized_traces.npz"
                )
                global_logger.error(
                    f"{self.animal_id} {self.session_id} No fluorescence data of type {fluorescence_type} in binarized_traces.npz"
                )
        else:
            print(f"{self.animal_id} {self.session_id} no binarized_traces.npz found")
            global_logger.error(
                f"{self.animal_id} {self.session_id} no binarized_traces.npz found"
            )
        return fluorescence

    def get_cells(self, merged=True, generate=False, regenerate=False):
        if self.cells and not regenerate:
            return self.cells

        found = False
        s2p_path = None
        if merged:
            print(f"Searing for suite2p_merged folder...")
            global_logger.info(f"Searing for suite2p_merged folder...")
            for s2p_path in self.suite2p_dirs:
                if "merged" in s2p_path:
                    found = True
                    print(f"Loading Cells from merged Suite2P folder {s2p_path}")
                    global_logger.info(
                        f"Loading Cells from merged Suite2P folder {s2p_path}"
                    )
                    break
        if not found or not merged:
            if merged == True:
                # print(f"Path to suite2p_merged not found.")
                global_logger.error(f"Path to suite2p_merged not found.")
            print(f"Searching for standard Suite2p folder...")
            global_logger.info(f"Searching for standard Suite2p folder...")
            for s2p_path in self.suite2p_dirs:
                if s2p_path.split("suite2p")[-1] == "":
                    found = True
                    print(f"Loading Cells from standard Suite2P folder {s2p_path}")
                    global_logger.info(
                        f"Loading Cells from standard Suite2P folder {s2p_path}"
                    )
                    break
        if not found:
            # print(f"No matching Suite2p folder found")
            global_logger.error(f"No matching Suite2p folder found")
            return None

        cell_fname = str(0) + ".npz"
        cell_npz_path = search_file(s2p_path, cell_fname)
        if not cell_npz_path:
            data_dir = os.path.join(s2p_path, "plane0")
            c = run_cabin_corr(
                Animal.root_dir,
                data_dir=data_dir,
                animal_id=self.animal_id,
                session_id=self.session_id,
                compute_corrs=generate,
                regenerate=regenerate,
            )
            cell_npz_path = search_file(s2p_path, cell_fname)
        if cell_npz_path:
            corr_path = search_file(s2p_path, cell_fname).split(cell_fname)[0]
            cells = {}
            for cell_fname in get_files(corr_path):
                cell_id = int(cell_fname.split(".npz")[0])
                cells[cell_id] = Cell(
                    self.animal_id, self.session_id, cell_id, s2p_path
                )
            # sort dictionary
            cells_sorted = {cell_id: cell for cell_id, cell in sorted(cells.items())}
            self.cells = cells_sorted
        else:
            # print(f"{cell_fname} not found in subdirectories of {s2p_path}")
            global_logger.error(
                f"{cell_fname} not found in subdirectories of {s2p_path}"
            )
        return self.cells

    def create_corr_matrix(
        self, corr_matrix_path, generate=False, regenerate=False, merged=True
    ):
        print("Loading correlation data from individual cell.npz files...")
        global_logger.info("Loading correlation data from individual cell.npz files...")
        corr_matrix, pval_matrix, z_score_matrix = None, None, None
        cells = self.get_cells(merged, generate=generate, regenerate=regenerate)
        if type(cells) == dict:
            pearson_corrs = []
            pvalue_pearson_corrs = []
            z_scores = []
            num_cells = len(cells)
            for cell_id, cell in cells.items():
                (
                    cell_pearson_corr,
                    cell_pvalue,
                    cell_z_scores,
                ) = cell.get_corr_pval_zscore()
                cell_pearson_corr, cell_pvalue, cell_z_scores = (
                    cell_pearson_corr[:num_cells],
                    cell_pvalue[:num_cells],
                    cell_z_scores[:num_cells],
                )
                pearson_corrs = np.concatenate([pearson_corrs, cell_pearson_corr])
                pvalue_pearson_corrs = np.concatenate(
                    [pvalue_pearson_corrs, cell_pvalue]
                )
                z_scores = np.concatenate([z_scores, cell_z_scores])

            corr_matrix = pearson_corrs.reshape([num_cells, num_cells])
            pval_matrix = pvalue_pearson_corrs.reshape([num_cells, num_cells])
            z_score_matrix = z_scores.reshape([num_cells, num_cells])

            print("Saving correlation matrix")
            global_logger.info("Saving correlation matrix")
            np.save(corr_matrix_path, (corr_matrix, pval_matrix, z_score_matrix))
        return corr_matrix, pval_matrix, z_score_matrix

    def load_corr_matrix(
        self, unit_id="merged", generate=False, regenerate=False, remove_geldrying=True
    ):
        """
        Loads the correlation matrix for the specified unit ID.

        This function first checks if the correlation matrix file exists for the specified unit ID. If it does, it loads the correlation matrix and p-value matrix from the file. If it does not exist, it loads the correlation data from individual cell.npz files and saves the correlation matrix and p-value matrix to a file.

        :param unit_id: The unit ID for which to load the correlation matrix. Can be "merged" or a specific unit ID. Defaults to "merged".
        :type unit_id: str
        :return: A tuple containing the correlation matrix and p-value matrix.
        :rtype: tuple
        """
        corr_matrix, pval_matrix, z_score_matrix = None, None, None
        merged = True if unit_id == "merged" else False
        s2p_folder_ending = "merged" if merged else unit_id
        s2p_folder_ending = "" if s2p_folder_ending == "all" else s2p_folder_ending
        suite2p_unit_path = None
        if self.suite2p_dirs:
            for path in self.suite2p_dirs:
                if (
                    path.split("suite2p")[-1] == s2p_folder_ending
                    or path.split("suite2p")[-1] == "_" + s2p_folder_ending
                ):
                    suite2p_unit_path = path
                    break
        if not suite2p_unit_path:
            # print(f"No suite2p path found for unit {unit_id}")
            global_logger.error(f"No suite2p path found for unit {unit_id}")
            return corr_matrix, pval_matrix, z_score_matrix

        corr_matrix_path = os.path.join(
            suite2p_unit_path, "plane0", f"allcell_corr_pval_zscore.npy"
        )
        cleaned_corr_matrix_path = os.path.join(
            suite2p_unit_path, "plane0", f"allcell_clean_corr_pval_zscore.npy"
        )

        if not os.path.exists(corr_matrix_path):
            if generate:
                corr_matrix, pval_matrix, z_score_matrix = self.create_corr_matrix(
                    corr_matrix_path, merged=merged, generate=generate
                )
            else:
                # print("No correlation data. Returning None, None")
                global_logger.error("No correlation data. Returning None, None")
        else:
            if regenerate:
                corr_matrix, pval_matrix, z_score_matrix = self.create_corr_matrix(
                    corr_matrix_path,
                    merged=merged,
                    generate=generate,
                    regenerate=regenerate,
                )
            else:
                print(f"Loading {corr_matrix_path}")
                global_logger.info(f"Loading {corr_matrix_path}")
                corr_matrix, pval_matrix, z_score_matrix = np.load(corr_matrix_path)

        if remove_geldrying and unit_id == "merged" and type(corr_matrix) == np.ndarray:
            # removes geldrying cells in matrix with shape (#cell x #cells)
            geldrying = self.load_geldrying()
            if type(geldrying) == np.ndarray:
                geldrying_indexes = np.argwhere(geldrying == True).flatten()
                number_cells = corr_matrix.shape[0]
                if sum(geldrying_indexes >= number_cells) > 0:
                    # print(
                    #    f"ERROR ERROR ERROR geldrying indexes do not correspond to cells in correlation matrix: {len(geldrying_indexes)} != {corr_matrix.shape[0]}"
                    # )
                    global_logger.error(
                        f"ERROR ERROR ERROR geldrying indexes do not correspond to cells in correlation matrix: {len(geldrying_indexes)} != {corr_matrix.shape[0]}"
                    )
                    return None, None, None
                corr_matrix = remove_rows_cols(
                    corr_matrix, geldrying_indexes, geldrying_indexes
                )
                pval_matrix = remove_rows_cols(
                    pval_matrix, geldrying_indexes, geldrying_indexes
                )
                z_score_matrix = remove_rows_cols(
                    z_score_matrix, geldrying_indexes, geldrying_indexes
                )
                if not os.path.exists(cleaned_corr_matrix_path):
                    np.save(
                        cleaned_corr_matrix_path,
                        (corr_matrix, pval_matrix, z_score_matrix),
                    )
                elif os.path.getmtime(cleaned_corr_matrix_path) < os.path.getmtime(
                    corr_matrix_path
                ):
                    np.save(
                        cleaned_corr_matrix_path,
                        (corr_matrix, pval_matrix, z_score_matrix),
                    )
                print("removed gelddrying cells")
                global_logger.info("removed gelddrying cells")
        return corr_matrix, pval_matrix, z_score_matrix

    def load_geldrying(self):
        self.cell_geldrying = None
        cells_geldrying_fpath = None
        for s2p_path in self.suite2p_dirs:
            if "merged" in s2p_path:
                cells_geldrying_fpath = os.path.join(
                    s2p_path, "plane0", Session.cell_geldrying_fname
                )
                if os.path.exists(cells_geldrying_fpath):
                    self.cell_geldrying = np.load(cells_geldrying_fpath)
        if type(self.cell_geldrying) != np.ndarray:
            # print(f"File {Session.cell_geldrying_fname} not found suite2p paths")
            global_logger.error(
                f"File {Session.cell_geldrying_fname} not found suite2p paths"
            )
        return self.cell_geldrying

    def get_units(
        self,
        generate=False,
        regenerate=False,
        unit_type="single",
        get_geldrying=False,
        restore=False,
        delete=False,
        min_needed_cells_per_unit=0,
    ):
        """
        This function load data from suiet2p folders corresponding to the same Experiment (animal_id, session_id)
        units: string
            can be defined as
                'single' for loading only single units,
                'summary' for loading only units composed of all single units e.g. standard suite2p or merged suite2p without geldrying,
                'all' or loading all units from tif folder in Session.suite2p_root_dir
        """
        defined_unit_types = ["single", "summary", "all"]
        if unit_type not in defined_unit_types:
            global_logger.critical(
                f"unit_type is only defined for {defined_unit_types}"
            )
            raise ValueError(f"unit_type is only defined for {defined_unit_types}")
        units = {}

        standard_s2p_path_naming = os.path.join(self.s2p_output_folder, "suite2p")
        units_s2p_fpath = []
        summary_suite2p_folder_endings = ["", "merged"]
        for ending in summary_suite2p_folder_endings:
            if unit_type == "single":
                break
            ending = "_" + ending if ending == "merged" else ending
            units_s2p_fpath.append(standard_s2p_path_naming + ending)

        unique_fout_names = self.get_unique_fout_names()
        for unique_fout_name in unique_fout_names:
            if unit_type == "summary":
                break
            unique_s2p_folder_ending = self.fname_extract_sessparts_munits(
                unique_fout_name
            )
            s2p_path = standard_s2p_path_naming + "_" + unique_s2p_folder_ending
            units_s2p_fpath.append(s2p_path)

        for s2p_path in units_s2p_fpath:
            unit_id = s2p_path.split("suite2p")[-1]
            unit_id = unit_id[1:] if len(unit_id) > 0 else unit_id
            unit_type = (
                "summary" if unit_id in summary_suite2p_folder_endings else "single"
            )
            if not self.suite2p_dirs or s2p_path not in self.suite2p_dirs:
                if unit_id == "merged":
                    continue
                # print(f"No s2p folder found for {unit_id}: {s2p_path}.")
                global_logger.error(f"No s2p folder found for {unit_id}: {s2p_path}.")
                wanted_combination = None
                if unit_type != "single":
                    for unique_fout_name in unique_fout_names:
                        print(f"looking at {unique_fout_name}")
                        global_logger.debug(f"looking at {unique_fout_name}")
                        if unit_id in unique_fout_name:
                            wanted_combination = unique_fout_name
                            print(f"found needed combination: {wanted_combination}")
                            global_logger.debug(
                                f"found needed combination: {wanted_combination}"
                            )
                            break
                self.generate_suite2p(
                    wanted_combination=wanted_combination,
                    generate=generate,
                    regenerate=regenerate,
                    unit_ids=unit_type,
                    delete=delete,
                )

            data_path = os.path.join(s2p_path, "plane0")
            if unit_type != "summary":
                backup_path_files(data_path, restore=False)
                backup_path_files(data_path, restore=restore)
            unit = Unit(data_path, session=self, unit_id=unit_id, unit_type=unit_type)
            num_good_cells = unit.print_s2p_iscell()
            if (
                False
            ):  # (num_good_cells < min_needed_cells_per_unit):  # If less than num good cells (0) are found, skip unit
                print(
                    f"Skipping Unit {unit.unit_id} ({num_good_cells}<{min_needed_cells_per_unit} cells)"
                )
                global_logger.info(
                    f"Skipping Unit {unit.unit_id} ({num_good_cells}<{min_needed_cells_per_unit} cells)"
                )
            else:
                units[unit_id] = unit
                # single cells sliding mean detector for gel detection
                if get_geldrying and unit_id != "":
                    cell_drying = unit.get_geldrying_cells()
                    bad = sum(cell_drying)
                    good = len(cell_drying) - bad
                    print(
                        f"Autodetection Cells: {good+bad}    Good: {good}   gel drying:{bad} "
                    )
                    global_logger.info(
                        f"Autodetection Cells: {good+bad}    Good: {good}   gel drying:{bad} "
                    )
        if not self.units:
            self.units = {}
        for unit_id, unit in units.items():
            self.units[unit_id] = unit
        return self.units

    def get_most_good_cell_unit(self, unit_type="single"):
        most_good_cells = 0
        best_unit = None
        for unit_id, unit in self.units.items():
            if unit.unit_type != unit_type:
                continue
            num_good_cells = unit.num_not_geldrying()
            if num_good_cells >= most_good_cells:
                most_good_cells = num_good_cells
                best_unit = unit
        if best_unit:
            print(
                f"Best Mask has {most_good_cells} cells and is from {best_unit.unit_id}"
            )
            global_logger.info(
                f"Best Mask has {most_good_cells} cells and is from {best_unit.unit_id}"
            )
        else:
            global_logger.critical(
                f"No unit found with enough good cells and unit_type: {unit_type}."
            )
            raise ValueError(
                f"No unit found with enough good cells and unit_type: {unit_type}."
            )
        return best_unit

    def get_usefull_units(self, min_num_usefull_cells, unit_type="single"):
        """
        This method filters out units with type not "single"
        This method updates the 'usefull' attribute of each unit in the 'units' dictionary and returns a dictionary of units that have more than 'min_num_usefull_cells' number of good cells.

        :param min_num_usefull_cells: The minimum number of good cells required for a unit to be considered useful.
        :type min_num_usefull_cells: int
        :return: A dictionary of useful units where the keys are unit IDs and the values are unit objects.
        :rtype: dict
        """
        # if not self.units:
        #    self.get_units(restore=True, generate=False, unit_type=unit_type)
        if not self.units:
            self.get_units(restore=True, generate=False, unit_type=unit_type)
        for unit_id, unit in self.units.items():
            if unit.unit_type != unit_type:
                unit.usefull = False
            else:
                unit.usefull = True

            ### Deprecated !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            ##num_good_cells = unit.num_not_geldrying()
            ##if num_good_cells > min_num_usefull_cells:
            ##    unit.usefull = True
            ##else:
            ##    unit.usefull = False
            ### Deprecated !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        return {unit_id: unit for unit_id, unit in self.units.items() if unit.usefull}

    def calc_unit_yx_shifts(self, best_unit, units, num_align_frames=1000):
        """
        S2P Registration (Footprint position shift determination)
        """
        # caly yx_shift
        refImg = best_unit.get_reference_image(n_frames_to_be_acquired=1000)
        # refImg = get_reference_image(best_unit)
        refAndMasks = register.compute_reference_masks(refImg, best_unit.ops)
        # refAndMasks = register.compute_reference_masks(refImg, ops)
        for unit_id, unit in units.items():
            if unit_id == best_unit.unit_id:
                continue
            # unit.yx_shift = calc_yx_shift(refAndMasks, unit, unit.ops, num_align_frames)
            if unit.usefull:
                unit.calc_yx_shift(refAndMasks, num_align_frames=num_align_frames)

    def merge_units(
        self,
        generate=True,
        regenerate=False,
        get_geldrying=True,
        unit_type="single",
        delete_used_subsessions=False,
        compute_corrs=False,
        image_x_size=512,
        image_y_size=512,
        parallel=True,
    ):
        """
        Takes MUnits with #cells> #most_cells/3 based on best MUnit (cells withoug geldrying).
        1. stat files are merged (suite2p) + deduplicated(cabincorr algo)
        2. Individual MUnit Suite2p folders are updated based on new Stat file (suite2p)
        3. Updated MUnits are merged to create full session data saved in suite2p_merged
        4. Gel drying is calculated for merged suite2p files

        :param generate: Whether to generate a new merged unit if it does not already exist. Defaults to True.
        :type generate: bool
        :param regenerate: Whether to regenerate the merged unit even if it already exists. Defaults to False.
        :type regenerate: bool
        :param image_x_size: The x size of the image. Defaults to 512.
        :type image_x_size: int
        :param image_y_size: The y size of the image. Defaults to 512.
        :type image_y_size: int
        :return: A merged unit object.
        :rtype: Unit
        """
        generate = True if regenerate == True else generate
        merged_s2p_path = os.path.join(
            self.suite2p_dirs[0].split("suite2p")[0], "suite2p_merged", "plane0"
        )
        if os.path.exists(merged_s2p_path):
            if regenerate:
                del_file_dir(merged_s2p_path)
            else:
                merged_unit = Unit(
                    merged_s2p_path, self, f"merged", unit_type="summary"
                )
                return merged_unit

        if generate:
            if not self.units:
                self.get_units(
                    get_geldrying=True, unit_type=unit_type, generate=generate
                )
            for unit_id, unit in self.units.items():
                binary_path = os.path.join(unit.suite2p_dir, Session.binary_fname)
                binary_file_present = os.path.exists(binary_path)
                if not binary_file_present:
                    ##print(f"Binary file not found in {unit.suite2p_dir}")
                    global_logger.error(f"Binary file not found in {unit.suite2p_dir}")
                    print(
                        f"recomputing suite2p for Unit {unit.animal_id} {unit.session_id} {unit_id}"
                    )
                    global_logger.info(
                        f"recomputing suite2p for Unit {unit.animal_id} {unit.session_id} {unit_id}"
                    )
                    unit_id_session_parts, unit_id_munits = unit_id.split("_MUnit_")
                    unique_fout_names = self.get_unique_fout_names()
                    for unique_fout_name in unique_fout_names:
                        if (
                            unit_id_session_parts in unique_fout_name
                            and "MUnit_" + unit_id_munits in unique_fout_name
                        ):
                            self.generate_suite2p(
                                wanted_combination=unique_fout_name,
                                generate=generate,
                                regenerate=True,
                                unit_ids=unit.unit_type,
                            )
                            unit.binary_path = find_binary_fpath(unit.suite2p_dir)

            # get unit with the most good cells (after geldrying detection)
            best_unit = self.get_most_good_cell_unit(unit_type=unit_type)
            # get units with enough usefull cells (at least 1/3 of best MUnit cells)
            min_num_usefull_cells = best_unit.num_not_geldrying() / 3
            units = self.get_usefull_units(min_num_usefull_cells, unit_type=unit_type)

            self.calc_unit_yx_shifts(best_unit, units)

            # merge statistical information of units and deduplicate
            merger = Merger()
            merged_stat = merger.merge_stat(units, best_unit, parallel=parallel)
            print(f"Number of cells after merging: {merged_stat.shape[0]}")
            global_logger.info(f"Number of cells after merging: {merged_stat.shape[0]}")

            updated_units = {}
            merged_unit_id = ""
            for unit_id, unit in units.items():
                # shift merged mask
                print(f"Updating Unit {unit_id}")
                global_logger.info(f"Updating Unit {unit_id}")
                merger.shift_update_unit_s2p_files(
                    unit,
                    merged_stat,
                    image_x_size=image_x_size,
                    image_y_size=image_y_size,
                )
                # TODO: unit.change_yaml_file("updated", True)
                updated_units[unit_id] = Unit(
                    unit.suite2p_dir,
                    session=self,
                    unit_id=unit_id,
                    unit_type=unit.unit_type,
                    parallel=parallel,
                )
                updated_units[unit_id].updated = True
                merged_unit_id += str(unit_id) + "_"
            # concatenate S2P results
            ops = default_ops()
            # TODO: how to decide which .mesc was imaged first?
            merged_F, _, _, _ = merger.merge_s2p_files(
                updated_units, merged_stat, ops
            )  # best_unit.c.ops)
            # merged_F, merged_Fneu, merged_spks, merged_iscell = merger.merge_s2p_files(updated_units, merged_stat, best_unit.c.ops)

            merged_unit = Unit(
                merged_s2p_path,
                self,
                unit_id=f"{merged_unit_id}_merged",
                unit_type="summary",
                compute_corrs=compute_corrs,
                parallel=parallel,
            )
            if get_geldrying:
                merged_unit.get_geldrying_cells()

            if delete_used_subsessions:
                units_s2p_paths = [
                    unit.suite2p_dir for unit_id, unit in updated_units.items()
                ]
                del updated_units
                for path in units_s2p_paths:
                    del_file_dir(path)

            self.update_paths()
            self.merged_unit = merged_unit
            try:
                self.merged_underground_is_wheel()
            except:
                print(
                    "error in merged_underground_is_wheel, not able to create underground attribute"
                )
        return merged_unit

    def convert_movement_data(self, wheel_processing_fname="process_wheel.m"):
        """
        Converts movement data from .mat files to velocity data and saves it as .npy files.

        Args:
            wheel_processing_fname (str, optional): Filename of the Octave/MATLAB function for processing wheel data.
                Defaults to "process_wheel.m".

        Returns:
            None: If the movement directory does not exist or if there is any error during the conversion process,
                None is returned. The velocity files are saved as .npy files in the specified movement directory.

        Raises:
            Any exceptions raised during the process are logged and handled, allowing the function to continue
            processing other files.

        Note:
            This function assumes the presence of specific directories and files within the movement directory
            for processing. It utilizes external resources like the Octave/MATLAB environment and the 'oct2py'
            package for MATLAB/Octave integration.

        """
        movement_dir = self.movement_dir
        if not os.path.exists(self.movement_dir):
            # print(f"No movement data directory found: {self.movement_dir}")
            global_logger.error(
                f"No movement data directory found: {self.movement_dir}"
            )
            return None
        from oct2py import octave

        root_function_path = "movement_converter\\convert_trd"
        octave.addpath(root_function_path)
        src_dirs = ["npy-matlab", "calcium\\calcium"]
        for src_dir in src_dirs:
            octave.addpath(os.path.join(root_function_path, src_dir))

        w = wheel.Wheel()
        fnames = get_files(
            movement_dir,
            ending=".mat",
            regex_search=f"{self.animal_id}_{self.session_id}",
        )
        for fname in fnames:
            fpath = os.path.join(self.movement_dir, fname)
            session_part = re.search("S[0-9]", fname)[0]
            veloctiy_fname = f"{session_part}_velocity.npy"
            velocity_fpath = os.path.join(self.movement_dir, veloctiy_fname)
            if os.path.exists(velocity_fpath):
                print(
                    f"{self.animal_id} {self.session_id}: Velocity file already exists: {velocity_fpath} skipping"
                )
                global_logger.info(
                    f"{self.animal_id} {self.session_id}: Velocity file already exists: {velocity_fpath} skipping"
                )
                continue
            print(f"Converting {fname}")
            global_logger.info(f"Converting {fname}")
            # data.mat columns
            # 1,2 encoder
            # 3   lap detector
            # 4   galvos
            # nout=0 means that the function will not return anything.
            octave.feval(wheel_processing_fname, fpath, nout=0)
            # w.root_dir = os.path.split("wheel.npy")[0]
            w.root_dir = movement_dir
            w.load_track(session_part=session_part)
            w.compute_velocity(session_part=session_part)

    def merge_movements(
        self,
        merged=True,
        min_num_usefull_cells=0,
        movement_data_types=["wheel", "triggers", "velocity"],
    ):
        """
        Merge movement data for specified data types and store the results in attributes.

        Parameters:
        - self (object): The instance of the class calling this method.
        - min_num_useful_cells (int, optional): The minimum number of useful cells required
        for a unit to be considered. Default is 80.
        - movement_data_types (list, optional): A list of movement data types to merge.
        Default is ["wheel", "triggers", "velocity"].

        Returns:
        - list: A list containing the merged movement data arrays in the order of the
        specified data types.

        This function merges movement data for the specified data types and stores the
        results as attributes with names like "merged_wheel," "merged_triggers," etc.
        It also saves the merged data as numpy files in the 'movement_dir' directory.

        Example usage:
        >>> merged_data = obj.merge_movements(min_num_useful_cells=100, movement_data_types=["wheel", "velocity"])
        >>> print(merged_data)
        [merged_wheel_data, merged_velocity_data]
        """
        # self.convert_movement_data()
        usefull_units = (
            self.get_usefull_units(min_num_usefull_cells=min_num_usefull_cells)
            if merged
            else self.units
        )
        print(f"Merging MUnit order: {usefull_units.keys()}")
        global_logger.info(f"Merging MUnit order: {usefull_units.keys()}")

        movement_data_types = make_list_ifnot(movement_data_types)
        for movement_data_type in movement_data_types:
            merged_movement_name = (
                f"merged_{movement_data_type}" if merged else f"{movement_data_type}"
            )
            save_path = os.path.join(self.movement_dir, f"{merged_movement_name}.npy")
            setattr(self, merged_movement_name, None)
            merged_movement = None
            for unit_id, unit in usefull_units.items():
                data_present = False
                if unit.underground != "platform":
                    data = unit.load_movement(movement_data_types=movement_data_type)[0]
                    if type(data) == np.ndarray:
                        data_present = True

                # Create Dummy Data
                if not data_present:
                    global_logger.error(
                        f"No data for Unit {unit.unit_id} {unit.underground} {movement_data_type}"
                    )
                    print(f"Assuming no movement: Creating dummy data")
                    global_logger.info(f"Assuming no movement: Creating dummy data")
                    num_fluoresence_frames = unit.c.dff.shape[1]
                    num_fluoresence_frames /= (
                        0.0030211382113821137 if movement_data_type == "wheel" else 1
                    )
                    dummy_shape = (
                        [int(num_fluoresence_frames), 1]
                        if movement_data_type == "wheel"
                        else [int(num_fluoresence_frames)]
                    )
                    data = np.full(dummy_shape, np.nan)

                merged_movement = (
                    np.concatenate([merged_movement, data])
                    if type(merged_movement) == np.ndarray
                    else data
                )
            np.save(save_path, merged_movement)

            setattr(self, merged_movement_name, merged_movement)

        merged_data = [
            getattr(self, f"merged_{movement_data_type}")
            for movement_data_type in movement_data_types
        ]
        return merged_data

    def load_movements(
        self,
        merged=True,
        min_num_usefull_cells=0,
        generate=True,
        regenerate=False,
        movement_data_types=["wheel", "triggers", "velocity"],
    ):
        movements = []
        for movement_data_type in movement_data_types:
            movement_name = (
                f"merged_{movement_data_type}" if merged else f"{movement_data_type}"
            )
            if movement_name not in self.__dict__.keys():
                setattr(self, movement_name, None)

            save_path = os.path.join(self.movement_dir, f"{movement_name}.npy")
            if os.path.exists(save_path) and not regenerate:
                print(f"Loading {movement_name} from {save_path}")
                global_logger.info(f"Loading {movement_name} from {save_path}")
                data = np.load(save_path, allow_pickle=True)
                setattr(self, movement_name, data)
            elif generate:
                self.merge_movements(
                    merged=merged,
                    min_num_usefull_cells=min_num_usefull_cells,
                    movement_data_types=movement_data_type,
                )
            movements.append(getattr(self, movement_name))
        return movements

    def merged_underground_is_wheel(self, min_num_usefull_cells=0, regenerate=False):
        is_wheel = None
        save_path = os.path.join(
            self.suite2p_root_dir, "tif", "suite2p_merged", "plane0", "is_wheel.npy"
        )
        if os.path.exists(save_path) and not regenerate:
            print(f"Loading merged underground ranges from {save_path}")
            global_logger.info(f"Loading merged underground ranges from {save_path}")
            is_wheel = np.load(save_path, allow_pickle=True)
        else:
            usefull_units = self.get_usefull_units(
                min_num_usefull_cells=min_num_usefull_cells
            )
            is_wheel = []
            for unit_id, unit in usefull_units.items():
                underground = "wheel" if not unit.underground else unit.underground
                frames = unit.c.dff.shape[1]
                value = True if underground == "wheel" else False
                is_wheel += [value] * frames
            np.save(save_path, is_wheel)
        self.is_wheel = is_wheel
        return is_wheel


class Unit:
    def __init__(
        self,
        suite2p_dir,
        session: Session,
        unit_id,
        unit_type,
        compute_corrs=False,
        regenerate=False,
        parallel=True,
        print_loading=True,
    ):
        self.animal_id = session.animal_id
        self.session_id = session.session_id
        self.suite2p_root_dir = session.suite2p_root_dir
        self.movement_dir = session.movement_dir
        self.unit_id = unit_id
        self.unit_type = unit_type
        self.suite2p_dir = suite2p_dir
        self.binary_path = find_binary_fpath(self.suite2p_dir)
        self.cabincorr_data_path = os.path.join(
            self.suite2p_dir, Session.cabincorr_fname
        )

        # Initiate unit properties
        self.mesc_data_path = None
        self.session_part = None
        self.ops = None
        self.set_ops()
        self.refImg = None
        self.yx_shift = [0, 0]
        self.rot_center_yx = [0, 0]
        self.rot_angle = 0
        self.usefull = None
        self.wheel = None
        self.triggers = None
        self.velocity = None
        self.dedup_cell_ids = None
        self.get_all_sliding_cell_stat = None
        self.cell_geldrying = None
        self.cell_geldrying_reasons = None
        self.get_attributes_from_session(session)

        if print_loading:
            print(f"Loading Unit {self.animal_id} {self.session_id} {self.unit_id}")
            global_logger.info(
                f"Loading Unit {self.animal_id} {self.session_id} {self.unit_id}"
            )
        self.fucntional_channel = session.fucntional_channel
        self.c, self.contours, self.footprints = self.get_c(
            compute_corrs=compute_corrs, regenerate=regenerate, parallel=parallel
        )
        self.fluorescence = (
            butter_lowpass_filter(self.c.dff, cutoff=0.5, fs=30, order=2)
            if self.c
            else None
        )
        self.load_geldrying()

        self.updated = self.old_backup_files(self.suite2p_dir)

    def get_attributes_from_session(self, session: Session):
        """
        Extract and set attributes from a Session object to the current object.

        This function populates the attributes of the current object using relevant information
        retrieved from the provided Session object. It sets attributes related to duration, underground,
        movement data, and camera data. Additionally, it calculates and assigns the 'mesc_data_path'
        attribute based on the current object's 'unit_type' and other conditions.

        Args:
            self: The object for which attributes are being populated.
            session (Session): A Session object containing the necessary data.

        Returns:
            None

        Note:
            - This function relies on the 'copy_object_attributes_to_object' function to set the attributes
            in the current object based on the properties defined in the 'properties' list.
            - It also relies on the 'session.get_unique_fout_names()' function to
            determine the 'mesc_data_path' and other related attributes.

        Example Usage:
            obj = YourClass()
            session = Session()
            obj.get_attributes_from_session(session)
        """
        properties = ["duration", "underground", "movement_data", "cam_data"]
        copy_object_attributes_to_object(properties, self)

        # Define mesc_data_path
        if self.unit_type == "single":
            # get mesc file name and munit combinations
            unique_fout_names = session.get_unique_fout_names()
            suite2p_folder_ending = os.path.split(
                self.suite2p_dir.split("suite2p")[-1]
            )[0]
            suite2p_session_parts, suite2p_munit = suite2p_folder_ending.split("_M")
            for unique_fout_name in unique_fout_names:
                if (
                    suite2p_session_parts in unique_fout_name
                    and suite2p_munit in unique_fout_name
                ):
                    mesc_fname_session_parts, munit = unique_fout_name.split("_MUnit_")
                    self.mesc_data_path = os.path.join(
                        self.suite2p_root_dir, mesc_fname_session_parts + ".mesc"
                    )

                    # get munit index and set metadata based on parts
                    for mesc_data_fname, munits in session.mesc_munit_pairs:
                        if mesc_data_fname in self.mesc_data_path:
                            mesc_fname = os.path.split(self.mesc_data_path)[-1]
                            munit_id = int(self.unit_id.split("MUnit_")[-1])
                            munit_index = munits.index(munit_id)
                            mesc_session_parts = re.findall("S[0-9]", mesc_fname)
                            self.session_part = mesc_session_parts[munit_index]
                            propertie_values = []
                            for propertie in properties:
                                if propertie in session.__dict__.keys():
                                    prop_array = getattr(session, propertie)
                                    # FIXME: why even not fitting? steffens spreadsheet nead to be fixed
                                    prop = (
                                        prop_array[munit_index]
                                        if munit_index < len(prop_array)
                                        else None
                                    )
                                    propertie_values.append(prop)
                            copy_object_attributes_to_object(
                                properties, self, propertie_values=propertie_values
                            )

        elif self.unit_type == "summary":  # set to session metadata
            self.mesc_data_path = session.mesc_data_paths
            self.session_part = session.session_parts
            copy_object_attributes_to_object(properties, self, get_object=session)
        else:
            global_logger.critical(f"Unknown unit_type: {self.unit_type}")
            raise ValueError(f"Unknown unit_type: {self.unit_type}")

    def old_backup_files(self, path):
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

    def get_c(self, compute_corrs=False, regenerate=False, parallel=True):
        # Merging cell footprints
        c_object, contours, footprints = None, None, None
        c = run_cabin_corr(
            Animal.root_dir,
            data_dir=self.suite2p_dir,
            animal_id=self.animal_id,
            session_id=self.session_id,
            compute_corrs=compute_corrs,
            regenerate=regenerate,
            parallel=parallel,
        )
        """
        Merging cell footprints and computing correlation data.

        Parameters:
            compute_corrs (bool): Whether to compute correlations.
            regenerate (bool): Whether to regenerate data.
            parallel (bool): Whether to run computations in parallel.

        Returns:
            Tuple of (c_object, contours, footprints), where c_object is the computed correlation data,f
            contours are the cell contours, and footprints are the cell footprints.
        """
        if c:
            c_object, contours, footprints = c, c.contours, c.footprints
        return c_object, contours, footprints

    def get_geldrying_cells(
        self,
        regenerate=False,
        parallel=True,
        bad_minutes=1.5,
        not_bad_minutes=0.5,
        mode="mean",
    ):
        """
        Detect gel drying in cells using sliding mean change analysis.

        Parameters:
            regenerate (bool): Whether to regenerate data.
            parallel (bool): Whether to run computations in parallel.
            bad_minutes (float): Duration threshold for labeling cells as "bad."
            not_bad_minutes (float): Duration threshold for labeling cells as "not bad."
            mode (str): Mode of analysis.

        Returns:
            numpy.ndarray: Boolean array where True indicates cells labeled as drying.
        """

        # detect gel_drying with sliding mean change. Too long increase of mean = bad
        # returns boolean list of cells, where True is a cell labeled as drying
        if type(self.cell_geldrying) is np.ndarray and not regenerate:
            return self.cell_geldrying
        if type(self.get_all_sliding_cell_stat) is not np.ndarray:
            anz = Analyzer()
            self.get_all_sliding_cell_stat = anz.get_all_sliding_cell_stat(
                parallel=parallel, fluorescence=self.fluorescence, mode=mode
            )
        anz = Analyzer()
        self.cell_geldrying = np.full([len(self.get_all_sliding_cell_stat)], True)
        self.cell_geldrying_reasons = [""] * len(self.get_all_sliding_cell_stat)
        for i, mean_stds in enumerate(self.get_all_sliding_cell_stat):
            self.cell_geldrying[i], self.cell_geldrying_reasons[i] = anz.geldrying(
                mean_stds,
                bad_minutes=bad_minutes,
                not_bad_minutes=not_bad_minutes,
                mode=mode,
            )
        self.geldrying_to_npy()
        return self.cell_geldrying

    def load_movement(self, movement_data_types=["wheel", "triggers", "velocity"]):
        """
        Load movement data from saved files.

        Parameters:
            movement_data_types (list): List of movement data types to load.

        Returns:
            List of numpy arrays containing the specified movement data."""

        movement_data_types = make_list_ifnot(movement_data_types)
        session_part = "merged" if self.unit_id == "merged" else self.session_part
        for movement_data_type in movement_data_types:
            variable = getattr(self, movement_data_type)
            if not type(variable) == np.ndarray:
                fname = f"{session_part}_{movement_data_type}.npy"
                fpath = os.path.join(self.movement_dir, fname)
                if os.path.exists(fpath):
                    data = np.load(fpath)
                    setattr(self, movement_data_type, data)
                else:
                    # print(f"No {movement_data_type} data found: {fpath}")
                    global_logger.error(f"No {movement_data_type} data found: {fpath}")

        loaded_data = [
            getattr(self, movement_data_type)
            for movement_data_type in movement_data_types
        ]
        return loaded_data

    def geldrying_to_npy(self):
        """
        Save the geldrying data to a NumPy file.

        This function saves the geldrying data to a NumPy file for future use."""

        fpath = os.path.join(self.suite2p_dir, Session.cell_geldrying_fname)
        np.save(fpath, self.cell_geldrying)

    def load_geldrying(self):
        """
        Load geldrying data from a saved NumPy file.

        Returns:
            numpy.ndarray: Loaded geldrying data or None if the file doesn't exist."""

        self.cell_geldrying = None
        fpath = os.path.join(self.suite2p_dir, Session.cell_geldrying_fname)
        if os.path.exists(fpath):
            self.cell_geldrying = np.load(fpath)
        return self.cell_geldrying

    def get_reference_image(
        self, n_frames_to_be_acquired=1000, image_x_size=512, image_y_size=512
    ):
        if self.refImg is None:
            b_loader = Binary_loader()
            frames = b_loader.load_binary(
                self.binary_path,
                n_frames_to_be_acquired=n_frames_to_be_acquired,
                image_x_size=image_x_size,
                image_y_size=image_y_size,
            )
            self.refImg = register.compute_reference(frames, ops=self.ops)
        return self.refImg

    def set_ops(self, ops=None):
        if not ops:
            if not self.ops:
                ops_path = os.path.join(self.suite2p_dir, "ops.npy")
                if os.path.exists(ops_path):
                    ops = np.load(ops_path, allow_pickle=True).item()
                if ops == None:
                    ops = default_ops()
                self.ops = ops
        else:
            self.ops = ops
        self.ops["nonrigid"] = False
        return self.ops

    def calc_yx_shift(
        self, refAndMasks, num_align_frames=1000, image_x_size=512, image_y_size=512
    ):
        if self.yx_shift == [0, 0]:
            b_loader = Binary_loader()
            frames = b_loader.load_binary(
                self.binary_path,
                n_frames_to_be_acquired=num_align_frames,
                image_x_size=image_x_size,
                image_y_size=image_y_size,
            )
            frames, ymax, xmax, cmax, ymax1, xmax1, cmax1, _ = register.register_frames(
                refAndMasks, frames, ops=self.ops
            )
            self.yx_shift = [round(np.mean(ymax)), round(np.mean(xmax))]
        return self.yx_shift

    def print_s2p_iscell(self):
        iscell_path = search_file(self.suite2p_dir, Session.iscell_fname)
        if iscell_path:
            iscell = np.load(iscell_path)
            num_cells = len(iscell[:, 0])
            num_good_cells = sum(iscell[:, 0])
            num_bad_cells = num_cells - num_good_cells
            print(
                f"Suite2p: Cells: {num_cells}  Good: {num_good_cells}  Bad: {num_bad_cells}"
            )
            global_logger.info(
                f"Suite2p: Cells: {num_cells}  Good: {num_good_cells}  Bad: {num_bad_cells}"
            )
        else:
            # print(f"No iscell file found in {self.suite2p_dir}")
            num_good_cells = None
            global_logger.error(f"No iscell file found in {self.suite2p_dir}")
        return num_good_cells

    def num_not_geldrying(self):
        return len(self.cell_geldrying) - sum(self.cell_geldrying)

    def update_s2p_files(self, stat):
        # Read in existing data from a suite2p run. We will use the "ops" and registered binary.
        suite2_data_path = self.suite2p_dir
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


class Cell:
    cell_geldrying_fname = "cell_drying.npy"

    def __init__(self, animal_id, session_id, cell_id, s2p_path):
        # super().__init__(animal_id, session_id, unit_ids=unit_ids)
        self.animal_id = animal_id
        self.session_id = session_id
        self.cell_id = cell_id
        self.s2p_path = s2p_path
        self.corr_path = search_file(s2p_path, str(cell_id) + ".npz")
        self.corr_props = None
        self.geldrying = None
        self.fluorescence = None
        self.num_bursts = None

    def get_correlation_properties(self):
        if not self.corr_props:
            self.corr_props = {}
            with np.load(self.corr_path) as corr_props:
                for key, value in corr_props.items():
                    self.corr_props[key] = value
        return self.corr_props

    def get_corr_prop_names(self):
        return list(self.get_correlation_properties().keys())

    def get_corr_pval_zscore(self):
        person_corr = np.array(self.get_correlation_properties()["pearson_corr"])
        pvalue_pearson_corr = np.array(
            self.get_correlation_properties()["pvalue_pearson_corr"]
        )
        z_score_pearson_corr = np.array(
            self.get_correlation_properties()["z_score_pearson_corr"]
        )
        return person_corr, pvalue_pearson_corr, z_score_pearson_corr

    def is_geldrying(self):
        if type(self.geldrying) != bool:
            geldrying_path = search_file(self.s2p_path, Cell.cell_geldrying_fname)
            if geldrying_path:
                self.geldrying = np.load(geldrying_path)[self.cell_id]
            else:
                # print(f"No cell_drying.npy file present")
                global_logger.error(f"No cell_drying.npy file present")
        return self.geldrying

    def get_fluorescence(self):
        if type(self.fluorescence) != np.ndarray:
            fluorescence_path = search_file(self.s2p_path, Session.fluoresence_fname)
            self.fluorescence = np.load(fluorescence_path)[self.cell_id]
        return self.fluorescence

    def get_number_bursts(self):
        # TODO: needed?
        num_bursts = None
        return num_bursts


class Analyzer:
    # Pearson and histogram plot and save
    mean_threshold = 0.1
    std_threshold = 0.15
    correct_mean = 0.007428876195354758

    def __init__(self, animals={}):
        self.animals = animals

    def good_mean_std(self, mean, std):
        return (
            True
            if mean < Analyzer.mean_threshold or std > Analyzer.std_threshold
            else False
        )

    def lin_reg(self, data):
        length = np.arange(len(data))
        lin_reg = scipy.stats.linregress(length, data)
        return lin_reg

    def get_linreg_slope_intercept(self, data):
        linreg = self.lin_reg(data)
        return linreg.slope, linreg.intercept

    def cont_mean_increase(
        self, mean_stds, num_bad_means=30 * 60 * 1.5, num_not_bad_means=30 * 60 * 0.5
    ):
        """
        Check if the mean of the data increases for 1.5 minutes without a 0.5 minutes break (30fps)

        Args:
            data (numpy.ndarray): A 2D numpy array containing mean and standard deviation values.

        Returns:
            bool: True if the mean values are within the threshold, False otherwise.
        """
        bad = False
        reason = ""
        bad_mean_counter = 0
        maybe_not_bad_counter = 0
        old_mean = mean_stds[0][0]
        min_std = np.min(mean_stds[:, 1])

        for pos, mean_std in enumerate(mean_stds[1:]):
            mean = mean_std[0]
            mean_diff = mean - old_mean
            mean_diff -= min_std * abs(mean_diff / mean)
            old_mean = mean
            if math.isnan(mean):
                bad = True
                reason = "nan"
                break
            if mean_diff > 0:
                bad_mean_counter += 1
            else:
                maybe_not_bad_counter += 1
                if maybe_not_bad_counter > num_not_bad_means:
                    bad_mean_counter = 0
                    maybe_not_bad_counter = 0

            if (
                bad_mean_counter >= num_bad_means
            ):  # 1 minute wide window mean to high for 1 minute
                bad = True
                reason = "cont. increase"
                break
        return bad, reason + " c: " + str(bad_mean_counter) + " not bad " + str(
            maybe_not_bad_counter
        )  # , pos/30

    def cont_mode_increase(
        self, mode_stds, num_bad_modes=30 * 60 * 1, num_not_bad_modes=30 * 60 * 0.45
    ):
        """
        !!!!!!!!!!!!Not usefull takes too long!!!!!!!!!!!!!!!!!!!!!!!!!!!
        Check if the mode of the data increases for 1.5 minutes without a 0.45 minutes break (30fps)

        Args:
            data (numpy.ndarray): A 2D numpy array containing mean and standard deviation values.

        Returns:
            bool: True if the mean values are within the threshold, False otherwise.
        """
        print("Warning: this method is not finetuned")
        global_logger.warning("Warning: this method is not finetuned")
        bad = False
        reason = ""
        bad_mode_counter = 0
        maybe_not_bad_counter = 0
        old_mode = mode_stds[0][0]
        min_std = np.min(mode_stds[:, 1])

        for pos, mode_std in enumerate(mode_stds[1:]):
            mode = mode_std[0]
            mode_diff = mode - old_mode
            mode_diff -= min_std / (1 / (abs(mode_diff / mode)))
            old_mode = mode
            if math.isnan(mode):
                bad = True
                reason = "nan"
                break
            if mode_diff > 0:
                bad_mode_counter += 1
            else:
                maybe_not_bad_counter += 1
                if maybe_not_bad_counter > num_not_bad_modes:
                    bad_mode_counter = 0
                    maybe_not_bad_counter = 0

            if (
                bad_mode_counter >= num_bad_modes
            ):  # 1 minute wide window mode to high for 1 minute
                bad = True
                reason = "num bad"
                break
        return bad, reason + " c: " + str(bad_mode_counter) + " not bad " + str(
            maybe_not_bad_counter
        )  # , pos/30

    def geldrying(self, m_stds, bad_minutes=1.5, not_bad_minutes=0.5, mode="mean"):
        """
        Geldrying detection
        Args:
            data (numpy.ndarray): A 2D numpy array containing mean/mode and standard deviation values.

        Returns:
            bool: True if the standard deviation values are within the threshold, False otherwise.
        """
        # TODO: improve good bad detection currently only for geldrying used
        if mode == "mean":
            bad, reason = self.cont_mean_increase(
                m_stds,
                num_bad_means=30 * 60 * bad_minutes,
                num_not_bad_means=30 * 60 * not_bad_minutes,
            )
        elif mode == "mode":
            bad, reason = self.cont_mode_increase(
                m_stds,
                num_bad_modes=30 * 60 * bad_minutes,
                num_not_bad_modes=30 * 60 * not_bad_minutes,
            )
        return bad, reason

    def sliding_window(self, arr, window_size, step_size=1):
        """
        Generate sliding windows of size window_size over an array.

        Args:
            arr (list): The input array.
            k (int): The size of the sliding window.

        Yields:
            list: A sliding window of size window_size.

        Returns:
            None: If the length of the array is less than window_size.
        """
        n = len(arr)
        if n < window_size:
            return None
        window = arr[:window_size]
        yield window
        for i in range(window_size, n, step_size):
            window = np.append(window[step_size:], arr[i])
            yield window

    def sliding_mode_std(self, arr, window_size):
        """
        Compute the mode for each sliding window of size k over an array.

        Args:
            arr (list): The input array.
            k (int): The size of the sliding window.

        Returns:
            list: A list of mode values for each sliding window.
        """
        num_windows = len(arr) - window_size + 1
        mode_stds = np.zeros([num_windows, 2])
        for num, window in enumerate(self.sliding_window(arr, window_size)):
            mode_stds[num, 0], count = scipy.stats.mode(window)
            mode_stds[num, 1] = np.std(window)
        return mode_stds

    def sliding_mean_std(self, arr, window_size):
        """
        Compute the sliding window mean and standard deviation of an array.

        Parameters:
        arr (array-like): Input array.
        k (int): Window size.

        Returns:
        list: A list of tuples containing the mean and standard deviation of each window.
        """
        num_windows = len(arr) - window_size + 1
        mean_stds = np.zeros([num_windows, 2])
        for num, window in enumerate(self.sliding_window(arr, window_size)):
            mean_stds[num, 0] = np.mean(window)
            mean_stds[num, 1] = np.std(window)
        return np.array(mean_stds)

    def get_all_sliding_cell_stat(
        self,
        fluorescence,
        window_size=30 * 60,
        parallel=True,
        processes=16,
        mode="mean",
    ):
        """
        Calculate the mean and standard deviation of sliding window (default: 30*60 = 1 sec.) fluorescence for each cell.

        Args:
            fluorescence (numpy.ndarray): A 3D numpy array containing fluorescence data for each cell.

        Returns:
            numpy.ndarray: A (cells, frames, 2) Dimensional numpy array containing the mean [:,:,0] and
            standard deviation [:,:,1] of fluorescence for each cell.

        Example:
            means = np.array(get_all_sliding_cell_stat)[:,:,0]
            stds = np.array(get_all_sliding_cell_stat)[:,:,1]
        """
        if mode == "mean":
            get_all_sliding_cell_stat = parmap.map(
                self.sliding_mean_std,
                fluorescence,
                window_size,
                pm_processes=processes,
                pm_pbar=True,
                pm_parallel=parallel,
            )
        elif mode == "mode":
            get_all_sliding_cell_stat = parmap.map(
                self.sliding_mode_std,
                fluorescence,
                window_size,
                pm_processes=processes,
                pm_pbar=True,
                pm_parallel=parallel,
            )
        return get_all_sliding_cell_stat


class Vizualizer:
    def __init__(self, animals={}, save_dir=Animal.root_dir):
        self.animals = animals
        self.save_dir = os.path.join(save_dir, "figures")
        dir_exist_create(self.save_dir)
        self.max_color_number = 301
        # Collor pallet for plotting
        self.colors = mlp.colormaps["rainbow"](range(0, self.max_color_number))

    def add_animal(self, animal):
        self.animals[animal.animal_id] = animal

    def create_colorsteps(self, min_value, max_value, max_color_number=None):
        """
        This function calculates the number of color steps between a minimum and maximum value.

        :param min_value: The minimum value in the range.
        :type min_value: int or float
        :param max_value: The maximum value in the range.
        :type max_value: int or float
        :param max_color_number: The maximum number of colors to use, defaults to 250.
        :type max_color_number: int, optional
        :return: The number of color steps between the minimum and maximum values.
        :rtype: int
        """
        if not max_color_number:
            max_color_number = self.max_color_number
        value_diff = max_value - min_value if max_value - min_value != 0 else 1
        return math.floor(max_color_number / (value_diff))

    def plot_colorsteps_example(self):
        # Colorexample
        for num, c in enumerate(self.colors):
            plt.plot([num, num], color=c, linewidth=2)

        handles = []
        for age in [0, 15, 30, 50, 75, 100, 125, 150, 180, 200, 220, 240]:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=self.colors[age],
                    linewidth=2,
                    linestyle="-",
                    label=f"Age {age}",
                )
            )

        plt.legend(handles=handles)
        # plt.show()

        #### save figures

    def bursts(
        self,
        animal_id,
        session_id,
        fluorescence_type="F_raw",
        num_cells="all",
        unit_id="all",
        remove_geldrying=True,
        dpi=300,
        fps="30",
    ):
        # for s2p_folder in self.animals[animal_id].sessions[session].suite2p_dirs:
        session = self.animals[animal_id].sessions[session_id]

        fluorescence = session.load_fluoresence_from_cabincorr(
            unit_id=unit_id, fluorescence_type=fluorescence_type
        )

        if (
            remove_geldrying
            and unit_id == "merged"
            and type(fluorescence) == np.ndarray
        ):
            geldrying = session.load_geldrying()
            geldrying_indexes = np.argwhere(geldrying == True).flatten()
            fluorescence = np.delete(fluorescence, geldrying_indexes, 0)

        if type(fluorescence) == np.ndarray:
            self.traces(
                fluorescence,
                animal_id,
                session_id,
                unit_id,
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
        unit_id="all",
        num_cells="all",
        fluorescence_type="",
        low_pass_filter=True,
        fps=30,
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
        plt.figure(figsize=(12, 7))
        if num_cells != "all":
            plt.plot(fluorescence[:, : int(num_cells)])
        else:
            plt.plot(fluorescence)

        if unit_id != "all":
            file_name = f"{animal_id} {session_id} Unit {unit_id}"
        else:
            file_name = f"{animal_id} {session_id}"

        seconds = 5
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

    def save_rasters_fig(
        self,
        binarized_fluoresence,
        animal_id,
        session_id,
        unit_id="all",
        fps=30,
        num_ticks=50,
        dpi=300,
    ):
        title = f"Raster plot {animal_id} {session_id} {unit_id}"
        plt.figure(figsize=(20, 6))
        plt.title(title)
        plt.ylabel(f"Neuron id")
        plt.xlabel(f"Seconds")

        # set_xticks_plot
        num_neurons, num_frames = binarized_fluoresence.shape

        seconds_interval = 5
        written_label_steps = 2
        frame_interval = fps * seconds_interval
        time = [
            int(frame / frame_interval) * seconds_interval
            for frame in range(num_frames)
            if frame % frame_interval == 0
        ]
        steps = round(len(time) / (2 * num_ticks))
        time_shortened = time[::steps]
        pos = np.arange(0, num_frames, frame_interval)[::steps]
        labels = [
            time if num % written_label_steps == 0 else ""
            for num, time in enumerate(time_shortened)
        ]

        plt.xticks(pos, labels, rotation=40)

        # Find spike indices for each neuron
        spike_indices = np.nonzero(binarized_fluoresence)
        # Creating an empty image grid
        image = np.zeros((num_neurons, num_frames))
        # Marking spikes as pixels in the image grid
        image[spike_indices[0], spike_indices[1]] = 1
        # Plotting the raster plot using pixels
        plt.imshow(image, cmap="gray", aspect="auto", interpolation="none")
        plt.gca().invert_yaxis()  # Invert y-axis for better visualization of trials/neurons

        plt.savefig(
            os.path.join(self.save_dir, title.replace(" ", "_") + ".png"), dpi=dpi
        )
        plt.show()
        plt.close()

    def pearson_hist(
        self,
        animal_id,
        session_id,
        unit_id="all",
        remove_geldrying=True,
        dpi=300,
        generate_corr=False,
        color_classify=False,
        facecolor="tab:blue",
    ):
        title_unit_text = "Suite2P" if unit_id == "all" else unit_id
        title = f"Corr_Hist {animal_id} {session_id} {title_unit_text}"
        unit_id = "" if unit_id == "all" else unit_id
        # Create a figure and two subplots
        session = self.animals[animal_id].sessions[session_id]
        corr_matrix, pval_matrix, z_score_matrix = session.load_corr_matrix(
            unit_id, generate=generate_corr, remove_geldrying=remove_geldrying
        )
        if type(corr_matrix) == np.ndarray:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
            # First subplot
            sns.heatmap(corr_matrix, annot=False, cmap="YlGnBu", ax=ax1)
            ax1.set_xlabel("Neuron id")
            ax1.set_ylabel("Neuron id")
            ax1.set_title("Pearson Correlation Matrix")

            # Second subplot
            hist_data = (
                corr_matrix
                if isinstance(corr_matrix, np.ndarray)
                else corr_matrix.to_numpy()
            )
            mean = np.nanmean(corr_matrix)
            std = np.nanstd(corr_matrix)
            if color_classify:
                anz = Analyzer()
                corr_mean_std_good = anz.good_mean_std(mean, std)
                facecolor = facecolor if corr_mean_std_good else "tab:red"
            sns.histplot(
                data=hist_data.flatten(), binwidth=0.1, ax=ax2, facecolor=facecolor
            )
            ax2.set_title("Pearson Correlation Coefficient Histogram")
            hist_xlabel = "Coefficients combined in 0.1 size bins"
            hist_ylabel = "Number of coefficients in bin"
            ax2.set_xlabel(hist_xlabel)
            ax2.set_ylabel(hist_ylabel)
            ax2.legend()

            mean_text = f"Mean: {mean:.2}"
            std_text = f"Std: {std:.2}"

            extra = Rectangle(
                (0, 0), 1, 1, fc=facecolor, fill=True, edgecolor="none", linewidth=0
            )
            extra = Rectangle(
                (0, 0), 1, 1, fc=facecolor, fill=True, edgecolor="none", linewidth=0
            )
            plt.legend(
                [extra, extra], [mean_text, std_text], loc="upper right"
            )  # , title='Legend')

            fig.suptitle(title)
            plt.savefig(os.path.join(self.save_dir, title.replace(" ", "_")), dpi=dpi)
            plt.show()
        else:
            # print(f"No correlation data to be plotted")
            global_logger.error(f"No correlation data to be plotted")
        return corr_matrix, pval_matrix

    def pearson_kde(
        self,
        filters=[],
        unit_id="all",
        x_axes_range=[-0.5, 0.5],
        y_axes_range=None,
        generate_corr=False,
        remove_geldrying=True,
        average_by_pday=False,
        dpi=300,
        show_print=False,
    ):
        filters = make_list_ifnot(filters)
        title_unit_text = "Suite2P" if unit_id == "all" else unit_id
        title = f"All correlation coefficient KDE {filters} {title_unit_text} {x_axes_range}"
        unit_id = "" if unit_id == "all" else unit_id
        # Plot Kernel density Estimation
        filtered_animals = filter_animals(self.animals, filters)
        unique_sorted_ages, min_age, max_age = get_pday_range(filtered_animals)
        colorsteps = self.create_colorsteps(min_age, max_age)

        plt.figure()
        plt.figure(figsize=(12, 7))

        sum_corrs_by_pday = {}
        num_corrs = {}

        for animal_id, session_id, session in yield_animal_session(filtered_animals):
            age = session.pday
            print(animal_id, session_id)
            global_logger.debug(f"{animal_id} {session_id}")
            corr_matrix, pval_matrix, z_score_matrix = session.load_corr_matrix(
                unit_id, generate=generate_corr, remove_geldrying=remove_geldrying
            )
            if type(corr_matrix) != np.ndarray:
                continue
            if not average_by_pday:
                sns.kdeplot(
                    data=corr_matrix.flatten(),
                    color=self.colors[(age - min_age) * colorsteps],
                    linewidth=1,
                )  # , fill=True, alpha=.001,)#, hist_kws=dict(edgecolor="k", linewidth=2))
            else:
                if age not in sum_corrs_by_pday:
                    sum_corrs_by_pday[age] = corr_matrix.flatten()
                    num_corrs[age] = 1
                else:
                    sum_corrs_by_pday[age] = np.append(
                        sum_corrs_by_pday[age], corr_matrix
                    )
                    num_corrs[age] += 1
        if average_by_pday:
            for age, sum_corrs in sum_corrs_by_pday.items():
                corr_matrix = sum_corrs / num_corrs[age]
                sns.kdeplot(
                    data=corr_matrix,
                    color=self.colors[(age - min_age) * colorsteps],
                    linewidth=1,
                )
        handles = []
        line_plot_steps = 1
        if len(unique_sorted_ages) > 17:
            line_plot_steps = round(len(unique_sorted_ages) / 17)

        for age in np.unique(unique_sorted_ages[::line_plot_steps]):
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=self.colors[(age - min_age) * colorsteps],
                    linewidth=2,
                    linestyle="-",
                    label=f"Age {age}",
                )
            )
        # handles=[Patch(color="tab:red", label="Bad=mean+sigma > 0.3"), Patch(color="tab:blue", label="Good=mean+sigma < 0.3")]
        plt.xlim(left=x_axes_range[0], right=x_axes_range[1])
        if y_axes_range:
            plt.ylim(bottom=y_axes_range[0], top=y_axes_range[1])
            title += f" {y_axes_range}"
        plt.xlabel("Correlation")
        plt.ylabel("Frequency")
        plt.title(title)
        plt.legend(handles=handles)
        plt.savefig(
            os.path.join(self.save_dir, title.replace(" ", "_") + ".png"), dpi=300
        )
        plt.show()

    def plot_means_stds(
        self,
        filters=[],
        unit_id="",
        dpi=300,
        x_tick_jumps=4,
        generate_corr=False,
        remove_geldrying=True,
        show_print=False,
    ):
        filters = make_list_ifnot(filters)

        mean_threshold = Analyzer.mean_threshold
        std_threshold = Analyzer.std_threshold

        filtered_animals = filter_animals(self.animals, filters)
        unique_sorted_ages, min_age, max_age = get_age_range(filtered_animals)
        colorsteps = self.create_colorsteps(0, len(filtered_animals))
        drawn_animal_ids = []
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        for number, (animal_id, animal) in enumerate(filtered_animals.items()):
            ages = []
            means = []
            stds = []
            for session_id, session in animal.sessions.items():
                show_prints(show=show_print)
                corr_matrix, pval_matrix, z_score_matrix = session.load_corr_matrix(
                    unit_id, generate=generate_corr, remove_geldrying=remove_geldrying
                )
                show_prints(show=True)
                if type(corr_matrix) != np.ndarray:
                    continue
                ages.append(session.pday)
                means.append(np.nanmean(corr_matrix))
                stds.append(np.nanstd(corr_matrix))

                drawn_animal_ids.append(animal_id)
            sort_ages_ids = np.argsort(ages)
            sorted_ages = np.array(ages)[sort_ages_ids]
            sorted_means = np.array(means)[sort_ages_ids]
            sorted_stds = np.array(stds)[sort_ages_ids]
            if animal_id in drawn_animal_ids:
                ax1.plot(
                    sorted_ages,
                    sorted_means,
                    color=self.colors[number * colorsteps],
                    marker=".",
                )
                ax2.plot(
                    sorted_ages,
                    sorted_stds,
                    color=self.colors[number * colorsteps],
                    marker=".",
                )

        age_labels = [
            str(age) if num % x_tick_jumps == 0 else ""
            for num, age in enumerate(unique_sorted_ages)
        ]
        unique_draws_animal_ids = np.unique(drawn_animal_ids)
        lines = [
            Line2D(
                [0],
                [0],
                color=self.colors[number * colorsteps],
                linewidth=3,
                linestyle="-",
                label=unique_draws_animal_ids[number],
            )
            for number in range(len(unique_draws_animal_ids))
        ]
        unit_text = "Suite2P" if unit_id == "all" else unit_id
        title = f"{filters}{unit_text} Means and Standard Deviations"
        fig.suptitle(title)

        ax1.axhline(y=mean_threshold, color="r", linestyle="--", label="Mean Threshold")
        ax1.set_xticks(
            unique_sorted_ages,
            age_labels,
            rotation=40,
            ha="right",
            rotation_mode="anchor",
        )
        ax1.set_xlabel("pday")
        ax1.set_ylabel("Mean")
        ax1.set_title(f"Means of pearson correlations")
        mean_threshold_legend_object = Line2D(
            [0],
            [0],
            color="r",
            linewidth=2,
            linestyle="--",
            label=f"Mean thr={mean_threshold}",
        )
        ax1_handles = lines + [mean_threshold_legend_object]

        ax2.axhline(y=std_threshold, color="r", linestyle="--", label="Std Threshold")
        ax2.set_xticks(
            unique_sorted_ages,
            age_labels,
            rotation=40,
            ha="right",
            rotation_mode="anchor",
        )
        ax2.set_xlabel("pday")
        ax2.set_ylabel("Standard Deviation")
        ax2.set_title(f"Std of pearson correlations")
        std_threshold_legend_object = Line2D(
            [0],
            [0],
            color="r",
            linewidth=2,
            linestyle="--",
            label=f"Std thr={std_threshold}",
        )
        ax2_handles = lines + [std_threshold_legend_object]

        ax1.legend(handles=ax1_handles)
        ax2.legend(handles=ax2_handles)
        plt.savefig(
            os.path.join(self.save_dir, title.replace(" ", "_") + ".png"), dpi=300
        )
        plt.show()

    def sanky_diagram(self):
        """
        Not implemented, only example
        """
        # TODO: implement
        # https://plotly.com/python/sankey-diagram/
        import plotly.graph_objects as go

        fig = go.Figure(
            go.Sankey(
                arrangement="snap",
                node={
                    "label": ["A", "B", "C", "D", "E", "F"],
                    "x": [0.2, 0.1, 0.5, 0.7, 0.3, 0.5],
                    "y": [0.7, 0.5, 0.2, 0.4, 0.2, 0.3],
                    "pad": 10,
                },  # 10 Pixels
                link={
                    "source": [0, 0, 1, 2, 5, 4, 3, 5],
                    "target": [5, 3, 4, 3, 0, 2, 2, 3],
                    "value": [1, 2, 1, 1, 1, 1, 1, 2],
                },
            )
        )

        fig.show()
        pass

    def unit_footprints(self, unit, rot90_times=1, remove_geldrying=False, cmap=None):
        # plot footprints of a unit
        plt.figure()
        title = f"{unit.animal_id}_{unit.session_id}_MUnit_{unit.unit_id}"

        if remove_geldrying and unit.unit_id == "merged":
            geldrying = unit.cell_geldrying
            footprints = np.array(unit.footprints)[geldrying == False]
        else:
            footprints = unit.footprints
        plt.title(f"{len(footprints)} footprints {title}")
        self.footprints(footprints, rot90_times=rot90_times, cmap=cmap)
        plt.savefig(os.path.join(self.save_dir, f"Footprints_{title}.png"), dpi=300)

    def footprints(self, footprints, rot90_times=1, cmap=None):
        # plot all footprints
        for footprint in footprints:
            idx = np.where(footprint == 0)
            footprint[idx] = np.nan
            plt.imshow(np.rot90(footprint, k=rot90_times), cmap=cmap)

    def unit_contours(
        self,
        unit,
        figsize=(10, 10),
        color=None,
        plot_center=False,
        remove_geldrying=False,
        comment="",
    ):
        # Plot Contours
        plt.figure(figsize=(10, 10))
        title = f"{unit.animal_id}_{unit.session_id}_MUnit_{unit.unit_id}"

        geldrying = None
        contours = unit.contours
        drawn_cells = len(contours)
        if remove_geldrying and unit.unit_id == "merged":
            geldrying = unit.cell_geldrying
            drawn_cells = sum(geldrying == False)
        self.contours(contours, geldrying, color, plot_center, comment)
        plt.title(f"{drawn_cells} contours {title}")
        plt.savefig(os.path.join(self.save_dir, f"Contours_{title}.png"), dpi=300)

    def contour_to_point(self, contour):
        x_mean = np.mean(contour[:, 0])
        y_mean = np.mean(contour[:, 1])
        return np.array([x_mean, y_mean])

    def contours(
        self, contours, geldrying=None, color=None, plot_center=False, comment=""
    ):  # plot_contours_points
        geldrying = (
            [False] * len(contours) if type(geldrying) != np.ndarray else geldrying
        )
        drawn_cells = 0
        for contour, cell_geldrying in zip(contours, geldrying):
            if cell_geldrying:
                continue
            drawn_cells += 1
            y_corr = contour[:, 0]
            x_corr = contour[:, 1]
            plt.plot(x_corr, y_corr, color=color)
            if plot_center:
                xy_mean = self.contour_to_point(contour)
                plt.plot(xy_mean[1], xy_mean[0], ".", color=color)
        plt.title(f"{drawn_cells} Contours{comment}")

    def multi_contours(
        self,
        multi_contours,
        plot_center=False,
        colors=["red", "green", "blue", "yellow", "purple", "orange", "cyan"],
    ):
        for contours, col in zip(multi_contours, colors):
            self.contours(contours, color=col, plot_center=plot_center)

    def multi_unit_contours(
        self, units, combination=None, plot_center=False, shift=False, figsize=(20, 20)
    ):
        """
        units : dict
        combination : list of dict keys
        """
        plt.figure(figsize=figsize)
        handles = []
        plot_contours = []
        plot_colors = []
        combination = list(units.keys()) if combination == None else combination
        colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan"]
        for (unit_id, unit), col in zip(units.items(), colors):
            if unit_id not in combination:
                continue
            good_cell_contours = (
                np.array(unit.contours)[unit.cell_geldrying == False]
                if len(unit.cell_geldrying) == len(unit.contours)
                else np.array(unit.contours)
            )

            # TODO: Integrate rotation if rotation is wanted in plots
            """
            #shift, rotate contours
            all_shifted_rotated_contour_points = []
            if shift and session_id != "day0":
                yx_shift = session.yx_shift if yx_shift == None else yx_shift
                rot_angle = session.rot_angle if rot_angle == None else rot_angle
                rot_center_yx = session.rot_center_yx if rot_center_yx == None else rot_center_yx
                for points_yx in contours:
                    shifted_rotated_contour_points = merger.shift_rotate_yx_points(points_yx, 
                                                                                yx_shift=yx_shift, 
                                                                                rot_angle=rot_angle,
                                                                                rot_center_yx=rot_center_yx)
                    
                    all_shifted_rotated_contour_points.append(shifted_rotated_contour_points)
                contours = all_shifted_rotated_contour_points
                shift_label = f" yx_shift: {yx_shift}  yx_center: {rot_center_yx}  angle: {rot_angle}"
            """
            # shift contours
            good_cell_contours = (
                [
                    good_cell_contour + unit.yx_shift  # TODO: + instead of -
                    for good_cell_contour in good_cell_contours
                ]
                if shift
                else good_cell_contours
            )
            plot_colors.append(col)
            plot_contours.append(good_cell_contours)
            shift_label = (
                f" y: {unit.yx_shift[0]}  x: {unit.yx_shift[1]}" if shift else ""
            )

            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=col,
                    linewidth=2,
                    linestyle="-",
                    label=f"MUnit: {unit_id}{shift_label}",
                )
            )
        self.multi_contours(plot_contours, colors=plot_colors, plot_center=plot_center)
        plt.title(f"Contours for MUnits: {combination}")
        plt.legend(handles=handles, fontsize=20)
        shift_label = f"_shifted" if shift else ""
        plt.savefig(
            os.path.join(
                self.save_dir, f"Contours_MUnits_{combination}{shift_label}.png"
            ),
            dpi=300,
        )
        # plt.show()

    def unit_fluorescence_good_bad(
        self, unit, batch_size=10, starting=0, interactive=False, plot_duplicates=True
    ):
        cell_geldrying = unit.get_geldrying_cells()
        fluorescence = unit.fluorescence

        title = f"{unit.animal_id}_{unit.session_id}_MUnit_{unit.unit_id}"

        if not plot_duplicates:
            if isinstance(unit.dedup_cell_ids, np.ndarray):
                cell_geldrying = cell_geldrying[unit.dedup_cell_ids]
                fluorescence = fluorescence[unit.dedup_cell_ids]

        cell_geldrying = cell_geldrying[starting:]
        fluorescence = fluorescence[starting:]
        cell_geldrying_batches = split_array(cell_geldrying, batch_size)
        fluorescence_batches = split_array(fluorescence, batch_size)
        num_batches = len(fluorescence_batches)

        for i, (cell_geldrying_batch, fluorescence_batch) in enumerate(
            zip(cell_geldrying_batches, fluorescence_batches)
        ):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))

            for num, (cell_geldrying, neuron_data) in enumerate(
                zip(cell_geldrying_batch, fluorescence_batch)
            ):
                cell_number = (i) * batch_size + num if batch_size != "all" else num
                cell_number += starting
                if not cell_geldrying:
                    ax1.plot(
                        neuron_data, label=f"Cell: {cell_number}"
                    )  # {unit.cell_geldrying_reasons[cell_number]}")
                else:
                    ax2.plot(
                        neuron_data, label=f"Cell: {cell_number}"
                    )  # {unit.cell_geldrying_reasons[cell_number]}")

            bad = sum(cell_geldrying_batch)
            good = len(cell_geldrying_batch) - bad

            seconds = 5
            num_frames = 30 * seconds
            x_pos = np.arange(0, len(neuron_data), num_frames)
            x_time = [
                int(frame / num_frames) * seconds
                for frame in range(len(neuron_data))
                if frame % num_frames == 0
            ]
            num_written_labels = round(len(x_time) / 100)
            x_labels = [
                time if time % num_written_labels == 0 else "" for time in x_time
            ]
            ax1.set_xticks(x_pos, x_labels, rotation=40, fontsize=8)
            ax2.set_xticks(x_pos, x_labels, rotation=40, fontsize=8)

            batch_title = f"Batch_{i+1}_of_{num_batches}"
            legend_fontsize = 10
            fig.suptitle(f"F of {good+bad} Cells {title} {batch_title}", fontsize=20)
            ax1.set_title(f"Good Cells: {good}")
            ax1.set_ylim(bottom=-0.1, top=0.8)
            ax2.set_ylabel("F_filtered")
            ax2.set_xlabel("seconds")
            ax2.set_title(f"Bad Cells: {bad}")
            ax2.set_ylim(bottom=-0.1, top=0.8)

            if not batch_size == "all" and not batch_size > 16:
                ax1.legend(fontsize=legend_fontsize)
                ax2.legend(fontsize=legend_fontsize)

            plt.savefig(
                os.path.join(self.save_dir, f"F_slide_{title}_{batch_title}.png"),
                dpi=300,
            )
            # plt.show()
            dir_exist_create(os.path.join(self.save_dir, "html"))
            # interactive html

            if interactive:
                mpld3.save_html(
                    fig,
                    os.path.join(
                        self.save_dir, "html", f"F_slide_{title}_{batch_title}.html"
                    ),
                )

    def binary_frames(self, frames, num_images_x=2):
        num_frames = frames.shape[0]
        num_rows = round(num_frames / num_images_x)
        fig, ax = plt.subplots(
            num_rows, num_images_x, figsize=(5 * num_images_x, 5 * num_rows)
        )
        fig.suptitle(f"{num_frames} Binary Frames", fontsize=20)
        for i, image in enumerate(frames):
            x = int(i / num_images_x)
            y = i % num_images_x
            if len(ax.shape) == 2:
                ax[x, y].imshow(image)
                ax[x, y].invert_yaxis()
                ax[x, y].set_title(f"Frame {i}")
            else:
                ax[i].imshow(image)
                ax[i].invert_yaxis()
                ax[i].set_title(f"Frame {i}")
        # plt.show()

    def show_survived_cell_percentage(self, animals=None, pipeline_stats=None):
        if type(pipeline_stats) != pd.DataFrame:
            if not animals:
                animals = self.animals
            else:
                global_logger.critical(
                    f"No data was given for survived cell percentage"
                )
                raise ValueError("No data was given for survived cell percentage")
            cell_numbers_dict = extract_cell_numbers(animals)
            # Create table to show statistics for comparison of S2P vs Own Pipeline
            pipeline_stats = summary_df_s2p_vs_geldrying(cell_numbers_dict)
        fig, (ax1) = plt.subplots(1, 1, figsize=(18, 4))
        ax1.bar(pipeline_stats.index, pipeline_stats.survived_cells)

        # fig.suptitle('Survived cell percentages')
        ax1.set_ylabel("% Cells")
        ax1.set_xlabel("Animal")
        ax1.set_title(f"Survived cell: {np.mean(pipeline_stats.survived_cells):.2%}")
        animal_ids = list(pipeline_stats.index)
        ax1.set_xticks(
            range(len(animal_ids)),
            animal_ids,
            rotation=40,
            ha="right",
            rotation_mode="anchor",
        )
        for i, v in enumerate(pipeline_stats.survived_cells):
            plt.text(range(len(pipeline_stats.index))[i] - 0.2, v + 0.01, f"{v:.2%}")
        plt.savefig(
            os.path.join(self.save_dir, f"Survived_cells_after_removing_geldrying.png"),
            dpi=300,
        )

    def show_survived_cell_numbers(
        self, animals=None, cell_numbers_dict=None, min_num_cells=200
    ):
        if not cell_numbers_dict:
            if not animals:
                animals = self.animals
            else:
                global_logger.critical(f"No data was given for survived cell numbers")
                raise ValueError("No data was given for survived cell numbers")
            cell_numbers_dict = extract_cell_numbers(animals)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        for animal_id, animal in cell_numbers_dict.items():
            (
                ages,
                iscells,
                notgeldrying,
                corrs,
                gel_corrs,
            ) = get_sorted_cells_notgeldyring_lists(animal)
            usefull_iscells = iscells >= min_num_cells
            usefull_notgeldrying = notgeldrying >= min_num_cells
            if len(usefull_iscells) > 0:
                ax1.plot(
                    ages[usefull_iscells],
                    iscells[usefull_iscells],
                    label=f"{animal_id}",
                    marker=".",
                )
            if len(usefull_notgeldrying) > 0:
                ax2.plot(
                    ages[usefull_notgeldrying],
                    notgeldrying[usefull_notgeldrying],
                    label=f"{animal_id}",
                    marker=".",
                )

        title = f"Compare Cell Numbers before, after Geldrying Detector with at least {min_num_cells} Cells"
        fig.suptitle(title)
        ax1.set_ylabel("# Cells")
        ax1.set_xlabel("pday")
        ax1.set_ylim(bottom=0, top=1300)
        ax1.set_title("Suite2P iscell")
        ax1.legend()
        ax2.set_ylabel("# Cells")
        ax2.set_xlabel("pday")
        ax2.set_ylim(bottom=0, top=1300)
        ax2.set_title("Not Geldrying Cells (Own Pipeline)")
        ax2.legend()
        plt.savefig(
            os.path.join(self.save_dir, title.replace(" ", "_") + ".png"), dpi=300
        )

    def show_session_pday_histo(
        self,
        animals=None,
        cell_numbers_dict=None,
        min_num_cells=200,
        dpi=300,
        facecolor="tab:blue",
    ):
        if not cell_numbers_dict:
            if not animals:
                animals = self.animals
            else:
                global_logger.critical(f"No data was given for session comparisson")
                raise ValueError("No data was given for session comparisson")
            cell_numbers_dict = extract_cell_numbers(animals)

        pday_usefull_session_count = {}
        for animal_id, animal in cell_numbers_dict.items():
            (
                ages,
                iscells,
                notgeldrying,
                corrs,
                gel_corrs,
            ) = get_sorted_cells_notgeldyring_lists(animal)
            for age, num_iscells, num_notgeldrying in zip(ages, iscells, notgeldrying):
                if age not in pday_usefull_session_count:
                    pday_usefull_session_count[age] = {"s2p": 0, "notgeldrying": 0}
                pday_usefull_session_count[age]["s2p"] += (
                    1 if num_iscells >= min_num_cells else 0
                )
                pday_usefull_session_count[age]["notgeldrying"] += (
                    1 if num_notgeldrying >= min_num_cells else 0
                )
        pdays = sorted(list(pday_usefull_session_count.keys()))
        pday_usefull_session_count = sorted(pday_usefull_session_count.items())

        s2p_count_list = []
        notgeldrying_count_list = []
        for age, counts in pday_usefull_session_count:
            s2p_count_list.append(counts["s2p"])
            notgeldrying_count_list.append(counts["notgeldrying"])
        xticks = range(15, max(pdays) + 5, 5)

        fig, ax2 = plt.subplots(1, 1, figsize=(24, 4))
        title = f"Number of Sessions distributed across pdays with cell numbers > {min_num_cells}"
        fig.suptitle(title)
        minx, maxx = min(pdays) - 1, max(pdays) + 1

        ax2.bar(pdays, notgeldrying_count_list)
        ax2.set_title(
            f"{sum(notgeldrying_count_list)} Sessions after removing geldrying cells"
        )
        ax2.set_xlim(minx, maxx)
        ax2.set_ylabel("# Sessions")
        ax2.set_xlabel("pday")
        ax2.set_xticks(
            xticks, labels=xticks, rotation=40, ha="right", rotation_mode="anchor"
        )
        ax2.grid(color="gray", linestyle="-", linewidth=0.3)
        plt.savefig(
            os.path.join(
                self.save_dir,
                title.replace(" ", "_").replace(">", "bigger than") + ".png",
            ),
            dpi=300,
        )

    def show_usefull_sessions_comparisson(
        self,
        animals=None,
        cell_numbers_dict=None,
        min_num_cells=200,
        dpi=300,
        facecolor="tab:blue",
    ):
        if not cell_numbers_dict:
            if not animals:
                animals = self.animals
            else:
                global_logger.critical(f"No data was given for session comparisson")
                raise ValueError("No data was given for session comparisson")
            cell_numbers_dict = extract_cell_numbers(animals)

        pday_usefull_session_count = {}
        for animal_id, animal in cell_numbers_dict.items():
            (
                ages,
                iscells,
                notgeldrying,
                corrs,
                gel_corrs,
            ) = get_sorted_cells_notgeldyring_lists(animal)
            for age, num_iscells, num_notgeldrying in zip(ages, iscells, notgeldrying):
                if age not in pday_usefull_session_count:
                    pday_usefull_session_count[age] = {"s2p": 0, "notgeldrying": 0}
                pday_usefull_session_count[age]["s2p"] += (
                    1 if num_iscells >= min_num_cells else 0
                )
                pday_usefull_session_count[age]["notgeldrying"] += (
                    1 if num_notgeldrying >= min_num_cells else 0
                )
        pdays = sorted(list(pday_usefull_session_count.keys()))
        pday_usefull_session_count = sorted(pday_usefull_session_count.items())

        s2p_count_list = []
        notgeldrying_count_list = []
        for age, counts in pday_usefull_session_count:
            s2p_count_list.append(counts["s2p"])
            notgeldrying_count_list.append(counts["notgeldrying"])
        xticks = range(15, max(pdays) + 5, 5)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 4))
        title = f"Number of Sessions distributed across pdays with cell numbers > {min_num_cells}"
        fig.suptitle(title)
        minx, maxx = min(pdays) - 1, max(pdays) + 1
        ax1.set_title(f"{sum(s2p_count_list)} Sessions after Suite2P")
        ax1.bar(pdays, s2p_count_list)
        ax1.set_xlim(minx, maxx)
        ax1.set_ylabel("# Sessions")
        ax1.set_xlabel("pday")
        ax1.grid(color="gray", linestyle="-", linewidth=0.3)
        ax1.set_xticks(
            xticks, labels=xticks, rotation=40, ha="right", rotation_mode="anchor"
        )
        miny, maxy = ax1.get_ylim()

        ax2.bar(pdays, notgeldrying_count_list)
        ax2.set_title(
            f"{sum(notgeldrying_count_list)} Sessions after removing geldrying cells"
        )
        ax2.set_xlim(minx, maxx)
        # ax2.set_ylim(miny, maxy)
        ax2.set_ylabel("# Sessions")
        ax2.set_xlabel("pday")
        ax2.set_xticks(
            xticks, labels=xticks, rotation=40, ha="right", rotation_mode="anchor"
        )
        ax2.grid(color="gray", linestyle="-", linewidth=0.3)
        plt.savefig(
            os.path.join(
                self.save_dir,
                title.replace(" ", "_").replace(">", "bigger than") + ".png",
            ),
            dpi=300,
        )

    def plot_usefull_session_pdays(
        self,
        animals=None,
        cell_numbers_dict=None,
        min_num_cells=200,
        suite2p_cells=False,
        different_color_after_k_green=None,
    ):
        if not cell_numbers_dict:
            if not animals:
                animals = self.animals
            else:
                global_logger.critical(f"No data was given: cell_numbers_dict")
                raise ValueError("No data was given: cell_numbers_dict")
            cell_numbers_dict = extract_cell_numbers(animals)
        pday_cell_count_df = get_cells_pdays_df(
            cell_numbers_dict, suite2p_cells=suite2p_cells
        )
        # from pandas import *
        # display(pday_cell_count_df)
        vals = np.around(pday_cell_count_df.values, 2)
        red = mlp.colors.TABLEAU_COLORS["tab:red"]
        green = mlp.colors.TABLEAU_COLORS["tab:green"]
        blue = mlp.colors.TABLEAU_COLORS["tab:blue"]
        gray = mlp.colors.TABLEAU_COLORS["tab:gray"]
        black = mlp.colors.BASE_COLORS["k"]
        colours = []
        for animal_values in vals:
            colours.append([])
            num_green = 0
            for val in animal_values:
                col = gray
                if val < min_num_cells and val > -1:
                    col = red
                elif val >= min_num_cells:
                    num_green += 1
                    if different_color_after_k_green != None:
                        col = (
                            blue if num_green > different_color_after_k_green else green
                        )
                    else:
                        col = green
                col = mlp.colors.to_rgba(col)
                colours[-1].append(col)
            # color all green cells in blue if the number of blue cells is above 15
            # colours[-1] = colours[-1] if num_green<16 else [blue if col==mlp.colors.to_rgba(green) else col
            #                                                for col in colours[-1]]

        fig = plt.figure(figsize=(15, 3))
        title = "Usefull Sessions by Animal ID and pday"
        fig.suptitle(title)
        ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])
        ax.set_frame_on = False
        table = plt.table(  # cellText=vals,
            rowLabels=pday_cell_count_df.index,
            rowColours=[black] * len(pday_cell_count_df.index),
            colLabels=pday_cell_count_df.columns,
            colColours=[black] * len(pday_cell_count_df.columns),
            colWidths=[0.02] * vals.shape[1],
            loc="center",
            cellColours=colours,
        )
        plt.savefig(
            os.path.join(
                self.save_dir,
                title.replace(" ", "_").replace(">", "bigger than") + ".png",
            ),
            dpi=300,
        )
        plt.show()
        return pday_cell_count_df

    def plot_velocity(
        self, velocity, average=False, window_size=30, comment="", show=True
    ):
        velocity_lable = "Velocity"
        plot_velocity = velocity
        if average:
            anz = Analyzer()
            plot_velocity = anz.sliding_mean_std(velocity, window_size=window_size)[
                :, 0
            ]
            velocity_lable = "Averaged Velocity"

        title = f"Velocity {comment}"
        plt.figure(figsize=(20, 5))  # Adjust the figure size as needed
        # Plot the averaged velocity data with labels
        plt.plot(plot_velocity, label=velocity_lable)
        plt.xlabel("Frames")
        plt.ylabel("Velocity m/s")
        plt.ylim(-0.3, 0.8)
        plt.title(title)
        plt.legend()
        plt.grid(True, color="gray")
        to_replace = [
            [" ", "_"],
            [">", "bigger than"],
            ["/", ""],
            ["|", ""],
            ["%", "percentage"],
        ]
        for character, replacer in to_replace:
            title = title.replace(character, replacer)
        plt.savefig(
            os.path.join(
                self.save_dir,
                title + ".png",
            ),
            dpi=300,
        )
        if show:
            plt.show()
        plt.close()

    def sanitize_filename(self, title):
        # Remove invalid characters for filenames in Windows
        return re.sub(r'[<>:"/\\|?*%]', "_", title)

    def plot_value_per_session(
        self, values_dict, pdays, comment="", show=True, ax=None
    ):
        title = f"{comment}"
        if ax is None:
            plt.figure(figsize=(20, 5))  # Adjust the figure size as needed
            ax = plt.gca()

        # Plot the averaged velocity data with labels
        for label, values in values_dict.items():
            ax.plot(pdays, values, label=label, linewidth=2, alpha=0.9)  # Thicker lines
            ax.scatter(pdays, values, s=25, alpha=0.7)  # Dots at every value

        ax.set_xlabel("pday")
        ax.set_ylabel("value")
        ax.set_xticks(pdays)
        ax.set_title(title)
        if "%" in comment:
            ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, color="gray", alpha=0.5)

        if ax is None:  # Only save and show if no external axis is provided
            sanitized_title = self.sanitize_filename(title)
            plt.savefig(
                os.path.join(
                    self.save_dir,
                    sanitized_title + ".png",
                ),
                dpi=300,
            )

            # Show the plot
            if show:
                plt.show()
            plt.close()

    def plot_combined_values(
        self, values_dict1, values_dict2, pdays, comment1="", comment2="", show=True
    ):
        title1 = f"{comment1}"
        title2 = f"{comment2}"
        fig, axs = plt.subplots(
            2, 1, figsize=(20, 10)
        )  # Adjust the figure size as needed

        # Plot the first set of values in the first subplot
        self.plot_value_per_session(
            values_dict1, pdays, comment=title1, show=False, ax=axs[0]
        )

        # Plot the second set of values in the second subplot
        self.plot_value_per_session(
            values_dict2, pdays, comment=title2, show=False, ax=axs[1]
        )

        # Save the combined figure
        # combined_title = f"{title1.split(" ")[0]}"
        combined_title = title1.split(" ")[0]
        sanitized_combined_title = self.sanitize_filename(combined_title)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.save_dir,
                sanitized_combined_title + ".png",
            ),
            dpi=300,
        )

        # Show the combined plot
        if show:
            plt.show()
        plt.close()

    @staticmethod
    def plot_raster(
        binarized_traces,
        save_dir,
        animal_id="",
        session_id="",
        fluorescence_type="",
        fps=30,
        save=True,
        override=False,
    ):
        bin_traces = np.nan_to_num(binarized_traces.transpose(), nan=0.0)
        num_time_steps, num_neurons = bin_traces.shape

        # set image title, skip if already present
        file_name = f"{animal_id} {session_id}"
        burst_percentage = (
            np.nansum(bin_traces) / (bin_traces.shape[0] * bin_traces.shape[1]) * 100
        )
        title = f"Bursts from {file_name} {fluorescence_type} {burst_percentage:.2f}%"
        save_title = os.path.join(save_dir, title.replace(" ", "_") + ".png")
        if os.path.exists(save_title) and not override:
            return

        neurons_per_y = 50
        y_size = num_neurons / neurons_per_y
        if y_size < 5:
            y_size = 5
        plt.figure(figsize=(20, y_size))
        # Find spike indices for each neuron
        spike_indices = np.nonzero(bin_traces)
        # Creating an empty image grid
        image = np.zeros((num_neurons, num_time_steps))
        # Marking spikes as pixels in the image grid
        image[spike_indices[1], spike_indices[0]] = 1
        # Plotting the raster plot using pixels

        plt.imshow(image, cmap="binary", aspect="auto", interpolation="none")
        plt.gca().invert_yaxis()  # Invert y-axis for better visualization of trials/neurons

        xticks_frames_to_seconds(num_time_steps, fps=fps)

        xlabel = f"seconds"
        ylabel = "Neuron ID"

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if save:
            plt.savefig(save_title, dpi=300)
        plt.show()
        plt.close()

    @staticmethod
    def plot_traces_shifted(
        traces, savepath, figsize_x=20, title="Ca traces", save=True, fps=30.97
    ):
        """
        Plots traces shifted up by 10 for each trace
        """
        traces_per_y = 2
        y_size = len(traces) / traces_per_y / 3
        y_size = 1 if y_size < 1 else y_size
        figsize = (figsize_x, y_size)
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize)
        for i, trace in enumerate(traces):
            ax.plot(trace + i / traces_per_y)

        num_neurons, num_time_steps = traces.shape
        # xticks_frames_to_seconds(num_time_steps, fps=fps)
        plt.ylim(0, traces.shape[0] / traces_per_y)
        plt.xlim(0, traces.shape[1])
        plt.title(title)
        if save:
            plt.savefig(savepath)
        plt.show()
        plt.close()


class Binary_loader:
    """
    A class for loading binary data and converting it into an animation.

    This class provides methods for loading binary data from a file and converting a sequence of binary frames into an animated GIF. The `load_binary` method loads binary data from a specified file and returns it as a NumPy array. The `binary_frames_to_gif` method takes a sequence of binary frames and converts them into an animated GIF, which is saved to the specified directory.

    Attributes:
        None
    """

    def load_binary(
        self, fpath, n_frames_to_be_acquired, image_x_size=512, image_y_size=512
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
        # load binary file from suite2p_folder from unit
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
                global_logger.info(f"Frame {i}")
            p1 = ax.text(512 / 2 - 50, 0, f"Frame {i}", animated=True)
            p2 = ax.imshow(frame, animated=True)
            images.append([p1, p2])
            if i > range_end:
                break
        ani = animation.ArtistAnimation(
            fig, images, interval=delay_between_frames, blit=True, repeat_delay=1000
        )
        ani.save(gif_save_path)
        return ani


class Merger:
    """
    Merges indiviual MUnits/Subsession of a Session
    """

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
        unit: Unit = None,
        stat: np.ndarray = None,
        yx_shift: [int, int] = None,
        rot_angle: float = None,
        rot_center_yx: [float, float] = None,
        roation_first=False,
    ):
        """
        Shift and rotate the statistical data of cells within a unit.

        Parameters:
        - unit (unit, optional): The unit object containing unit-specific parameters.
        - stat (np.ndarray, optional): Statistical data for multiple cells.
        - yx_shift (list, optional): A list containing the (y, x) shift values.
        - rot_angle (float, optional): Angle of rotation in degrees.
        - rot_center_yx (list, optional): The (y, x) coordinates for the rotation center.
        - rotation_first (bool, optional): Whether to apply rotation before shifting (default is False).

        Returns:
        - np.ndarray: Modified statistical data with shifted and rotated cell information.

        This function shifts and rotates the statistical data of cells within a unit, taking into account various parameters, including shift, rotation, and rotation center.

        """
        # stat files first value ist y-value second is x-value
        stat = unit.c.stat if type(stat) != np.ndarray else stat
        yx_shift = unit.yx_shift if not yx_shift else yx_shift
        rot_angle = unit.rot_angle if rot_angle == None else rot_angle
        rot_center_yx = unit.rot_center_yx if not rot_center_yx else rot_center_yx
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

    def merge_stat(
        self, units, best_unit, parallel=True, image_x_size=512, image_y_size=512
    ):
        """
        shift and merge, deduplicate, stat files with best_unit as reference position
        """
        num_batches = get_num_batches_based_on_available_ram()

        # cooootours = create_contours(stat)
        # scooootours = create_contours(all_shifted_rotated_contour_points)
        # fcp([cooootours, scooootours], colors=["red", "green"])

        #####################DEBUGING#####################
        # viz2 = Vizualizer(save_dir="")
        # viz2.multi_unit_contours(units, plot_center=True)
        # viz2.multi_unit_contours(units, plot_center=True, shift=True)
        # plt.show()
        #####################DEBUGING#####################

        shifted_unit_stat_no_abroad = self.remove_abroad_cells(
            best_unit.c.stat,
            units,
            image_x_size=image_x_size,
            image_y_size=image_y_size,
        )
        merged_footprints = self.stat_to_footprints(shifted_unit_stat_no_abroad)
        merged_stat = shifted_unit_stat_no_abroad
        for unit_id, unit in units.items():
            if unit_id == best_unit.unit_id:
                continue
            print(f"Working on unit {unit_id}...")
            global_logger.info(f"Working on unit {unit_id}...")
            moved_session_stat = self.shift_rotate_stat_cells(unit)
            shifted_unit_stat_no_abroad = self.remove_abroad_cells(
                moved_session_stat,
                units,
                image_x_size=image_x_size,
                image_y_size=image_y_size,
            )
            shifted_footprints = self.stat_to_footprints(shifted_unit_stat_no_abroad)
            clean_cell_ids, merged_footprints = self.merge_deduplicate_footprints(
                merged_footprints,
                shifted_footprints,
                parallel=parallel,
                num_batches=num_batches,
            )
            merged_stat = np.concatenate([merged_stat, shifted_unit_stat_no_abroad])[
                clean_cell_ids
            ]
        return merged_stat

    def remove_abroad_cells(
        self, stat: np.ndarray, units: dict, image_x_size=512, image_y_size=512
    ):
        if len(stat) == 0:
            return stat
        # removing out of bound cells
        remove_cells = []
        # check for every shift and rotation combination if cell is in bounds
        for unit_id, unit in units.items():
            # negate shift and rotation angle to cover all possible cell positions
            yx_shift = list(-np.array(unit.yx_shift))
            rot_angle = -unit.rot_angle
            rot_center_yx = unit.rot_center_yx
            # if rot_angle == 0 and sum(abs(np.array(yx_shift))) == 0:
            #    continue
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
        for num, cellstat in enumerate(stat):
            for axis in ["xpix", "ypix"]:
                if sum(np.array(cellstat[axis]) >= 512):
                    print(num, cellstat[axis])
        return stat

    def merge_s2p_files(self, units, stat, ops):
        """
        Merges F, Fneu, spks, iscell from individual Units
        Does not merge the individual corrected stat files
        Does not merge ops
        """
        path = units[list(units.keys())[0]].suite2p_dir
        merged_F = np.load(os.path.join(path, "F.npy"))
        merged_Fneu = np.load(os.path.join(path, "Fneu.npy"))
        merged_spks = np.load(os.path.join(path, "spks.npy"))
        merged_iscell = np.load(os.path.join(path, Session.iscell_fname))
        for unit_id, unit in units.items():
            if unit_id == list(units.keys())[0]:
                continue
            path = unit.suite2p_dir
            F = np.load(os.path.join(path, "F.npy"))
            merged_F = np.concatenate([merged_F, F], axis=1)
            Fneu = np.load(os.path.join(path, "Fneu.npy"))
            merged_Fneu = np.concatenate([merged_Fneu, Fneu], axis=1)
            spks = np.load(os.path.join(path, "spks.npy"))
            merged_spks = np.concatenate([merged_spks, spks], axis=1)
            # sum iscells
            is_cell = np.load(os.path.join(path, Session.iscell_fname))
            merged_iscell += is_cell

        # let cells life if one of the cells is detected as cell. Average probabilities for ifcell
        merged_iscell /= len(list(units.keys()))
        merged_iscell[:, 0] = np.ceil(merged_iscell[:, 0])

        root = path.split("suite2p")[0]
        merged_s2p_path = create_dirs([root, "suite2p_merged", "plane0"])

        np.save(os.path.join(merged_s2p_path, "F.npy"), merged_F)
        np.save(os.path.join(merged_s2p_path, "Fneu.npy"), merged_Fneu)
        np.save(os.path.join(merged_s2p_path, "spks.npy"), merged_spks)
        np.save(os.path.join(merged_s2p_path, Session.iscell_fname), merged_iscell)
        np.save(os.path.join(merged_s2p_path, "stat.npy"), stat)
        np.save(os.path.join(merged_s2p_path, "ops.npy"), ops)
        return merged_F, merged_Fneu, merged_spks, merged_iscell

    def stat_to_footprints(self, stat, dims=[512, 512]):
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

    def find_overlaps1(self, ids, footprints):
        #
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

        print("... computing cell overlaps ...")
        global_logger.info("... computing cell overlaps ...")

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
        clean_cells = np.delete(np.arange(num_cells), removed_cells)
        print("Removed duplicated cells: ", len(removed_cells))
        print("Remaining cells: ", len(clean_cells))
        global_logger.info("Removed duplicated cells: " + str(len(removed_cells)))
        global_logger.info("Remaining cells: " + str(len(clean_cells)))

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

    def shift_update_unit_s2p_files(
        self, unit, new_stat, image_x_size=512, image_y_size=512
    ):
        data_path = unit.suite2p_dir
        # shift merged mask
        shift_to_unit = list(-np.array(unit.yx_shift))
        rotate_to_angle = -unit.rot_angle
        rot_center_yx = unit.rot_center_yx
        shifted_rotated_unit_stat = self.shift_rotate_stat_cells(
            stat=new_stat,
            yx_shift=shift_to_unit,
            rot_angle=rotate_to_angle,
            rot_center_yx=rot_center_yx,
            roation_first=True,
        )
        backup_path_files(data_path)
        unit.update_s2p_files(shifted_rotated_unit_stat)


def load_all(
    root_dir,
    wanted_animal_ids=["all"],
    wanted_session_ids=["all"],
    restore=False,
    generate=False,
    regenerate=False,
    unit_ids="all",
    delete=False,
    print_loading=True,
):
    """
    Loads animal data from the specified root directory for the given animal IDs.

    Parameters:
    - root_dir (string): The root directory path where the animal data is stored.
    - animal_ids (list, optional): A list of animal IDs to load. Default is ["all"].
    - generate (bool, optional): If True, generates new session data. Default is False.
    - regenerate (bool, optional): If True, regenerates existing session data. Default is False.
    - units (string, optional): Specifies the units. Default is "all".
    - delete (bool, optional): If True, deletes session data. Default is False.

    Returns:
    - animals_dict (dict): A dictionary containing animal IDs as keys and corresponding Animal objects as values.
    """
    present_animal_ids = get_directories(root_dir, regex_search="DON-")
    animals_dict = {}

    # Search for animal_ids
    if not present_animal_ids:
        raise ImportError(f"No animal_ids found in {root_dir}")
    for animal_id in present_animal_ids:
        if animal_id in wanted_animal_ids or "all" in wanted_animal_ids:
            animal = load_animal(root_dir, animal_id, wanted_session_ids, restore)
            animals_dict[animal_id] = animal
    animals_dict = {
        animal_id: animal for animal_id, animal in sorted(animals_dict.items())
    }
    return animals_dict


def load_animal(
    root_dir,
    animal_id,
    wanted_session_ids=["all"],
    restore=False,
    unit_ids="all",
    delete=False,
    print_loading=True,
):
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
                unit_ids=unit_ids,
                delete=delete,
                print_loading=print_loading,
            )
    return animal


def run_cabin_corr(
    root_dir,
    data_dir,
    animal_id,
    session_id,
    compute_corrs=False,
    regenerate=False,
    parallel=True,
):
    # Init
    current_fluorescence_data_path = search_file(data_dir, Session.fluoresence_fname)
    if current_fluorescence_data_path == None:
        # print(
        #    f"Failed to run CaBinCorr \n No Suite2P data found: {data_dir}. return None"
        # )
        global_logger.error(
            f"Failed to run CaBinCorr \n No Suite2P data found: {data_dir}. return None"
        )
        return None
    cabincorr_path = os.path.join(data_dir, "binarized_traces.npz")
    if regenerate:
        del_file_dir(cabincorr_path)
        if compute_corrs:
            correlations_path = os.path.join(data_dir, "correlations")
            del_file_dir(correlations_path)
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
    if compute_corrs:
        c.corr_parallel_flag = True
        c.zscore = True
        c.n_tests_zscore = 1000
        c.n_cores = 32
        c.recompute_correlation = regenerate
        c.binning_window = 30  # binning window in frames
        c.subsample = 1  # subsample traces by this factor
        c.scale_by_DFF = True  # scale traces by DFF
        c.shuffle_data = False
        c.subselect_moving_only = False
        c.subselect_quiescent_only = False
        c.make_correlation_dirs()
        c.compute_correlations()
    return c


def run_compute_correlations(c, parallel=True, min_number_bursts=0):
    c.corr_parallel_flag = parallel
    c.zscore = True
    c.n_tests_zscore = 1000
    c.n_cores = 32
    c.recompute_correlation = False
    c.binning_window = 30  # binning window in frames
    c.subsample = 1  # subsample traces by this factor
    c.scale_by_DFF = True  # scale traces by DFF
    c.shuffle_data = False
    c.subselect_moving_only = False
    c.subselect_quiescent_only = False
    c.make_correlation_dirs()
    c.compute_correlations(min_number_bursts=min_number_bursts)


def delete_bin_tiff_s2p_intermediate(
    session, binary=True, tiff=True, intermediate_s2p=False
):
    # Delete binaries
    del_tiff = True
    for s2p_folder in session.suite2p_dirs:
        binary_path = os.path.join(s2p_folder, "plane0", "data.bin")
        binary_backup_path = os.path.join(s2p_folder, "plane0", "backup", "data.bin")

        s2p_folder_ending = s2p_folder.split("suite2p")[-1]
        iscell_path = os.path.join(s2p_folder, "plane0", Session.iscell_fname)
        iscell_count = -1
        if os.path.exists(iscell_path):
            iscell = np.load(iscell_path)[:, 0]
            iscell_count = sum(iscell)

        notgel_path = os.path.join(s2p_folder, "plane0", Session.cell_geldrying_fname)
        notgel_count = -1
        if os.path.exists(notgel_path):
            notgel = np.load(notgel_path) == 0
            notgel_count = sum(notgel)
        if s2p_folder_ending == "" or iscell_count != -1 and notgel_count != -1:
            if binary:
                del_file_dir(binary_path)
                del_file_dir(binary_backup_path)
        else:
            del_tiff = False

    # Delete Tiffs
    if del_tiff and tiff and session.tiff_data_paths:
        for tiff_path in session.tiff_data_paths:
            del_file_dir(tiff_path)

    # delete not needed suite2p MUnits
    if del_tiff and intermediate_s2p:
        keep_endings = ["", "_merged"]
        for s2p_path in session.suite2p_dirs:
            s2p_path_ending = s2p_path.split("suite2p")[-1]
            if s2p_path_ending not in keep_endings:
                del_file_dir(s2p_path)


def fcp(contours, colors):
    """
    fast contour plots
    """
    plt.figure(figsize=[20, 20])
    viz2 = Vizualizer(save_dir="")
    viz2.multi_contours(multi_contours=contours, colors=colors, plot_center=True)
    plt.show()
