# work with files 
import yaml
import copy

# do Math stuff
import numpy as np

import sys
import os

# Suite2p for TIFF file analysis
import suite2p
from suite2p.run_s2p import run_s2p, default_ops
from suite2p.registration import register

module_path = os.path.abspath(os.path.join('../'))
sys.path.append(module_path)
from Helper import *
from manifolds.donlabtools.utils.calcium import calcium
from manifolds.donlabtools.utils.calcium.calcium import *

#helper
def gif_to_mp4(path):
    """
    Converts a GIF file to an MP4 file.

    This function takes the path of a GIF file as input, converts it to an MP4 file, and saves the resulting MP4 file in the same directory as the input GIF file. The name of the output MP4 file is the same as the input GIF file, with the file extension changed from `.gif` to `.mp4`.

    Args:
        path (str): The path of the input GIF file.

    Returns:
        None
    """
    import moviepy.editor as mp
    clip = mp.VideoFileClip(path)
    save_path = path.replace('.gif', '.mp4')
    clip.write_videofile(save_path)

def search_file(directory, filename):
    """
    This function searches for a file with a given filename within a specified directory and its subdirectories.

    :param directory: The directory in which to search for the file.
    :type directory: str
    :param filename: The name of the file to search for.
    :type filename: str
    :return: The full path of the file if found, otherwise returns the string "Not found".
    :rtype: str
    """
    for root, dirs, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None

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
    ignore_folders = ["figures", "merged"]
    directories = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name)) and name not in ignore_folders]
    return directories

def load_all(root_dir, wanted_animal_ids=["all"], wanted_session_ids=["all"], restore=False, print_loading=True):
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
    present_animal_ids = get_directories(root_dir)
    animals_dict = {}

    # Search for animal_ids
    for animal_id in present_animal_ids:
        if animal_id in wanted_animal_ids or "all" in wanted_animal_ids:
            sessions_path = os.path.join(root_dir, animal_id)
            present_sessions = get_directories(sessions_path)
            yaml_file_name = os.path.join(root_dir, animal_id, f"{animal_id}.yaml")
            animal = Animal(yaml_file_name, print_loading=print_loading)
            Animal.root_dir = root_dir
            # Search for 2P Sessions
            for session in present_sessions:
                if session in wanted_session_ids or "all" in wanted_session_ids:
                    animal.get_session_data(session, image_x_size=512, 
                                            image_y_size=512, 
                                            restore=restore,
                                            print_loading=print_loading)
            animals_dict[animal_id] = animal
    return animals_dict

def run_cabin_corr(root_dir, data_dir, animal_id, session_id, regenerate_cabincorr=False, parallel=True):
    #Init
    print(f"Getting cabincorr data from {data_dir}")
    cabincorr_path = os.path.join(data_dir, "binarized_traces.npz")
    if regenerate_cabincorr:
        if os.path.exists(cabincorr_path):
            os.remove(cabincorr_path)
    c = calcium.Calcium(root_dir, animal_id, session_name=session_id, data_dir=data_dir)

    #c.parallel_flag = parallel
    c.animal_id = animal_id 
    c.detrend_model_order = 1
    c.recompute_binarization = False
    c.remove_ends = False
    c.detrend_filter_threshold = 0.001
    c.mode_window = 30*30
    c.percentile_threshold = 0.000001
    c.dff_min = 0.02
    c.data_type = "2p"
    #
    c.load_suite2p()

    c.load_binarization()

    # getting contours and footprints
    c.load_footprints()
    return c

def backup_path_files(data_path, backup_folder_name="backup", 
                      redo_backup=False, restore=False):
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
                if os.path.exists(backup_path):
                    shutil.rmtree(backup_path)
                shutil.copytree(data_path, backup_path)
            else:
                print("Backup path already exists. Skipping")

def update_s2p_files(data_path, stat):
    # Read in existing data from a suite2p run. We will use the "ops" and registered binary.
    suite2_data_path = os.path.join(data_path, "plane0")
    binary_file_path = os.path.join(data_path, "data.bin")
    binary_file_path = search_file(data_path, "Image_001_001.raw")
    if not binary_file_path:
        binary_file_path = search_file(data_path, "data.bin")
    if not binary_file_path:
        print("No binary file found. Canceling Suite2P file update.")
        return None

    binary_file_path if os.path.exists(binary_file_path) else os.path.join(data_path, "Image_001_001.raw")
    
    ops = np.load(os.path.join(suite2_data_path, "ops.npy"), allow_pickle=True).item()
    Lx = ops['Lx']
    Ly = ops['Ly']
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
    stat_after_extraction, F, Fneu, F_chan2, Fneu_chan2 = suite2p.extraction_wrapper(stat, f_reg, f_reg_chan2 = None, ops=ops)
    # Do cell classification
    classfile = suite2p.classification.builtin_classfile
    iscell = suite2p.classify(stat=stat_after_extraction, classfile=classfile)
    # Apply preprocessing step for deconvolution
    dF = F.copy() - ops['neucoeff']*Fneu
    dF = suite2p.extraction.preprocess(
            F=dF,
            baseline=ops['baseline'],
            win_baseline=ops['win_baseline'],
            sig_baseline=ops['sig_baseline'],
            fs=ops['fs'],
            prctile_baseline=ops['prctile_baseline']
        )
    # Identify spikes
    spks = suite2p.extraction.oasis(F=dF, batch_size=ops['batch_size'], tau=ops['tau'], fs=ops['fs'])

    # backing up original suite2p files first
    backup_path_files(suite2_data_path) 

    old_files = ["binarized_traces.mat", "binarized_traces.npz", "Fall.mat"]
    old_folders = ["correlations", "figures"]
    for old_folder in old_folders:
        fpath = os.path.join(suite2_data_path, old_folder)
        if os.path.exists(fpath):
            shutil.rmtree(fpath)
    for old_file in old_files:
        fpath = os.path.join(suite2_data_path, old_file)
        if os.path.exists(fpath):
            os.remove(fpath)

    np.save(os.path.join(suite2_data_path, 'F.npy'), F)
    np.save(os.path.join(suite2_data_path, 'Fneu.npy'), Fneu)
    np.save(os.path.join(suite2_data_path, 'iscell.npy'), iscell)
    np.save(os.path.join(suite2_data_path, 'ops.npy'), ops)
    np.save(os.path.join(suite2_data_path, 'spks.npy'), spks)
    np.save(os.path.join(suite2_data_path, 'stat.npy'), stat)


#classes
class Animal:
    """
    This class represents an animal in an experiment.

    Attributes:
    root_dir (str): The root directory where the data is stored.
    sessions (dict): A dictionary to store session objects for this animal.
    cohort_year (int): The year of the cohort that the animal belongs to.
    dob (str): The date of birth of the animal.
    animal_id (str): The ID of the animal.
    session_dates (list of str): The dates when the sessions were conducted.
    session_names (list of str): The names of the sessions.
    sex (str): The sex of the animal.

    Methods:
    load_data(yaml_path): Loads metadata for the animal from a YAML file.
    get_session_data(session_id, print_loading=True): Loads data for a specific session.
    """
    root_dir = "D:\\Animals" 

    def __init__(self, yaml_file_path, print_loading=True) -> None:
        self.sessions = {}
        self.cohort_year = None
        self.dob = None
        self.animal_id = None 
        self.pdays = None 
        self.session_dates = None 
        self.session_names = None 
        self.sex = None 
        self.session_shifts = None
        self.load_data(yaml_file_path)
        self.animal_dir = os.path.join(Animal.root_dir, self.animal_id)
        if print_loading:
            print(f"Added animal: {self.animal_id}")

    def load_data(self, yaml_path):
        with open(yaml_path, "r") as yaml_file:
            animal_metadata_dict = yaml.safe_load(yaml_file)
        cohort_year = animal_metadata_dict["cohort_year"]
        self.cohort_year = int(cohort_year) if type(cohort_year)==str else int(cohort_year[0])
        self.dob = animal_metadata_dict["dob"]
        self.animal_id = animal_metadata_dict["name"]
        self.pdays = animal_metadata_dict["pdays"]
        self.session_dates = animal_metadata_dict["session_dates"]
        session_name_list = animal_metadata_dict["session_names"]
        self.session_names = [str(session_name) if type(session_name)!=str else session_name for session_name in session_name_list]
        self.sex = animal_metadata_dict["sex"]
        self.session_shifts = animal_metadata_dict["shifts"]
    
    def get_session_data(self, session_id, 
                         image_x_size=512, image_y_size=512, 
                         reference_session_id="day0",
                         restore=False,
                         print_loading=True):
        if session_id != "merged":
            yaml_file_index = self.session_names.index(session_id)
            session_date = None if len(self.session_dates) != len(self.session_names) else self.session_dates[yaml_file_index]
            pday = None if len(self.pdays) != len(self.session_names) else self.pdays[yaml_file_index]
            session = Session(self.animal_id, session_id, pday=pday, 
                            session_date=session_date, 
                            human_shift=self.session_shifts[yaml_file_index],
                            image_x_size=image_x_size, image_y_size=image_y_size,
                            restore=restore,
                            print_loading=print_loading)
        else:
            reference_image_x_size = self.sessions[reference_session_id].image_x_size
            reference_image_y_size = self.sessions[reference_session_id].image_y_size
            session = Session(animal_id=self.animal_id, session_id=session_id,
                      pday=None, session_date=None, human_shift=None,
                      image_x_size=reference_image_x_size, 
                      image_y_size=reference_image_y_size,
                      print_loading=True)
        self.sessions[session_id] = session

    def merge_sessions(self, reference_session_id="day0", regenerate=False, n_frames_to_be_acquired=1000, num_align_frames=1000):
        reference_session = self.sessions[reference_session_id]
        sessions = self.sessions

        # check if already all, merged, suite2p files are present
        suite2p_files_list = []
        merged_s2p_path = os.path.join(self.animal_dir, "merged", "plane0")
        for s2p_file in suite2p_files_list:
            s2p_file_path = search_file(merged_s2p_path, s2p_file)
            if not s2p_file_path:
                merged_s2p_path = None
                break 

        merger = Merger()
        print(f"Generating yx-shifts based on reference session {reference_session_id}")
        merger.set_yx_shifts(reference_session, sessions, n_frames_to_be_acquired, num_align_frames)

        # merge masks, updated sessions based on merged masks
        if regenerate or not os.path.exists(merged_s2p_path):
            # reload original session files
            for session_id, session in sessions.items():
                backup_path_files(session.suite2p_path, restore=True) 
                backup_path_files(session.suite2p_path, restore=False)
                session.c, session.contours, session.footprints = session.get_c_contours_footprints()
            # create a master mask by
            # merging masks of every session, remove abroad cells and deduplicate
            print("Creating master mask...")
            merged_stat = merger.merge_stat(sessions, reference_session, parallel = True)
            print(f"Number of cells after merging: {merged_stat.shape[0]}")

            # Update all sessions based on merged mask
            updated_sessions = {} 
            for session_id, session in sessions.items():
                # shift merged mask and redo Suite2P analysis
                print(f"Updating session {session_id}")
                merger.shift_update_session_s2p_files(session, merged_stat)
                updated_sessions[session_id] = Session(session.animal_id, session.session_id, 
                                                    session.pday, session.image_x_size,
                                                    session.image_y_size, regenerate_cabincorr=True,
                                                    print_loading=True)
            self.sessions = updated_sessions
            merger.merge_s2p_files(updated_sessions, merged_stat)

        # create Session entrie in dictionary
        reference_image_x_size = reference_session.image_x_size
        reference_image_y_size = reference_session.image_y_size
        self.get_session_data(session_id="merged", image_x_size=reference_image_x_size,
                            image_y_size=reference_image_y_size, print_loading=True)
        #get_geldrying_cells()
        return self.sessions["merged"]

class Session:
    corr_fname = "allcell_clean_corr_pval_zscore.npy"
    cabincorr_fname = "binarized_traces.npz"

    def __init__(self, animal_id, session_id, pday, session_date, 
                 human_shift, image_x_size=512, image_y_size=512, 
                 restore=False,
                 regenerate_cabincorr=False, print_loading=True):
        if print_loading:
            print(f"Loading session: {animal_id} {session_id}")
        self.animal_id = animal_id
        self.session_id = session_id # = session_name
        self.session_date = session_date
        self.human_shift = human_shift
        self.pday = pday
        self.session_dir = os.path.join(Animal.root_dir, animal_id, session_id)
        self.suite2p_path = os.path.join(self.session_dir, "plane0")
        if restore:
            backup_path_files(self.suite2p_path, restore=restore)
        self.ops = self.set_ops()
        self.image_x_size, self.image_y_size = image_x_size, image_y_size
        self.refImg = None
        self.yx_shift = [0, 0] if "day0" in session_id else None
        self.c, self.contours, self.footprints = self.get_c_contours_footprints(regenerate_cabincorr=regenerate_cabincorr)
        self.bin_traces_zip = None

    def set_ops(self, ops=None):
        if not ops:
            ops_path = os.path.join(self.suite2p_path, "ops.npy")
            if os.path.exists(ops_path):
                ops = np.load(ops_path, allow_pickle=True)
            else:
                ops = register.default_ops()
            try:
                ops["nonrigid"] = False
            except:
                ops = np.load(ops_path, allow_pickle=True).item()
                ops["nonrigid"] = False
        else:
            self.ops = ops
        return ops

    def get_reference_image(self, n_frames_to_be_acquired=1000):
        """
        This function gets the reference image. If the reference image is not already set, 
        it loads a binary file and computes the reference image.

        Parameters:
        n_frames_to_be_acquired (int): The number of frames to be acquired. Default is 1000.

        Returns:
        numpy.ndarray: The reference image.
        """
        if self.refImg is None:
            b_loader = Binary_loader()
            frames = b_loader.load_binary(self.session_dir, n_frames_to_be_acquired=n_frames_to_be_acquired, 
                                          image_x_size=self.image_x_size, image_y_size=self.image_y_size)
            self.refImg = register.compute_reference(frames, ops=self.ops)
        return self.refImg

    def set_yx_shift(self, refAndMasks, num_align_frames=1000, yx_shift=None):
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
            b_loader = Binary_loader()
            frames = b_loader.load_binary(self.session_dir, n_frames_to_be_acquired=num_align_frames, image_x_size=self.image_x_size, image_y_size=self.image_y_size)
            frames, ymax, xmax, cmax, ymax1, xmax1, cmax1, _ = register.register_frames(refAndMasks, frames, ops=self.ops)
            self.yx_shift = [round(np.mean(ymax)), round(np.mean(xmax))]
        return self.yx_shift

    def get_c_contours_footprints(self, regenerate_cabincorr=False):
        #Merging cell footprints
        c = run_cabin_corr(Animal.root_dir, data_dir=self.suite2p_path,
                            animal_id=self.animal_id, session_id=self.session_id, 
                            regenerate_cabincorr=regenerate_cabincorr)
        contours = c.contours
        footprints = c.footprints
        return c, contours, footprints
    
    def load_cabincorr_data(self):
        if type(self.bin_traces_zip) != np.ndarray: 
            path = os.path.join(self.suite2p_path, Session.cabincorr_fname)
            if os.path.exists(path):
                self.bin_traces_zip = np.load(path, allow_pickle=True)
            else:
                print("No CaBincorrPath found")
        return self.bin_traces_zip
    
class Vizualizer:
    def __init__(self, animals={}, save_dir=Animal.root_dir):
        self.animals = animals
        self.save_dir = os.path.join(save_dir, "figures")
        dir_exist_create(self.save_dir)
        # Collor pallet for plotting
        self.max_color_number = 301
        self.colors = mlp.colormaps["rainbow"](range(0, self.max_color_number))

    def session_footprints(self, session, figsize=(10,10), cmap=None):
        # plot footprints of a session
        plt.figure(figsize=figsize)
        title = f"{session.animal_id}_{session.session_id}"
        footprints = session.footprints
        plt.title(f"{len(footprints)} footprints {title}")
        self.footprints(footprints, cmap=cmap)
        plt.savefig(os.path.join(self.save_dir, f"Footprints_{title}.png"), dpi=300)

    def footprints(self, footprints, cmap=None):
        # plot all footprints
        for footprint in footprints:
            idx = np.where(footprint==0)
            footprint[idx] = np.nan
            plt.imshow(footprint, cmap=cmap)
        plt.gca().invert_yaxis()

    def session_contours(self, session, figsize=(10,10), color=None, plot_center=False, comment=""):
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

    def contours(self, contours, color=None, plot_center=False, comment=""): #plot_contours_points
        for contour in contours:
            y_corr = contour[:, 0]
            x_corr = contour[:, 1]
            plt.plot(x_corr, y_corr, color = color)
            if plot_center:
                xy_mean = self.contour_to_point(contour)
                plt.plot(xy_mean[1], xy_mean[0], ".", color = color)
        plt.title(f"{len(contours)} Contours{comment}")

    def multi_contours(self, multi_contours, plot_center=False, colors=["white", "red", "green", "blue", "yellow", "purple", "orange", "cyan", "pink"]):
        for contours, col in zip(multi_contours, colors):
            self.contours(contours, color=col, plot_center=plot_center)

    def multi_session_contours(self, sessions, combination=None, plot_center=False, shift=False, figsize=(20, 20), comment=""):
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
        combination = list(sessions.keys()) if combination==None else combination
        colors = ["white", "red", "green", "blue", "yellow", "purple", "orange", "cyan", "pink"]
        for (session_id, session), col in zip(sessions.items(), colors):
            if session_id not in combination:
                continue
            #shift contours
            plot_colors.append(col)
            if shift:
                shift_amount = session.yx_shift if shift_type=="algo" else session.human_shift
            else:
                shift_amount = [0, 0]
            contours = [contour - shift_amount for contour in session.contours] if shift else session.contours
            plot_contours.append(contours)
            shift_label = f" y: {shift_amount[0]}  x: {shift_amount[1]}" if shift else ""
            handles.append(Line2D([0], [0], color=col, linewidth=2, linestyle='-', label=f"Msession: {session_id}{shift_label}"))
        self.multi_contours(plot_contours, colors=plot_colors, plot_center=plot_center)
        plt.title(f"Contours for : {combination} {comment}")
        plt.legend(handles=handles, fontsize=20)
        shift_label = f"_shifted" if shift else ""
        plt.savefig(os.path.join(self.save_dir, f"Contours_{combination}{shift_label}{comment}.png"), dpi=300)
        #plt.show()
        plt.close()

    def multi_session_refImg(self, sessions, num_images_x=2):
        num_sessions = len(sessions)
        num_rows = round(num_sessions/num_images_x)
        fig, ax = plt.subplots(num_rows, num_images_x, figsize =(5*num_images_x, 5*num_rows))
        for i, (session_id, session) in enumerate(sessions.items()):
            title = f"Reference Images of {session.animal_id}"
            x = int(i/num_images_x)
            y = i%num_images_x
            if len(ax.shape) == 2:
                ax[x, y].imshow(session.refImg)
                ax[x, y].invert_yaxis()
                ax[x, y].set_title(f'{session_id}')
            else:
                ax[i].imshow(session.refImg)
                ax[i].invert_yaxis()
                ax[i].set_title(f'{session_id}')
        fig.suptitle(title, fontsize=20)
        plt.savefig(os.path.join(self.save_dir, title.replace(" ", "_")+".png"), dpi=300)
        plt.show()

    def bursts(self, animal_id, session_id, fluorescence_type="F_raw", num_cells="all", dpi=300, fps="30"):

        #TODO: insert possibility to filter for good cells?
        #is_cells_ids = np.where(calcium_object.iscell==1)[0]
        #is_not_cells_ids = np.where(calcium_object.iscell==0)[0]
        #num_is_cells = is_cells_ids.shape[0] #get is cells
        #calcium_object.plot_traces(calcium_object.F_filtered, np.arange(num_is_cells))

        session = self.animals[animal_id].sessions[session_id]
        bin_traces_zip = session.load_cabincorr_data()
        fluorescence = None
        if bin_traces_zip:
            if fluorescence_type in list(bin_traces_zip.keys()):
                fluorescence = bin_traces_zip[fluorescence_type]
            else:
                print(f"{animal_id} {session_id} No fluorescence data of type {fluorescence_type} in binarized_traces.npz")
        else:
            print(f"{animal_id} {session_id} no binarized_traces.npz found")
        
        if type(fluorescence)==np.ndarray:
            self.traces(fluorescence, animal_id, session_id, num_cells, fluorescence_type=fluorescence_type, dpi=dpi)
        return fluorescence

    def traces(self, fluorescence, animal_id, session_id, num_cells="all", fluorescence_type="", low_pass_filter=True, dpi=300):
        # plot fluorescence
        if low_pass_filter:
            fluorescence = butter_lowpass_filter(fluorescence, cutoff=0.5, fs=30, order=2)
        
        fluorescence = np.array(fluorescence)
        fluorescence = np.transpose(fluorescence) if len(fluorescence.shape)==2 else fluorescence
        plt.figure()
        plt.figure(figsize=(12, 7))
        if num_cells != "all":
            plt.plot(fluorescence[:, :int(num_cells)])
        else:
            plt.plot(fluorescence)

        file_name = f"{animal_id} {session_id}"
        seconds = 5
        fps=30
        num_frames = fps*seconds
        num_x_ticks = 50
        written_label_steps = 2

        x_time = [int(frame/num_frames)*seconds for frame in range(len(fluorescence)) if frame%num_frames==0] 
        steps = round(len(x_time)/(2*num_x_ticks))
        x_time_shortened = x_time[::steps]
        x_pos = np.arange(0, len(fluorescence), num_frames)[::steps] 
        
        title = f"Bursts from {file_name} {fluorescence_type}"
        xlabel=f"seconds"
        ylabel='fluorescence based on Ca in Cell'
        x_labels = [time if num%written_label_steps==0 else "" for num, time in enumerate(x_time_shortened)]
        plt.xticks(x_pos, x_labels, rotation=40, fontsize=8)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.savefig(os.path.join(self.save_dir, title.replace(" ", "_")+".png"), dpi=dpi)
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
    def load_binary(self, data_path, n_frames_to_be_acquired, fname="Image_001_001.raw", image_x_size=512, image_y_size=512):
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
        image_size=image_x_size*image_y_size
        fpath = search_file(data_path, fname)
        binary = np.memmap(fpath,
                            dtype='uint16',
                            mode='r',
                            shape=(n_frames_to_be_acquired, image_x_size, image_y_size))
        binary_frames = copy.deepcopy(binary)
        return binary_frames
    
    def binary_frames_to_gif(self, frames, frame_range=[0, -1], fps=30, save_dir="animation", comment=""):
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
        comment = comment+"_" if comment != "" else comment
        save_dir = os.path.join(save_dir, "animation")
        gif_save_path = os.path.join(save_dir, f"{comment}{range_start}-{range_end}.gif")

        delay_between_frames = int(1000/fps)# ms
        images = []
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, frame in enumerate(frames):
            if i%1000 == 0:
                print(i)
            p1 = ax.text(512/2-50, 0, f"Frame {i}", animated=True)
            p2 = ax.imshow(frame, animated=True)
            images.append([p1, p2])
        ani = animation.ArtistAnimation(fig, images, interval=delay_between_frames, blit=True,
                                        repeat_delay=1000)
        ani.save(gif_save_path)
        return ani
    
class Merger:

    def set_yx_shifts(self, reference_session: Session, sessions: dict, 
                      n_frames_to_be_acquired=1000, num_align_frames=1000):
        """
        This function calculates the yx_shifts for all sessions relative to a reference session. 
        It first gets the reference image from the reference session and computes the reference masks. 
        Then it sets the yx_shift for each session in the sessions dictionary.

        Parameters:
        reference_session (Session): The session to be used as reference.
        sessions (dict): A dictionary of sessions keyed by session_id.
        n_frames_to_be_acquired (int): The number of frames to be acquired. Default is 1000.

        Returns:
        None
        """
        refImg = reference_session.get_reference_image(n_frames_to_be_acquired = n_frames_to_be_acquired)
        refAndMasks = register.compute_reference_masks(refImg, reference_session.ops)
        for session_id, session in sessions.items():
            if session_id == reference_session.session_id:
                continue   
            session.set_yx_shift(refAndMasks, num_align_frames=num_align_frames)
        return sessions
    
    def shift_stat_cells(self, stat: list, yx_shift: list, image_x_size=512, image_y_size=512):
        """
        Shifts the cells in the stat array based on the given yx_shift.

        Parameters:
        stat (list): List of cell statistics.
        yx_shift (list): List containing y and x shifts.
        image_x_size (int): Size of the image in x direction. Default is 512.
        image_y_size (int): Size of the image in y direction. Default is 512.

        Returns:
        new_stat (list): List of shifted cell statistics.
        """
        # stat files first value ist y-value second is x-value
        new_stat = copy.deepcopy(stat)

        for num, cell_stat in enumerate(new_stat):
            y_shifted = []
            for y in cell_stat["ypix"]:
                y_shifted.append(y-yx_shift[0])
            cell_stat["ypix"] = np.array(y_shifted)
            
            x_shifted = []
            for x in cell_stat["xpix"]:
                x_shifted.append(x-yx_shift[1])
            cell_stat["xpix"] = np.array(x_shifted)

            #center of cell_stat
            med = cell_stat["med"]
            med_shifted = [med[0]-yx_shift[0], med[1]-yx_shift[1]]
            cell_stat["med"] = med_shifted
        return new_stat

    def remove_abroad_cells(self, stat: list, sessions: dict, image_x_size=512, image_y_size=512):
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
        # removing out of bound cells 
        remove_cells = []
        for cell_num, cell in enumerate(stat):
            abroad = False
            #check for every shift 
            for session_id, session in sessions.items():
                if abroad:
                    break
                yx_shift = session.yx_shift
                for axis in ["ypix", "xpix"]:
                    shift = yx_shift[0] if axis=="ypix" else yx_shift[1]
                    shifted = cell[axis]+shift

                    # check if cell is out of bound
                    max_location = image_y_size if axis=="ypix" else image_x_size
                    if sum(shifted>=max_location)>0 or sum(shifted<0)>0:
                        abroad = True
                        break    
            if abroad:
                remove_cells.append(cell_num)
                
        for abroad_cell in remove_cells[::-1]:
            stat = np.delete(stat, abroad_cell)
        if len(remove_cells)>0:
            print(f"removed abroad cells: {remove_cells}")
        return stat

    def stat_to_footprints(self, stat: list, dims=[512, 512]):
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
            x = stat[k]['xpix']
            y = stat[k]['ypix']

            # save footprint
            img_temp = np.zeros((dims[0], dims[1]))
            img_temp[x, y] = stat[k]['lam']

            img_temp_norm = (img_temp - np.min(img_temp)) / (np.max(img_temp) - np.min(img_temp))
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
        
        shifted_session_stat_no_abroad = self.remove_abroad_cells(reference_session.c.stat, sessions, image_x_size=image_x_size, image_y_size=image_y_size)
        merged_footprints = self.stat_to_footprints(shifted_session_stat_no_abroad)
        merged_stat = shifted_session_stat_no_abroad
        for session_id, session in sessions.items():
            if session_id == reference_session.session_id:
                continue    
            shifted_session_stat = self.shift_stat_cells(session.c.stat, yx_shift=session.yx_shift, image_x_size=image_x_size, image_y_size=image_y_size)
            shifted_session_stat_no_abroad = self.remove_abroad_cells(shifted_session_stat, sessions, image_x_size=image_x_size, image_y_size=image_y_size)
            shifted_footprints = self.stat_to_footprints(shifted_session_stat_no_abroad)
            clean_cell_ids, merged_footprints = self.merge_deduplicate_footprints(merged_footprints, shifted_footprints, parallel=parallel, num_batches=num_batches)
            merged_stat = np.concatenate([merged_stat, shifted_session_stat_no_abroad])[clean_cell_ids]
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

    def generate_batch_cell_overlaps(self, footprints, parallel=True, recompute_overlap=False, 
                                     n_cores=16, num_batches=3):
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
        print ("... computing cell overlaps ...")
        
        num_footprints = footprints.shape[0]
        num_min_cells_per_process = 10
        num_parallel_processes = 30 if num_footprints/30>num_min_cells_per_process else int(num_footprints/num_min_cells_per_process)
        ids = np.array_split(np.arange(num_footprints, dtype="int64"), num_parallel_processes)

        if num_batches > num_parallel_processes:
            num_batches = num_parallel_processes

        #TODO: will results in an error, if np.array_split is used on inhomogeneouse data like ids on Scicore
        batches = np.array_split(ids, num_batches) if num_batches!=1 else [ids]
        results = np.array([])
        num_cells = 0
        for batch in batches:
            res = parmap.map(find_overlaps1,
                            batch,
                            footprints,
                            #c.footprints_bin,
                            pm_processes=16,
                            pm_pbar=True,
                            pm_parallel=parallel)
            for cell_batch in res:
                num_cells += len(cell_batch)
                for cell in cell_batch:
                    results = np.append(results, cell)
        results = results.reshape(num_cells, 5)
        res = [results]
        df = make_overlap_database(res)
        return df

    def find_candidate_neurons_overlaps(self, df_overlaps: pd.DataFrame, 
                                        corr_array=None, deduplication_use_correlations=False, 
                                        corr_max_percent_overlap=0.25, corr_threshold=0.3):
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
            cell1 = int(row['cell1'])
            cell2 = int(row['cell2'])
            percent1 = row['percent_cell1']
            percent2 = row['percent_cell2']

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
            idx2 = np.where(dist_corr_matrix[idx1, 2] >= corr_threshold)[0]   # note these are zscore thresholds for zscore method
            idx3 = idx1[idx2]
        else:
            idx3 = idx1
        #
        candidate_neurons = dist_corr_matrix[idx3][:, :2]
        return candidate_neurons

    def make_correlated_neuron_graph(self, num_cells: int, candidate_neurons: np.ndarray):
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

    def delete_duplicate_cells(self, num_cells: int, G, corr_delete_method='highest_connected_no_corr'):
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
        if corr_delete_method=='highest_connected_no_corr':
            connected_cells, removed_cells = del_highest_connected_nodes_without_corr(G)
        # 
        print ("Removed duplicated cells: ", len(removed_cells))
        clean_cells = np.delete(np.arange(num_cells),
                                removed_cells)

        #
        clean_cell_ids = clean_cells
        removed_cell_ids = removed_cells
        connected_cell_ids = connected_cells
        return clean_cell_ids

    def merge_deduplicate_footprints(self, footprints1: np.ndarray, footprints2: np.ndarray,
                                      parallel=True, num_batches=4):
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

        df_overlaps = self.generate_batch_cell_overlaps(merged_footprints, recompute_overlap=True, parallel=parallel, num_batches=num_batches)
        candidate_neurons = self.find_candidate_neurons_overlaps(df_overlaps, corr_array=None, deduplication_use_correlations=False, corr_max_percent_overlap=0.25, corr_threshold=0.3)
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
        image_x_size = session.image_x_size
        image_y_size = session.image_y_size
        data_path = os.path.join(session.session_dir)
        suite2p_data_path = session.suite2p_path
        # shift merged mask
        shift_to_session = np.array([-1]) * session.yx_shift
        shifted_session_stat = self.shift_stat_cells(new_stat, yx_shift=shift_to_session, image_x_size=image_x_size, image_y_size=image_y_size)

        backup_path_files(suite2p_data_path)
        update_s2p_files(data_path, shifted_session_stat)

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
        merged_Fneu = np.load(os.path.join(path,   "Fneu.npy"))
        merged_spks = np.load(os.path.join(path,   "spks.npy"))
        merged_iscell = np.load(os.path.join(path, "iscell.npy"))
        for session_id, session in sessions.items():
            if session_id == first_session_object.session_id:
                continue
            path = session.suite2p_path
            F =  np.load(os.path.join(path, "F.npy"))
            merged_F = np.concatenate([merged_F, F], axis=1)
            Fneu =  np.load(os.path.join(path, "Fneu.npy"))
            merged_Fneu = np.concatenate([merged_Fneu, Fneu], axis=1)
            spks =  np.load(os.path.join(path, "spks.npy"))
            merged_spks = np.concatenate([merged_spks, spks], axis=1)
            # sum iscells
            is_cell = np.load(os.path.join(path, "iscell.npy"))
            merged_iscell += is_cell
        
        #let cells life if one of the cells is detected as cell. Average probabilities for ifcell
        merged_iscell /= len(list(sessions.keys()))
        merged_iscell[:, 0] = np.ceil(merged_iscell[:, 0])

        animal_folder = os.path.join(Animal.root_dir, session.animal_id)
        merged_s2p_path = os.path.join(animal_folder, "merged")
        dir_exist_create(merged_s2p_path)
        merged_s2p_path = os.path.join(animal_folder, "merged", "plane0")
        dir_exist_create(merged_s2p_path)

        np.save(os.path.join(merged_s2p_path, "F.npy"), merged_F)
        np.save(os.path.join(merged_s2p_path, "Fneu.npy"), merged_Fneu)
        np.save(os.path.join(merged_s2p_path, "spks.npy"), merged_spks)
        np.save(os.path.join(merged_s2p_path, "iscell.npy"), merged_iscell)
        np.save(os.path.join(merged_s2p_path, "stat.npy"), stat)
        np.save(os.path.join(merged_s2p_path, "ops.npy"), ops)
        return merged_s2p_path