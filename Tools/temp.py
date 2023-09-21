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

def load_all(root_dir, wanted_animal_ids=["all"], wanted_session_ids=["all"], print_loading=True):
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
                    animal.get_session_data(session, print_loading=print_loading)
            animals_dict[animal_id] = animal
    return animals_dict

def run_cabin_corr(root_dir, data_dir, animal_id, session_id, parallel=True):
    #Init
    print(f"Loading cabincorr data from {data_dir}")
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
    
    def get_session_data(self, session_id, print_loading=True):
        yaml_file_index = self.session_names.index(session_id)
        session_date = None if len(self.session_dates) != len(self.session_names) else self.session_dates[yaml_file_index]
        pday = None if len(self.pdays) != len(self.session_names) else self.pdays[yaml_file_index]
        session = Session(self.animal_id, session_id, pday=pday, 
                        session_date=session_date, 
                        human_shift=self.session_shifts[yaml_file_index],
                        print_loading=print_loading)
        self.sessions[session_id] = session

class Session:
    corr_fname = "allcell_clean_corr_pval_zscore.npy"

    def __init__(self, animal_id, session_id, pday, session_date, 
                 human_shift, image_x_size=512, image_y_size=512, print_loading=True):
        if print_loading:
            print(f"Loading session: {animal_id} {session_id}")
        self.animal_id = animal_id
        self.session_id = session_id
        self.session_date = session_date
        self.human_shift = human_shift
        self.pday = pday
        self.session_dir = os.path.join(Animal.root_dir, animal_id, session_id)
        self.suite2p_path = os.path.join(self.session_dir, "plane0")
        self.ops = self.set_ops()
        self.image_x_size, self.image_y_size = image_x_size, image_y_size
        self.refImg = None
        self.yx_shift = [0, 0] if session_id == "day0" else None
        self.c, self.contours, self.footprints = self.get_c_contours_footprints()

    def set_ops(self, ops=None):
        if not ops:
            ops = register.default_ops()
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

    def get_c_contours_footprints(self):
        #Merging cell footprints
        c = run_cabin_corr(Animal.root_dir, data_dir=os.path.join(self.suite2p_path),
                            animal_id=self.animal_id, session_id=self.session_id)
        contours = c.contours
        footprints = c.footprints
        return c, contours, footprints
    
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

class Binary_loader:
    """
    A class for loading binary data and converting it into an animation.

    This class provides methods for loading binary data from a file and converting a sequence of binary frames into an animated GIF. The `load_binary` method loads binary data from a specified file and returns it as a NumPy array. The `binary_frames_to_animation` method takes a sequence of binary frames and converts them into an animated GIF, which is saved to the specified directory.

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
    
    def binary_frames_to_animation(self, frames, frame_range=[0, -1], save_dir="animation", comment=""):
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
        if comment != "":
            comment += "_"
        gif_save_path = os.path.join(save_dir, f"{comment}{range_start}-{range_end}.gif")

        images = []
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, frame in enumerate(frames):
            if i%1000 == 0:
                print(i)
            p1 = ax.text(512/2-50, 0, f"Frame {i}", animated=True)
            p2 = ax.imshow(frame, animated=True)
            images.append([p1, p2])
        ani = animation.ArtistAnimation(fig, images, interval=50, blit=True,
                                        repeat_delay=1000)
        ani.save(gif_save_path)
        return ani
    
class Merger:
    def set_yx_shifts(self, reference_session: Session, sessions: dict, n_frames_to_be_acquired=1000, num_align_frames=1000):
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