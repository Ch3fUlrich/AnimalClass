# Imports
import os
import numpy as np
import yaml
import re
import matplotlib.pyplot as plt
from datetime import datetime


def load_all(
    root_dir, wanted_animal_ids=["all"], wanted_session_ids=["all"], print_loading=True
):
    """
    Loads animal data from the specified root directory for the given animal IDs.

    Parameters:
    - root_dir (string): The root directory path where the animal data is stored.
    - animal_ids (list, optional): A list of animal IDs to load. Default is ["all"].
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
                    animal.get_session_data(session_path, print_loading=print_loading)
            animals_dict[animal_id] = animal
    return animals_dict


# Helper
def get_directories(directory, regex_search=""):
    """
    This function returns a list of directories from the specified directory that match the regular expression search pattern.

    Parameters:
    directory (str): The directory path where to look for directories.
    regex_search (str, optional): The regular expression pattern to match. Default is an empty string, which means all directories are included.

    Returns:
    list: A list of directory names that match the regular expression search pattern.
    """
    directories = None
    if os.path.exists(directory):
        directories = [
            name
            for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))
            and len(re.findall(regex_search, name)) > 0
        ]
    else:
        print(f"Directory does not exist: {directory}")
    return directories


def get_files(directory, ending="", regex_search=""):
    """
    This function returns a list of files from the specified directory that match the regular expression search pattern and have the specified file ending.

    Parameters:
    directory (str): The directory path where to look for files.
    ending (str, optional): The file ending to match. Default is '', which means all file endings are included.
    regex_search (str, optional): The regular expression pattern to match. Default is an empty string, which means all files are included.

    Returns:
    list: A list of file names that match the regular expression search pattern and have the specified file ending.
    """
    files_list = None
    if os.path.exists(directory):
        files_list = [
            name
            for name in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, name))
            and len(re.findall(regex_search, name)) > 0
            and name.endswith(ending)
        ]
    else:
        print(f"Directory does not exist: {directory}")
    return files_list


def sliding_window(arr, window_size, step_size=1):
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


def sliding_mean_std(arr, window_size):
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
    for num, window in enumerate(sliding_window(arr, window_size)):
        mean_stds[num, 0] = np.mean(window)
        mean_stds[num, 1] = np.std(window)
    return np.array(mean_stds)


def xticks_frames_to_seconds(frames, fps=30):
    seconds = 5
    num_frames = fps * seconds
    num_x_ticks = 50
    written_label_steps = 2

    x_time = [
        int(frame / num_frames) * seconds
        for frame in range(frames)
        if frame % num_frames == 0
    ]
    steps = round(len(x_time) / (2 * num_x_ticks))
    x_time_shortened = x_time[::steps]
    x_pos = np.arange(0, frames, num_frames)[::steps]

    x_labels = [
        time if num % written_label_steps == 0 else ""
        for num, time in enumerate(x_time_shortened)
    ]
    plt.xticks(x_pos, x_labels, rotation=40, fontsize=8)


def traces(
    fluorescence,
    animal_id,
    session_id,
    unit_id="all",
    num_cells="all",
    fluorescence_type="",
    is_wheel=True,
    is_wheel_color="lightgray",
    fps=30,
    dpi=300,
):
    # plot fluorescence
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

    color_background(plt, is_wheel, color=is_wheel_color)

    xticks_frames_to_seconds(len(fluorescence), fps=fps)

    file_name = f"{animal_id} {session_id}"
    title = f"Bursts from {file_name} {fluorescence_type}"
    xlabel = f"seconds"
    ylabel = "fluorescence based on Ca in Cell"

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.savefig(os.path.join(self.save_dir, title.replace(" ", "_")+".png"), dpi=dpi)
    plt.show()
    plt.close()


def plot_raster(
    binarized_traces,
    animal_id="",
    session_id="",
    fluorescence_type="",
    fps=30,
):
    bin_traces = binarized_traces.transpose()
    plt.figure(figsize=(15, 5))
    num_time_steps, num_neurons = bin_traces.shape
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

    file_name = f"{animal_id} {session_id}"
    title = f"{int(np.nansum(bin_traces))} Bursts from {file_name} {fluorescence_type}"
    xlabel = f"seconds"
    ylabel = "Neuron ID"

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.savefig(os.path.join(self.save_dir, title.replace(" ", "_")+".png"), dpi=dpi)
    plt.show()
    plt.close()


def plot_velocity(
    velocity, average=False, window_size=30, is_wheel=True, is_wheel_color="lightgray"
):
    velocity_lable = "Velocity"
    plot_velocity = velocity
    if average:
        plot_velocity = sliding_mean_std(velocity, window_size=window_size)[:, 0]
        velocity_lable = "Averaged Velocity"

    plt.figure(figsize=(15, 10))  # Adjust the figure size as needed
    color_background(plt, is_wheel, color=is_wheel_color)

    # Plot the averaged velocity data with labels
    plt.plot(plot_velocity, label=velocity_lable)
    plt.xlabel("Frames")
    plt.ylabel("Velocity")
    plt.title("Velocity Data")
    plt.legend()
    plt.grid(True, color="gray")

    plt.show()


def color_background(plt, is_wheel, color="lightgray"):
    """
    Color the background of a plot based on the value of "is_wheel".

    Parameters:
    plt (matplotlib.pyplot): The matplotlib.pyplot object used for plotting.
    is_wheel (numpy.ndarray): An array containing the wheel data.
    color (str, optional): The color to use for the background. Defaults to "lightgray".

    Returns:
    None
    """
    if isinstance(is_wheel, np.ndarray):
        old_pos = 0
        new_pos = 0
        end_range = len(is_wheel)

        prev_value = is_wheel[old_pos]
        for i, value in enumerate(is_wheel):
            if value != prev_value:
                print(value, i)
                new_pos = i
                facecolor = "white" if prev_value else color
                plt.axvspan(old_pos, new_pos - 1, facecolor=facecolor)
                old_pos = new_pos
                prev_value = value
            if i == end_range - 1:
                facecolor = "white" if value else color
                plt.axvspan(old_pos, end_range - 1, facecolor=facecolor)
    else:
        print("No wheel data available")


def plot_contours(contours=None, centers=None, session=None, clean=False):
    if not contours and not session:
        ValueError("No contours or session provided.")

    plt.figure(figsize=(15, 15))
    title = f"Contours"

    if session:
        if not isinstance(centers, np.ndarray):
            centers = session.get_cell_centers(clean=clean)
        if not isinstance(contours, np.ndarray):
            contours = session.get_cell_contours(clean=clean)
        title = f"Contours for session: {session.animal_id} {session.session_id}"

    for contour in contours:
        plt.plot(contour[:, 0], contour[:, 1])

    if isinstance(centers, np.ndarray):
        plt.scatter(np.array(centers)[:, 0], np.array(centers)[:, 1], s=5, c="black")

    title = f"{len(contours)} " + title
    plt.title(title)
    plt.show()


def np_load_if_exists(fpath, allow_pickle=True):
    return np.load(fpath, allow_pickle=allow_pickle) if os.path.exists(fpath) else None


def num_to_date(date_string):
    if type(date_string) != str:
        date_string = str(date_string)
    date = datetime.strptime(date_string, "%Y%m%d")
    return date


# Classes
class Animal:
    """
    Represents an animal in a research experiment.

    Attributes:
        root_dir (str): The root directory where animal data is stored.
        sessions (dict): A dictionary to store information about sessions associated with the animal.
        cohort_year (int): The year the animal was part of a cohort.
        dob (str): The date of birth of the animal.
        animal_id (str): A unique identifier for the animal.
        sex (str): The gender of the animal.

    Methods:
        __init__(self, yaml_file_path, print_loading=True):
            Initialize an Animal object by loading metadata from a YAML file.

        load_metadata(self, yaml_path):
            Load metadata from a YAML file and set attributes for the Animal object.

        get_session_data(self, path, print_loading=True):
            Get session data associated with the animal from a specified directory.

    Parameters:
        - yaml_file_path (str): The path to the YAML file containing animal metadata.
        - print_loading (bool, optional): If True, print a message when an animal is added.

    Returns:
        None
    """

    root_dir = "undefined"

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
            else int(cohort_year[0]) if type(cohort_year) == list else cohort_year
        )
        self.dob = animal_metadata_dict["dob"]
        for animal_id_key in ["name", "animal_id"]:
            if animal_id_key in animal_metadata_dict.keys():
                self.animal_id = animal_metadata_dict["animal_id"]
        self.sex = animal_metadata_dict["sex"]
        return animal_metadata_dict

    def get_session_data(self, path, print_loading=True):
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
                    print_loading=print_loading,
                )
                # checking if sessin is correct
                if str(session.date) not in session_yaml_fname:
                    print(
                        f"Yaml file naming does not match session date: {session_yaml_fname} != {session.date}"
                    )
                    match = False
                if match:
                    session.pday = (
                        num_to_date(session.date) - num_to_date(self.dob)
                    ).days
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
        return session


class Session:
    """
    A class for managing and loading session metadata and data for an experimental session.

    Attributes:
        animal_id (str): The ID of the animal associated with the session.
        binarized_traces (numpy.ndarray): Binarized traces data.
        cabincorr_fname (str): The name of the binary traces file.
        cell_centers (numpy.ndarray): Centers of the cells.
        cell_contours (numpy.ndarray): Contours of the cells.
        cell_drying (numpy.ndarray): Indicates the drying status of cells.
        clean (bool): Indicates whether data is cleaned.
        corr_fname (str): The name of the correlation file.
        corr_mat (numpy.ndarray): Correlation matrix.
        date (str): The date of the session.
        F_detrended (numpy.ndarray): Detrended fluorescence data.
        F_upphase (numpy.ndarray): Upphase fluorescence data.
        fps (float): Frames per second for the session.
        functional_chan (int): The functional channel.
        image_x_size (int): The width of the image in pixels.
        image_y_size (int): The height of the image in pixels.
        is_wheel (numpy.ndarray): Indicates whether a wheel is present.
        iscell (numpy.ndarray): Indicates whether a cell is classified.
        pday (str): The postnatal day of the session.
        pval_mat (numpy.ndarray): P-value matrix.
        session_dir (str): The directory path for the session.
        session_id (str): The unique identifier for the session.
        stat (numpy.ndarray): Statistical data.
        velocity (numpy.ndarray): Velocity data.
        yaml_file_path (str): The path to the YAML file containing session metadata.
        zscore_mat (numpy.ndarray): Z-score matrix.

    Parameters:
        - yaml_file_path (str): The path to the YAML file containing session metadata.
        - animal_id (str, optional): The ID of the animal associated with the session.
        - print_loading (bool, optional): If True, print a message when loading session metadata.

    Returns:
        None
    """

    cabincorr_fname = "binarized_traces.npz"
    corr_fname = "allcell_clean_corr_pval_zscore.npy"

    def __init__(self, yaml_file_path, animal_id=None, print_loading=True) -> None:
        self.animal_id = animal_id
        self.yaml_file_path = yaml_file_path
        self.image_x_size = 512
        self.image_y_size = 512
        self.functional_chan = 1
        self.session_id = None
        self.fps = None
        self.date = None
        self.pday = None
        self.load_metadata(yaml_file_path)

        self.clean = False
        self.F_detrended, self.F_upphase = None, None
        self.iscell = None
        self.cell_drying = None
        self.session_dir = os.path.join(
            Animal.root_dir, self.animal_id, self.session_id
        )
        self.velocity = None
        self.is_wheel = None
        self.stat = None
        self.cell_centers = None
        self.cell_contours = None
        self.corr_mat = None
        self.pval_mat = None
        self.zscore_mat = None

        if print_loading:
            print(f"Initialized session: {animal_id} {self.session_id}")

    def load_metadata(self, yaml_path):
        with open(yaml_path, "r") as yaml_file:
            session_metadata_dict = yaml.safe_load(yaml_file)

        # Load any additional metadata into session object
        for variable_name, value in session_metadata_dict.items():
            setattr(self, variable_name, value)
        if not self.session_id:
            self.session_id = str(self.date)
        needed_variables = ["date"]
        for needed_variable in needed_variables:
            defined_variable = getattr(self, needed_variable)
            if defined_variable == None:
                raise KeyError(
                    f"Variable {needed_variable} is not defined in yaml file {yaml_path}"
                )

    def load_fps(self):
        if self.fps:
            return (self.fps,)
        fps_path = os.path.join(self.session_dir, "fps.npy")
        self.fps = np_load_if_exists(fps_path)
        return self.fps

    def load_cell_drying(self):
        if not type(self.cell_drying) == np.ndarray:
            cell_drying_path = os.path.join(self.session_dir, "cell_drying.npy")
            self.cell_drying = np_load_if_exists(cell_drying_path)
        return self.cell_drying

    def load_velocity(self):
        if not type(self.velocity) == np.ndarray:
            velocity_path = os.path.join(self.session_dir, "merged_velocity.npy")
            self.velocity = np_load_if_exists(velocity_path)
        return self.velocity

    def load_is_wheel(self):
        if not type(self.is_wheel) == np.ndarray:
            is_wheel_path = os.path.join(self.session_dir, "is_wheel.npy")
            self.is_wheel = np_load_if_exists(is_wheel_path)
        return self.is_wheel

    def remove_geldrying_cells(self, cell_array):
        cell_drying = self.load_cell_drying()
        cleaned_cell_array = None if cell_array is  None else np.array(cell_array)[cell_drying == False]
        return cleaned_cell_array

    def load_traces(self, clean=False):
        if not type(self.F_detrended) == np.ndarray:
            F_detrended_path = os.path.join(self.session_dir, "F_detrended.npy")
            self.F_detrended = np_load_if_exists(F_detrended_path)
        F_detrended = self.F_detrended
        if clean:
            F_detrended = self.remove_geldrying_cells(self.F_detrended)
        return F_detrended

    def load_binarized_traces(self, clean=False):
        if not type(self.F_upphase) == np.ndarray:
            F_upphase_path = os.path.join(self.session_dir, "F_upphase.npy")
            self.F_upphase = np_load_if_exists(F_upphase_path)
        F_upphase = self.F_upphase
        if clean:
            F_upphase = self.remove_geldrying_cells(self.F_upphase)
        return F_upphase

    def load_fluoresence(self, clean=False):
        F_detrended = self.load_traces(clean=clean)
        F_upphase = self.load_binarized_traces(clean=clean)
        return F_detrended, F_upphase

    def load_stat(self, clean=False):
        if not type(self.stat) == np.ndarray:
            stat_path = os.path.join(self.session_dir, "stat.npy")
            self.stat = np_load_if_exists(stat_path)
        stat = self.stat
        if clean:
            stat = self.remove_geldrying_cells(stat)
        return self.stat

    def get_cell_centers(self, clean=False):
        if not isinstance(self.cell_centers, np.ndarray):
            stat = self.load_stat(clean=clean)
            centers = np.array(
                [
                    [center_point["med"][1], center_point["med"][0]]
                    for center_point in stat
                ]
            )
            self.cell_centers = centers
        centers = self.cell_centers
        if clean:
            centers = self.remove_geldrying_cells(centers)
        return centers

    def get_cell_contours(self, clean=False):
        if not isinstance(self.cell_contours, np.ndarray):
            stat = self.load_stat(clean=clean)
            from scipy.spatial import ConvexHull

            contours = []
            for cell_stat in stat:
                x_y_points = [cell_stat["xpix"], cell_stat["ypix"]]
                # find convex hull of temp
                hull = ConvexHull(np.array(x_y_points).T)
                contour_open = np.array(x_y_points).T[hull.vertices]
                # add the last point to close the contour
                contour = np.vstack([contour_open, contour_open[0]])
                contours.append(contour)
            self.cell_contours = np.array(contours)
        cell_contours = self.cell_contours
        if clean:
            cell_contours = self.remove_geldrying_cells(cell_contours)
        return cell_contours

    def get_correlation_data(self):
        self.corr_mat, self.pval_mat, self.zscore_mat = np.load(
            os.path.join(self.session_dir, Session.corr_fname)
        )
        return self.corr_mat, self.pval_mat, self.zscore_mat
