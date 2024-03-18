# imports
import numpy as np

# Plotting
import matplotlib as mlp
import matplotlib.pyplot as plt, mpld3  # plotting and html plots

plt.style.use("dark_background")
# plt.style.use('default')
mlp.use("Agg")

# Regular Expression searching
import re

# interact with system
import os
import sys


# add root directory to be able to import packages
# todo: make all packages installable so they can be called/imported by environment
module_path = os.path.abspath(os.path.join("../"))
sys.path.append(module_path)

from manifolds.donlabtools.utils.calcium import calcium
from manifolds.donlabtools.utils.calcium.calcium import *
from Classes import *
from Helper import *

# Init Directories and Notebook settings
# root_dir = "\\\\toucan-all.scicore.unibas.ch\\donafl00-calcium$\\Users\\Sergej\\Steffen_Experiments"
root_dir = "/scicore/projects/donafl00-calcium/Users/Sergej/Steffen_Experiments"
Animal.root_dir = root_dir
mice21 = [
    "DON-002865",
    "DON-003165",
    "DON-003343",
    "DON-006084",
    "DON-006085",
    "DON-006087",
]
mice22 = [
    "DON-008497",
    "DON-008498",
    "DON-008499",
    "DON-009191",
    "DON-009192",
    "DON-010473",
    "DON-010477",
]
mice23 = [
    "DON-014837",
    "DON-014838",
    "DON-014840",
    "DON-014847",
    "DON-014849",
    "DON-015078",
    "DON-015079",
    "DON-017115",
    "DON-017117",
    "DON-017118",
    "DON-019207",
    "DON-019210",
    "DON-019213",
    "DON-019542",
    "DON-019545",
]


def main(
    wanted_animal_ids=["all"],
    wanted_session_ids=["all"],
    generate=True,
    mesc_to_tiff=True,
    suite2p=True,
    binarize=True,
    pairwise_correlate=False,
):
    # TODO: skipping option is not integrated
    # generate=False
    animals = load_all(
        root_dir,
        wanted_animal_ids=wanted_animal_ids,
        wanted_session_ids=wanted_session_ids,
    )  # Load all animals

    for animal_id, animal in animals.items():
        print(f"{animal_id}: {list(animal.sessions.keys())}")

    mesc_tiff_suite2p_binarize_correlate(
        animals,
        mesc_to_tiff=mesc_to_tiff,
        suite2p=suite2p,
        binarize=binarize,
        pairwise_correlate=pairwise_correlate,
        regenerate=False,
        compute_corrs=True,
        get_geldrying=False,
        delete_intermediate_files=True,
        plotting=True,
    )

    ## Merging pipeline used for steffens Data
    # for animal_id, animal in animals.items():
    #    for session_id, session in animal.sessions.items():
    #        mesc_folder = os.path.join(session.session_dir, "002P-F")
    #        del_file_dir(os.path.join(mesc_folder, "tif"))
    #        del_file_dir(os.path.join(mesc_folder, "suite2p"))
    #        tif_files = get_files(directory=os.path.join(mesc_folder), ending=".tiff")
    #        for fname in tif_files:
    #            del_file_dir(os.path.join(mesc_folder, fname))
    # animals = load_all(
    #    root_dir,
    #    wanted_animal_ids=wanted_animal_ids,
    #    wanted_session_ids=wanted_session_ids,
    # )  # Load all animals

    # analyze_munits_remove_geldrying_merge_sessions_binarize_compute_correlations(
    #    animals, regenerate=False, compute_corrs=True, delete_intermediate=True
    # )

    # create velocities
    # !!!! currently only for cleaned up version !!!!!
    # !!!!! mat files to npy conversion not working on Scicore!!!!!!
    # create_velo(animals)


def analyze_munits_remove_geldrying_merge_sessions_binarize_compute_correlations(
    animals, regenerate=False, compute_corrs=False, delete_intermediate=False
):
    plotting = True
    viz = Vizualizer(animals, save_dir=Animal.root_dir)
    for animal_id, animal in animals.items():
        for session_id, session in animal.sessions.items():
            global_logger_object.set_save_dir(session.session_dir)
            print(
                f"-----------------------------------Loading Units-----------------------------------"
            )
            global_logger_object.logger.info(
                f"-----------------------------------Loading Units-----------------------------------"
            )
            units = session.get_units(
                restore=True,
                get_geldrying=True,
                unit_type="single",
                generate=True,
                regenerate=regenerate,
            )

            print(
                f"-----------------------------------Merging MUnits-----------------------------------"
            )
            global_logger_object.logger.info(
                f"-----------------------------------Merging MUnits-----------------------------------"
            )
            merged_unit = session.merge_units(
                generate=True, regenerate=regenerate, compute_corrs=compute_corrs
            )

            print(
                f"-----------------------------------Creating correlations matrices-----------------------------------"
            )
            global_logger_object.logger.info(
                f"-----------------------------------Creating correlations matrices-----------------------------------"
            )
            session.load_corr_matrix(
                generate=True, regenerate=regenerate, unit_id="merged"
            )
            #
            print(
                f"-----------------------------------Delete intermediate files (tiff, binary, MUnit suite2p folders)-----------------------------------"
            )
            global_logger_object.logger.info(
                f"-----------------------------------Delete intermediate files (tiff, binary, MUnit suite2p folders)-----------------------------------"
            )
            if delete_intermediate:
                delete_bin_tiff_s2p_intermediate(
                    session, binary=True, tiff=True, intermediate_s2p=False
                )

            if plotting:
                dir_exist_create(os.path.join(viz.save_dir, animal_id))
                dir_exist_create(os.path.join(viz.save_dir, animal_id, session_id))
                init_save_dir = os.path.join(viz.save_dir, animal_id, session_id)
                viz.save_dir = init_save_dir
                mlp.use("Agg")
                print(
                    f"---------------------------------------------Plotting--------------------------------------------"
                )
                global_logger_object.logger.info(
                    f"---------------------------------------------Plotting--------------------------------------------"
                )
                print(
                    f"-----------------------------------Plotting Individual Munits-----------------------------------"
                )
                global_logger_object.logger.info(
                    f"-----------------------------------Plotting Individual Munits-----------------------------------"
                )
                try:
                    for unit_id, unit in units.items():
                        viz.unit_footprints(unit)
                        viz.unit_contours(unit)
                        viz.traces(
                            unit.fluoresence,
                            num_cells="all",
                            animal_id=animal_id,
                            session_id=session_id,
                            unit_id=unit.unit_id,
                        )
                        # viz.save_rasters_fig(unit.c, animal_id=animal_id, session_id=session_id, unit_id=unit.unit_id)
                        # Plot Good Bad fluorescence data in Batches of size 10
                        batch_save_dir = os.path.join(viz.save_dir, "batch_10")
                        dir_exist_create(batch_save_dir)
                        viz.save_dir = batch_save_dir
                        # viz.unit_fluorescence_good_bad(unit, batch_size="all", starting=0)
                        viz.unit_fluorescence_good_bad(unit, batch_size=10, starting=0)
                        viz.save_dir = init_save_dir
                except:
                    print(
                        f"###################################FAILED###################################FAILED###################################FAILED###################################"
                    )
                    global_logger_object.logger.error(
                        f"###################################FAILED###################################FAILED###################################FAILED###################################"
                    )

                print(
                    f"-----------------------------------Plotting Contours-----------------------------------"
                )
                global_logger_object.logger.info(
                    f"-----------------------------------Plotting Contours-----------------------------------"
                )
                # print contours of all combination of units size 2
                ##################################S2P Registration (Footprint position shift determination)##############################
                try:
                    best_unit = session.get_most_good_cell_unit()
                    min_num_usefull_cells = best_unit.num_not_geldrying() / 3
                    units = session.get_usefull_units(min_num_usefull_cells)
                    for unit_id, unit in units.items():
                        if unit_id != best_unit.unit_id:
                            if unit.yx_shift == [0, 0]:
                                session.calc_unit_yx_shifts(best_unit, units)
                                break
                    from itertools import permutations

                    unit_ids = list(units.keys())
                    combinations = list(permutations(unit_ids, 2))
                    ##Plotting original contours
                    # for combination in combinations:
                    #    if combination[0] < combination[1]:
                    #        plt.figure(figsize=(20, 20))
                    #        viz.multi_unit_contours(units, combination=combination, plot_center=True)
                    # Plotting shifted contours
                    for combination in combinations:
                        if combination[0] < combination[1]:
                            plt.figure(figsize=(20, 20))
                            viz.multi_unit_contours(
                                units,
                                combination=combination,
                                plot_center=True,
                                shift=True,
                            )
                except:
                    print(
                        f"###################################FAILED###################################FAILED###################################FAILED###################################"
                    )
                    global_logger_object.logger.error(
                        f"###################################FAILED###################################FAILED###################################FAILED###################################"
                    )
                print(
                    f"-----------------------------------Plotting Merged Session-----------------------------------"
                )
                global_logger_object.logger.info(
                    f"-----------------------------------Plotting Merged Session-----------------------------------"
                )
                try:
                    # Plot merged contours
                    merged_contours = best_unit.contours
                    for unit_id, unit in units.items():
                        if unit_id == best_unit.unit_id:
                            continue
                        merged_contours = np.concatenate(
                            [merged_contours, unit.contours]
                        )
                    plt.figure(figsize=(10, 10))
                    title_comment = (
                        f" {animal_id}_{session_id}_MUnit_{merged_unit.unit_id} merged"
                    )
                    viz.contours(merged_contours, comment=title_comment)
                    plt.savefig(
                        os.path.join(
                            viz.save_dir,
                            f"Contours_{title_comment.replace(' ', '_')}.png",
                        ),
                        dpi=300,
                    )
                except:
                    print(
                        f"###################################FAILED###################################FAILED###################################FAILED###################################"
                    )
                    global_logger_object.logger.error(
                        f"###################################FAILED###################################FAILED###################################FAILED###################################"
                    )
                print(
                    "-----------------------------------Plotting Merged Unit Contours-----------------------------------"
                )
                global_logger_object.logger.info(
                    "-----------------------------------Plotting Merged Unit Contours-----------------------------------"
                )
                try:
                    # Plot deduplicated contours
                    viz.unit_contours(merged_unit)
                except:
                    print(
                        f"###################################FAILED###################################FAILED###################################FAILED###################################"
                    )
                    global_logger_object.logger.error(
                        f"###################################FAILED###################################FAILED###################################FAILED###################################"
                    )
                print(
                    "-----------------------------------Plotting Merged Unit Contours without geldrying-----------------------------------"
                )
                global_logger_object.logger.info(
                    "-----------------------------------Plotting Merged Unit Contours without geldrying-----------------------------------"
                )
                try:
                    # plot contours without geldrying
                    plt.figure(figsize=(10, 10))
                    title_comment = f" {animal_id}_{session_id}_MUnit {merged_unit.unit_id} not geldrying"
                    viz.contours(
                        np.array(merged_unit.contours)[
                            merged_unit.cell_geldrying == False
                        ],
                        comment=title_comment,
                    )
                    plt.savefig(
                        os.path.join(
                            viz.save_dir,
                            f"Contours_{title_comment.replace(' ', '_')}.png",
                        ),
                        dpi=300,
                    )
                except:
                    print(
                        f"###################################FAILED###################################FAILED###################################FAILED###################################"
                    )
                    global_logger_object.logger.error(
                        f"###################################FAILED###################################FAILED###################################FAILED###################################"
                    )

                try:
                    # viz.unit_fluorescence_good_bad(merged_unit, batch_size=10, interactive=False, plot_duplicates=False)
                    batch_save_dir = os.path.join(viz.save_dir, "batch_10")
                    dir_exist_create(batch_save_dir)
                    viz.save_dir = batch_save_dir
                    viz.unit_fluorescence_good_bad(
                        merged_unit,
                        batch_size="10",
                        interactive=False,
                        plot_duplicates=False,
                    )
                except:
                    print(
                        f"###################################FAILED###################################FAILED###################################FAILED###################################"
                    )
                    global_logger_object.logger.error(
                        f"###################################FAILED###################################FAILED###################################FAILED###################################"
                    )


def mesc_tiff_suite2p_binarize_correlate(
    animals,
    mesc_to_tiff=True,
    suite2p=True,
    binarize=True,
    pairwise_correlate=True,
    regenerate=False,
    compute_corrs=False,
    get_geldrying=False,
    delete_intermediate_files=True,
    plotting=True,
):
    viz = Vizualizer(animals, save_dir=Animal.root_dir)
    for animal_id, animal in animals.items():
        for session_id, session in animal.sessions.items():
            log_save_dir = os.path.join(session.session_dir, "logs")
            if not os.path.exists(log_save_dir):
                os.mkdir(log_save_dir)
            global_logger_object.set_save_dir(save_dir=log_save_dir)
            print(
                f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Starting {animal_id} {session_id} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
            )
            global_logger_object.logger.info(
                f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Starting {animal_id} {session_id} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
            )
            print(
                f"-----------------------------------Generating TIFF from MESC-----------------------------------"
            )
            global_logger_object.logger.info(
                f"-----------------------------------Generating TIFF from MESC-----------------------------------"
            )
            session.generate_tiff_from_mesc(
                generate=mesc_to_tiff, regenerate=regenerate
            )

            print(
                f"-----------------------------------Generating Binarization Files-----------------------------------"
            )
            global_logger_object.logger.info(
                f"-----------------------------------Generating Binarization Files-----------------------------------"
            )
            session.generate_suite2p(
                generate=suite2p, regenerate=regenerate, unit_ids="all"
            )

            print(
                f"-----------------------------------Generating Binarization Files-----------------------------------"
            )
            global_logger_object.logger.info(
                f"-----------------------------------Generating Binarization Files-----------------------------------"
            )
            session.generate_cabincorr(
                generate=binarize,
                regenerate=regenerate,
                unit_ids="all",
                compute_corrs=compute_corrs,
            )

            print(
                f"-----------------------------------Creating correlations matrices-----------------------------------"
            )
            global_logger_object.logger.info(
                f"-----------------------------------Creating correlations matrices-----------------------------------"
            )
            session.load_corr_matrix(
                generate=pairwise_correlate, regenerate=regenerate, unit_id="all"
            )

            print(
                f"-----------------------------------Delete intermediate files (tiff, binary, MUnit suite2p folders)-----------------------------------"
            )
            global_logger_object.logger.info(
                f"-----------------------------------Delete intermediate files (tiff, binary, MUnit suite2p folders)-----------------------------------"
            )
            if delete_intermediate_files:
                delete_bin_tiff_s2p_intermediate(
                    session, binary=True, tiff=True, intermediate_s2p=False
                )

            if plotting:
                print(
                    f"-----------------------------------Loading Units-----------------------------------"
                )
                global_logger_object.logger.info(
                    f"-----------------------------------Loading Units-----------------------------------"
                )
                units = session.get_units(
                    restore=False,
                    get_geldrying=get_geldrying,
                    unit_type="summary",
                    generate=False,
                    regenerate=regenerate,
                )
                dir_exist_create(os.path.join(viz.save_dir, animal_id))
                dir_exist_create(os.path.join(viz.save_dir, animal_id, session_id))
                viz.save_dir = os.path.join(viz.save_dir, animal_id, session_id)
                print(
                    f"---------------------------------------------Plotting--------------------------------------------"
                )
                global_logger_object.logger.info(
                    f"---------------------------------------------Plotting--------------------------------------------"
                )
                print(
                    f"-----------------------------------Plotting Full Session-----------------------------------"
                )
                global_logger_object.logger.info(
                    f"-----------------------------------Plotting Full Session-----------------------------------"
                )
                unit_all = units[""]
                print(
                    f"-----------------------------------Plotting Full Session footprints-----------------------------------"
                )
                global_logger_object.logger.info(
                    f"-----------------------------------Plotting Full Session footprints-----------------------------------"
                )
                try:
                    viz.unit_footprints(unit_all)
                except:
                    print(
                        f"###################################FAILED###################################FAILED###################################FAILED###################################"
                    )
                    global_logger_object.logger.error(
                        f"###################################FAILED###################################FAILED###################################FAILED###################################"
                    )
                print(
                    f"-----------------------------------Plotting Full Session contours-----------------------------------"
                )
                global_logger_object.logger.info(
                    f"-----------------------------------Plotting Full Session contours-----------------------------------"
                )
                try:
                    viz.unit_contours(unit_all)
                except:
                    print(
                        f"###################################FAILED###################################FAILED###################################FAILED###################################"
                    )
                    global_logger_object.logger.error(
                        f"###################################FAILED###################################FAILED###################################FAILED###################################"
                    )
                print(
                    f"-----------------------------------Plotting Full Session traces-----------------------------------"
                )
                global_logger_object.logger.info(
                    f"-----------------------------------Plotting Full Session traces-----------------------------------"
                )
                try:
                    viz.traces(
                        unit_all.fluoresence,
                        num_cells="all",
                        animal_id=animal_id,
                        session_id=session_id,
                        unit_id=unit_all.unit_id,
                    )
                except:
                    print(
                        f"###################################FAILED###################################FAILED###################################FAILED###################################"
                    )
                    global_logger_object.logger.error(
                        f"###################################FAILED###################################FAILED###################################FAILED###################################"
                    )
                print(
                    f"-----------------------------------Plotting Full Session pearson histogram -----------------------------------"
                )
                global_logger_object.logger.info(
                    f"-----------------------------------Plotting Full Session pearson histogram -----------------------------------"
                )
                try:
                    viz.unit_footprints(unit_all)
                    viz.unit_contours(unit_all)
                    viz.traces(
                        unit_all.fluoresence,
                        num_cells="all",
                        animal_id=animal_id,
                        session_id=session_id,
                        unit_id=unit_all.unit_id,
                    )
                    corr_matrix, pval_matrix = viz.pearson_hist(
                        animal_id,
                        session_id,
                        unit_id="all",
                        remove_geldrying=get_geldrying,
                        generate_corr=compute_corrs,
                        color_classify=True,
                    )
                except:
                    print(
                        f"###################################FAILED###################################FAILED###################################FAILED###################################"
                    )
                    global_logger_object.logger.error(
                        f"###################################FAILED###################################FAILED###################################FAILED###################################"
                    )


def create_velo(animals):
    # FIXME: using matlab and python at the same time not working on scicore
    if False:
        for animal_id, session_id, session in yield_animal_session(animals):
            min_usefull_cells = 80  # if session.animal_id in mice23 else 100
            if not min_usefull_cells:
                print(f"{animal_id} {session_id} No min_usefull_cells Skipping...")
                continue
            session.get_units(
                restore=True,
                get_geldrying=False,
                unit_type="summary",
                generate=False,
                regenerate=False,
            )
            # session.convert_movement_data() # Already done in merge_movements
            session.get_units(
                restore=True,
                get_geldrying=False,
                unit_type="single",
                generate=False,
                regenerate=False,
            )
            wheel, triggers, velocity = session.load_movements(
                merged=True,
                min_num_usefull_cells=80,
                regenerate=True,
                movement_data_types=["wheel", "triggers", "velocity"],
            )
            if "merged" not in session.units.keys():
                continue
            fluor_fpath = os.path.join(
                session.units["merged"].suite2p_dir, Session.fluoresence_fname
            )
            if os.path.exists(fluor_fpath) and type(velocity) == np.ndarray:
                fluoresence = np.load(fluor_fpath)
                print(f"Fluoresence shape: {fluoresence.shape}")
                print(f"velocity shape: {velocity.shape}")
                global_logger_object.logger.info(
                    f"Fluoresence shape: {fluoresence.shape}"
                )
                global_logger_object.logger.info(f"velocity shape: {velocity.shape}")
            else:
                print(
                    f"not os.path.exists(fluor_fpath) and type(velocity) == np.ndarray"
                )
                global_logger_object.logger.error(
                    f"not os.path.exists(fluor_fpath) and type(velocity) == np.ndarray"
                )


if __name__ == "__main__":
    arguments = sys.argv[1:]
    wanted_animal_ids = sys.argv[1:2] if len(arguments) >= 1 else ["all"]
    wanted_session_ids = sys.argv[2:3] if len(arguments) >= 2 else ["all"]

    mesc_to_tiff = sys.argv[3:4] if len(arguments) >= 3 else True
    suite2p = sys.argv[4:5] if len(arguments) >= 4 else True
    binarize = sys.argv[5:6] if len(arguments) >= 5 else True
    pairwise_correlate = sys.argv[6:7] if len(arguments) >= 6 else True

    if len(arguments) > 3:
        print("Command line usage: <animal_id> <session_id>")
        print(
            "If an argument is not specified the corresponding argument is set to 'all'"
        )

    print(f"Start {wanted_animal_ids}, {wanted_session_ids}: ")
    main(
        wanted_animal_ids=wanted_animal_ids,
        wanted_session_ids=wanted_session_ids,
        mesc_to_tiff=mesc_to_tiff,
        suite2p=suite2p,
        binarize=binarize,
        pairwise_correlate=pairwise_correlate,
    )
