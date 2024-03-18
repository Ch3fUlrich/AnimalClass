import os
from datetime import timedelta
from create_commands_list import create_commands_file


def time_to_qos(days, hours, minutes, seconds):
    wanted_duration = timedelta(
        days=days, hours=hours, minutes=minutes, seconds=seconds
    )

    # define qos based on duration
    weeks2 = timedelta(weeks=2)
    weeks1 = timedelta(weeks=1)
    days1 = timedelta(days=1)
    hours6 = timedelta(hours=6)
    minutes30 = timedelta(minutes=30)

    qos_dict = {
        "2weeks": weeks2,
        "1week": weeks1,
        "1day": days1,
        "6hours": hours6,
        "30min": minutes30,
    }

    for qos, duration in qos_dict.items():
        if wanted_duration > duration:
            return qos
    return qos


def create_sbatch_file(
    commands_fname="commands.cmd",
    pipeline="pipeline",
    job_name="converting",
    n_cpus=10,
    memory_per_cpu=8,
    days=0,
    n_hours=6,
    n_minutes=00,
    n_seconds=00,
    start_line=None,
    end_line=None,
    output_name="converted",
    mail_user="",
    conda_env_name="animal_sergej",
    username="mauser00",
):

    # define sbatch file settings
    days, n_hours, n_minutes, n_seconds = (
        abs(int(days)),
        abs(int(n_hours)),
        abs(int(n_minutes)),
        abs(int(n_seconds)),
    )
    time = f"{days}-{n_hours:02d}:{n_minutes:02d}:{n_seconds:02d}"

    qos = time_to_qos(days, n_hours, n_minutes, n_seconds)

    n_jobs = 0
    if not os.path.exists(commands_fname):
        raise FileNotFoundError(
            f"File {commands_fname} does not exist. It is crucial for the pipeline to run."
        )
    with open(commands_fname, "r") as file:
        n_jobs = len(file.readlines())

    start_line = start_line or 1
    end_line = end_line or n_jobs
    array = f"{start_line}-{end_line}"

    mail_user = '"' + mail_user + '"'

    settings_dict = {
        "job_name": job_name,
        "n_cpus": n_cpus,
        "memory_per_cpu": memory_per_cpu,
        "time": time,
        "qos": qos,
        "array": array,
        "output_name": output_name,
        "mail_user": mail_user,
        "conda_env_name": conda_env_name,
        "username": username,
        "commands_fname": commands_fname,
    }

    # load the template file
    with open("sbatch_file_template.txt", "r") as file:
        sbatch_template = file.read()

    sbatch_text = sbatch_template.format(**settings_dict)

    sbatch_fname = f"run_{pipeline}.sh"
    # write the sbatch file
    with open(sbatch_fname, "w") as file:
        file.write(sbatch_text)

    return sbatch_fname


def run_pipeline(sbatch_fname):
    os.system(f"sbatch {sbatch_fname}")


def main(
    project_root_dir,
    commands_fname="commands.cmd",
    pipeline="pipeline",
    wanted_animal_ids=["all"],
    wanted_session_ids=["all"],
    job_name="converting",
    n_cpus=10,
    memory_per_cpu=8,
    days=0,
    n_hours=6,
    n_minutes=00,
    n_seconds=00,
    start_line=None,
    end_line=None,
    output_name="converted",
    mail_user="",
    conda_env_name="animal_sergej",
    username="mauser00",
    mesc_to_tiff=True,
    suite2p=True,
    binarize=True,
    pairwise_correlate=False,
):

    commands_fname = create_commands_file(
        commands_fname=commands_fname,
        wanted_animal_ids=wanted_animal_ids,
        wanted_session_ids=wanted_session_ids,
        project_root_dir=project_root_dir,
        mesc_to_tiff=mesc_to_tiff,
        suite2p=suite2p,
        binarize=binarize,
        pairwise_correlate=pairwise_correlate,
    )

    sbatch_fname = create_sbatch_file(
        commands_fname=commands_fname,
        pipeline=pipeline,
        job_name=job_name,
        n_cpus=n_cpus,
        memory_per_cpu=memory_per_cpu,
        days=days,
        n_hours=n_hours,
        n_minutes=n_minutes,
        n_seconds=n_seconds,
        start_line=start_line,
        end_line=end_line,
        output_name=output_name,
        mail_user=mail_user,
        conda_env_name=conda_env_name,
        username=username,
    )

    run_pipeline(sbatch_fname)
