import os
from string import Template
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
    n_cores=10,
    memory_per_core=8,
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
    time = f"{days}-{n_hours}:{n_minutes}:{n_seconds}"

    qos = time_to_qos(days, n_hours, n_minutes, n_seconds)

    n_jobs = 0
    with open(commands_fname, "r") as file:
        n_jobs = len(file.readlines())

    start_line = start_line or 1
    end_line = end_line or n_jobs
    array = f"{start_line}-{end_line}"

    mail_user = '"' + mail_user + '"'

    # load the template file
    with open("sbatch_file_template.txt", "r") as file:
        sbatch_template = Template(file.read())

    sbatch_text = sbatch_template.substitute(.....)

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
    n_cores=10,
    memory_per_core=8,
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

    commands_fname = create_commands_file(
        commands_fname=commands_fname,
        wanted_animal_ids=wanted_animal_ids,
        wanted_session_ids=wanted_session_ids,
        project_root_dir=project_root_dir,
    )

    sbatch_fname = create_sbatch_file(
        commands_fname,
        pipeline,
        job_name,
        n_cores,
        memory_per_core,
        days,
        n_hours,
        n_minutes,
        n_seconds,
        start_line,
        end_line,
        output_name,
        mail_user,
        conda_env_name,
        username,
    )

    run_pipeline(sbatch_fname)
