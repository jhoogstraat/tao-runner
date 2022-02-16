from os import environ
from pathlib import Path
from shutil import copyfile

from .context import ExperimentContext
from .parsers.argument_parser import Parser as ArgParser
from .parsers.project_parser import Parser as ProjParser
from . import tasks


def main() -> int:
    args = ArgParser().parse()
    project = ProjParser().parse(args.project)

    # TAO uses ~/.tao_mounts.json, so copying the file there...
    copyfile(Path('.tao_mounts.json'),
             Path.home().joinpath('.tao_mounts.json'))

    # Temporary fix for the TAO Docker Image using WSL 2.
    # https://forums.developer.nvidia.com/t/wsl2-tao-issues/195476
    environ['OVERRIDE_REGISTRY'] = 'local.pwn'

    assert args.command in tasks.known_tasks, f"Unknown task '{args.command}'"
    command = tasks.known_tasks[args.command]

    print(f"Running task '{args.command}' on project '{args.project}'")

    for experiment in args.experiments:
        print(f"Experiment: {experiment}")
        tao_config = project['tao_config']
        experiment_config = project['experiments'][experiment]
        context = ExperimentContext(
            project=args.project, experiment=experiment, config=experiment_config, tao=tao_config)
        command.run(context, **vars(args))

    return 0


if __name__ == "__main__":
    main()
