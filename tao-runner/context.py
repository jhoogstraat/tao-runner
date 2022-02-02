from pathlib import Path
from typing import Any, Dict, Optional


class TaoConfig:
    def __init__(self, config: Dict[str, Any]):
        self.gpus = str(config['gpus'])
        self.gpu_indices = ','.join([str(i) for i in config['gpu_indices']])


class ExperimentConfig:
    def __init__(self, config: Dict[str, Any]):
        self.head: str = config['head']
        self.backbone: str = config['backbone']
        self.repository: str = config['repository']
        self.model_key: str = config['model_key']
        self.dataset: str = config['dataset']
        self.export_model: str = config['export_model']
        self.export_type: str = config['export_type']


class ExperimentPaths:
    def __init__(self, base: Path, project: str, experiment: str, config: ExperimentConfig, pretrained_model_filename: Optional[str] = None):
        # Base dirs
        self.project_dir = base.joinpath('projects', project)
        self.data_dir = self.project_dir.joinpath('data')
        self.model_dir = self.project_dir.joinpath('models', experiment)
        self.specs_dir = self.project_dir.joinpath('specs', experiment)

        # Specialised dirs
        self.data_raw_dir = self.data_dir.joinpath(config.dataset)
        self.data_tfrecords_dir = self.data_dir.joinpath(
            "tfrecords_" + experiment)
        self.pretrained_model_dir = base.joinpath(
            'repositories', config.repository, config.repository + '_v' + config.backbone)

        # Files
        self.convert_spec_file = self.specs_dir.joinpath('convert.txt')
        self.train_spec_file = self.specs_dir.joinpath('train.txt')
        self.compiled_convert_spec_file = self.data_tfrecords_dir.joinpath(
            self.convert_spec_file.name)
        self.compiled_train_spec_file = self.model_dir.joinpath(
            self.convert_spec_file.name)
        self.pretrained_model_file = self.pretrained_model_dir.joinpath(pretrained_model_filename) if pretrained_model_filename else next(
            self.pretrained_model_dir.glob('*.hdf5'), None)


class ExperimentContext:
    def __init__(self, project: str, experiment: str, config: Dict[str, Any], tao: Dict[str, Any]):
        self.project = project
        self.experiment = experiment
        self.tao = TaoConfig(tao)
        self.config = ExperimentConfig(config)
        self.local_paths = ExperimentPaths(
            project=project, experiment=experiment, config=self.config, base=Path.cwd())
        self.docker_paths = ExperimentPaths(
            project=project, experiment=experiment, config=self.config, base=Path('/workspace'), pretrained_model_filename=self.local_paths.pretrained_model_file.name)
