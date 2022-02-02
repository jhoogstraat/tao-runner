from pathlib import Path
from typing import Any, Dict
import yaml

class Parser():
    def parse(self, project: str) -> Dict[str, Any]:
        experiments_file = Path('projects', project, 'experiments.yml')
        assert experiments_file.is_file(
        ), f'Experiments file {experiments_file} does not exist.'

        return yaml.safe_load(experiments_file.read_bytes())

