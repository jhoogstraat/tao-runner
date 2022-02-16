import subprocess
import csv
from pathlib import Path

#######################################################################
#Description    : Downloads all available pretrained models from NGC
#Command:       : python3 download_pretrained_models.py
#Author         : Joshua Hoogstraat
#Date           : 14.01.2021
#######################################################################

# Commands for NGC CLI
ngc_list_cmd = ['ngc', 'registry', 'model', 'list', '--format_type', 'csv']
ngc_download_cmd = ['ngc', 'registry', 'model', 'download-version']

# Existing Repositories in NGC (retrieved with 'ngc registry model list --format_type csv nvidia/tao/pretrained_*')
repositories = [
    'pretrained_classification',
    'pretrained_object_detection',
    'pretrained_detectnet_v2',
    'pretrained_efficientdet',
    'pretrained_semantic_segmentation',
    'pretrained_instance_segmentation',
]

# Download each model in each repository
for repository in repositories:
    print("***", repository, "***")

    directive = "nvidia/tao/" + repository
    destination = Path('./pretrained_models/' + repository)
    destination.mkdir(exist_ok=True)

    repo_models = subprocess.run([*ngc_list_cmd, directive + ":*"],
                                 check=True, text=True, capture_output=True)

    parsed = csv.reader(repo_models.stdout.splitlines()[1:])
    for row in parsed:
        print("Downloading", row[0])
        subprocess.run([*ngc_download_cmd,
                        '--dest', destination.as_posix(),
                        directive + ":" + row[0]],
                        check=True, text=True)
