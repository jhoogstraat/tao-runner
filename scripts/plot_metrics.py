import os.path as osp
import re
import matplotlib.pyplot as plt
import numpy as np

# Plots the metrics from a TAO log file
# LOG_DIR = "projects/neu-det/results_resnet18_fasterrcnn_01/"
LOG_DIR = "projects/neu-det/results_resnet50_fasterrcnn_02/"
# LOG_DIR = "projects/neu-det/results_vgg16_dssd"

LOG_FILE = "train.log"
# LOG_FILE = "eval_concat.log"

metrics =  {
    "loss": "step - loss: (\d.\d*)",
    "mAP": "mAP@0.5 = (\d.\d*)"
}

# Plotting
# with open(osp.join(LOG_DIR, LOG_FILE), "r") as log:
#     log_content = log.read()
#     for (name, regex) in metrics.items():
#         metric = [float(loss.group(1)) for loss in re.finditer(regex, log_content)]
#         if(len(metric) == 0):
#             print("Metric", name, "not found")
#             continue
#         plt.clf()
#         plt.plot(metric)
#         plt.savefig(name + ".jpg")

rows = []
with open(osp.join(LOG_DIR, LOG_FILE), "r") as log:
    log_content = log.read()
    for (name, regex) in metrics.items():
        metric = [float(loss.group(1)) for loss in re.finditer(regex, log_content)]
        rows.append(metric)

with open('frcnn-resnet50_eval.txt', 'w') as f:
    for i in range(0, len(rows[0])):
        f.write(str(i+1) + '\t\t' + str(rows[0][i]) + '\t\t' + str(rows[1][i]) + '\n')
