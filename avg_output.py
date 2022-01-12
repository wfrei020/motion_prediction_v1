import numpy as np
import motion_metric as mm
from waymo_open_dataset.metrics.python import config_util_py as config_util


set_50 = np.load('results_start_0_end_50.npy',allow_pickle=True)
set_100 = np.load('results_start_50_end_100.npy',allow_pickle=True)
set_150 = np.load('results_start_100_end_150.npy',allow_pickle=True)
metrics_config = mm.default_metrics_config()
metric_names = config_util.get_breakdown_names_from_motion_config(metrics_config)

for i, m in enumerate(
        ['min_ade', 'min_fde', 'miss_rate', 'overlap_rate', 'map']):
    for j, n in enumerate(metric_names):
        print('{}/{}: {}'.format(m, n, (set_50[i, j] + set_100[i,j] + set_150[i, j])/3))