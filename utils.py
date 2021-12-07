import os
import shutil
import yaml


def init_config(config_path):
    """Return parsed config and create relevant directories"""
    with open(config_path, "r") as config_file:
        config_params = yaml.safe_load(config_file)
    model_cfg        = config_params['Model']  # noqa:E221
    data_cfg         = config_params['Data']  # noqa:E221
    optimizer_cfg    = config_params['Optimizer']  # noqa:E221
    logging_cfg      = config_params['Logging']  # noqa:E221
    measurements_cfg = config_params['Measurements']

    save_dir = logging_cfg['save-dir']
    save_dir_data = os.path.join(save_dir, 'data')
    save_dir_measurements = os.path.join(save_dir, 'measurements')
    if not os.path.exists(save_dir_data):
        os.makedirs(save_dir_data)
    if not os.path.exists(save_dir_measurements):
        os.makedirs(save_dir_measurements)

    shutil.copy(config_path, os.path.join(save_dir, "config.yaml"), follow_symlinks=True)

    return config_params, (model_cfg, data_cfg, optimizer_cfg, logging_cfg, measurements_cfg), (save_dir, save_dir_data, save_dir_measurements)
