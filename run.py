#!/usr/bin/env python3

import logging
import os
import sys
from pathlib import Path

import pandas as pd

from analyze_models_method_1 import analyze_models_method_1
from analyze_models_method_2 import analyze_models_method_2
from find_anomalies import find_anomalies
from fit_models import fit_models
from funcs import get_device


if __name__ == "__main__":
    control_file = os.path.realpath(sys.argv[1])
    not_control_file = os.path.realpath(sys.argv[2])
    name = sys.argv[3]

    Path(f"{name}").mkdir(parents=True, exist_ok=True)
    os.chdir(f"{name}")

    # configure logging
    logging.basicConfig(
        format="%(levelname)s (%(asctime)s): %(message)s (Line: %(lineno)d [%(filename)s])",
        datefmt="%d/%m/%Y %I:%M:%S %p",
        level=logging.INFO,
        filename=f"logs_{name}.log",
        filemode="w",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # activate CUDA
    device = get_device()
    logging.info(f"Device: {device}")

    # fit models and find anomalies
    (
        external_layer_size,
        general_min,
        general_max,
        control_scaled,
        control_scaled_tensor,
    ) = fit_models(device, control_file, not_control_file, f"{name}_ctrl")

    pvalues_sorted_short, parameter_names, iterator_ctrl = find_anomalies(
        device,
        control_scaled,
        control_scaled_tensor,
        not_control_file,
        f"{name}_ctrl",
        general_min,
        general_max,
    )

    fit_models(device, not_control_file, control_file, f"{name}_not_ctrl")

    # create model iterator for further analysis
    characteristics_not_ctrl = pd.read_csv(
        f"characteristics_predict_{name}_not_ctrl.csv"
    )

    iterator_not_ctrl = list(
        zip(
            characteristics_not_ctrl["model_name"].to_list(),
            characteristics_not_ctrl["bottleneck_size"].to_list(),
        )
    )

    # perform model analysis
    analyze_models_method_1(
        device,
        external_layer_size,
        iterator_ctrl,
        iterator_not_ctrl,
        pvalues_sorted_short,
        parameter_names,
    )
    analyze_models_method_2(
        device, external_layer_size, iterator_ctrl, iterator_not_ctrl, parameter_names
    )
