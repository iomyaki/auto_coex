import gc
import logging

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error

from classes import AE
from funcs import data_loader, custom_min_max_scaling, get_z_scores


def find_anomalies(
    device,
    control_scaled,
    control_scaled_tensor,
    not_control_file,
    name,
    general_min,
    general_max,
    scaling="minmax",
    cutoff=200,  # how many anomalous genes to select
):
    # miscellaneous
    # torch.manual_seed(manual_seed)

    # load data
    not_control = data_loader(not_control_file)
    external_layer_size = len(not_control.columns)
    parameter_names = not_control.columns.values.tolist()
    logging.info(f"Data loaded")

    # scale merged data
    not_control_scaled = custom_min_max_scaling(
        not_control, general_min, general_max
    ).to_numpy()
    logging.info(f"Data scaled")

    # send scaled data to the device
    not_control_scaled_tensor = (
        torch.tensor(not_control_scaled).to(torch.float32).to(device)
    )
    logging.info(f"Scaled data sent to the device")

    # load data of predictive models
    predictive_model_characteristics = pd.read_csv(
        f"characteristics_predict_{name}.csv"
    )

    # calculate reconstructions for each model
    iterator = list(
        zip(
            predictive_model_characteristics["model_name"].to_list(),
            predictive_model_characteristics["bottleneck_size"].to_list(),
        )
    )

    logging.info(
        "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    )
    logging.info(f"Using predictive model set: {name}")
    logging.info(
        "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    )

    err_control, err_not_control = [], []

    for element in iterator:
        model_name, bottleneck_size = element[0], element[1]

        logging.info(
            "========================================================================================"
        )
        logging.info(f"Current model: {model_name}")

        # initialize a model
        model = AE(external_layer_size, bottleneck_size).to(device)
        logging.info("Model initialized")

        # send data to the model
        with torch.no_grad():
            model.load_state_dict(torch.load(f"model_{model_name}.pt"))
            model.eval()

            _, control_out = model(control_scaled_tensor)
            _, not_control_out = model(not_control_scaled_tensor)
        logging.info("Model has reconstructed its input")

        # calculate MSE between input and output vectors for each parameter at the current model
        err_control_m, err_not_control_m = [], []
        for i in range(external_layer_size):
            err_control_m.append(
                mean_squared_error(
                    control_scaled[:, i],
                    np.array(control_out.detach().cpu())[:, i]
                )
            )
            err_not_control_m.append(
                mean_squared_error(
                    not_control_scaled[:, i],
                    np.array(not_control_out.detach().cpu())[:, i],
                )
            )
        logging.info("MSE vectors calculated")

        err_control.append(err_control_m)
        err_not_control.append(err_not_control_m)

        # clear memory
        del model
        del control_out
        del not_control_out
        del err_control_m
        del err_not_control_m
        torch.cuda.empty_cache()
        gc.collect()

    # calculate z-scores of error differences and p-values
    p_values = get_z_scores(err_control, err_not_control, parameter_names)

    # range parameters based on z-score and select the top
    p_values_sorted = p_values.sort_values(by="z-score", ascending=False)[:cutoff]
    p_values_sorted_short = p_values_sorted[["z-score"]]

    # save the result
    p_values.to_csv(
        f"z_scores_{name}.csv",
        float_format=lambda x: "%.3e" % x if abs(x) < 0.001 else "%.3f" % x,
    )

    return p_values_sorted_short, parameter_names, iterator
