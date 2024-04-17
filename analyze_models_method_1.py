import gc
import logging

import numpy as np
import pandas as pd
import torch

from classes import AE


def analyze_models_method_1(
    device,
    external_layer_size: int,
    iterator_ctrl,
    iterator_not_ctrl,
    pvalues_sorted_short,
    parameter_names,
):
    def iterate_model_set(
        device,
        external_layer_size,
        identity_matrix_tensor,
        iterator,
        pvalues_sorted_short,
        selected_parameters,
        parameter_names,
    ):
        for element in iterator:
            model_name, bottleneck_size = element[0], element[1]

            logging.info(f"Now analyzing {model_name}")

            # initiate a model
            model = AE(external_layer_size, bottleneck_size).to(device)

            # predict the identity matrix
            with torch.no_grad():
                model.load_state_dict(torch.load(f"model_{model_name}.pt"))
                model.eval()

                _, output = model(identity_matrix_tensor)

            # save the prediction
            adjacency_matrix = pd.DataFrame(
                np.array(output.detach().cpu()),
                index=parameter_names,
                columns=parameter_names,
            )
            adjacency_matrix_T = adjacency_matrix.T

            # filter the adjacency matrix
            adjacency_matrix_index_filtered = adjacency_matrix[
                adjacency_matrix.index.isin(selected_parameters)
            ]
            adjacency_matrix_column_filtered = adjacency_matrix_T[
                adjacency_matrix_T.index.isin(selected_parameters)
            ]

            # concatenate the adjacency matrix with the list of anomalies
            result_index_filtered = pd.concat(
                [pvalues_sorted_short, adjacency_matrix_index_filtered], axis=1
            )
            result_column_filtered = pd.concat(
                [pvalues_sorted_short, adjacency_matrix_column_filtered], axis=1
            )

            # save the result
            result_index_filtered.to_csv(
                f"analysis_method_1_index_filtered_{model_name}.csv"
            )
            result_column_filtered.to_csv(
                f"analysis_method_1_column_filtered_{model_name}.csv"
            )
            logging.info(f"Model {model_name} analysis (m. 1) results saved")

            # clear memory
            del model
            del adjacency_matrix
            del adjacency_matrix_T
            del adjacency_matrix_index_filtered
            del adjacency_matrix_column_filtered
            del result_index_filtered
            del result_column_filtered
            torch.cuda.empty_cache()
            gc.collect()

            break

    # create testing data
    identity_matrix = np.eye(external_layer_size)
    identity_matrix_tensor = torch.tensor(identity_matrix).to(torch.float32).to(device)

    # analyze every model
    selected_parameters = list(pvalues_sorted_short.index)

    logging.info("Analysis by method #1 started")

    iterate_model_set(
        device,
        external_layer_size,
        identity_matrix_tensor,
        iterator_ctrl,
        pvalues_sorted_short,
        selected_parameters,
        parameter_names,
    )
    logging.info("Control models analysed (m. 1)")

    iterate_model_set(
        device,
        external_layer_size,
        identity_matrix_tensor,
        iterator_not_ctrl,
        pvalues_sorted_short,
        selected_parameters,
        parameter_names,
    )
    logging.info("Not control models analysed (m. 1)")

    logging.info("Analysis by method #1 finished")

    # clear memory
    del identity_matrix
    del identity_matrix_tensor
    del selected_parameters
