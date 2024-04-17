import gc
import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from classes import AE
from funcs import calculate_correlation_matrix


def analyze_models_method_2(
    device, external_layer_size: int, iterator_ctrl, iterator_not_ctrl, parameter_names
):
    def iterate_model_set(device, external_layer_size, iterator, parameter_names):
        for element in iterator:
            model_name, bottleneck_size = element[0], element[1]

            logging.info(f"Now analyzing {model_name}")

            # initiate a model
            model = AE(external_layer_size, bottleneck_size).to(device)

            with torch.no_grad():
                model.load_state_dict(torch.load(f"model_{model_name}.pt"))
                model.eval()

                for param_name, param in model.named_parameters():
                    if param_name == "decoder.0.weight":
                        weight_matrix = param.detach().cpu().numpy()
                        break

            # calculate correlation matrix
            corr_matrix = pd.DataFrame(
                calculate_correlation_matrix(weight_matrix, external_layer_size),
                index=parameter_names,
                columns=parameter_names,
            )

            corr_matrix.to_csv(f"analysis_method_2_corr_matrix_{model_name}.csv")

            plt.figure(figsize=(15, 15), dpi=600)
            sns.clustermap(corr_matrix)
            plt.savefig(f"analysis_method_2_clustermap_{model_name}.png")

            logging.info(f"Model {model_name} analysis (m. 2) result saved")

            # clear memory
            del model
            del weight_matrix
            del corr_matrix
            torch.cuda.empty_cache()
            gc.collect()

            break

    # analyze every model
    logging.info("Analysis by method #2 started")

    iterate_model_set(device, external_layer_size, iterator_ctrl, parameter_names)
    logging.info("Control models analysed (m. 2)")

    iterate_model_set(device, external_layer_size, iterator_not_ctrl, parameter_names)
    logging.info("Not control models analysed (m. 2)")

    logging.info("Analysis by method #2 finished")
