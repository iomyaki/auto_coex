import logging
import os
import pickle
import random
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
from torch.utils.data import DataLoader

from classes import CustomDataset


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_random_state(name):
    if os.path.exists(f"random_seed_{name}.seed"):
        # load random state
        with open(f"random_seed_{name}.seed", "rb") as f:
            st = pickle.load(f)
            np.random.set_state(st)
    else:
        np.random.seed(None)
        st = np.random.get_state()
        # save random state
        with open(f"random_seed_{name}.seed", "wb") as f:
            pickle.dump(st, f)


def data_loader(file: str) -> pd.DataFrame or None:
    extension = file.split(".")[-1].lower()

    if extension == "csv" or extension == "tsv":
        df = pd.read_csv(file, index_col=0).T
    elif extension == "xlsx" or extension == "xls":
        df = pd.read_excel(file, index_col=0).T
    else:
        logging.error("Incorrect input format")
        return None

    df.reset_index(inplace=True)
    df.drop("index", axis=1, inplace=True)

    return df


def get_std_vector(df: pd.DataFrame) -> list:
    return df.std().tolist()


def repeat_dataset(df: pd.DataFrame, multiplier: int) -> pd.DataFrame:
    return df.loc[df.index.repeat(multiplier)].reset_index(drop=True)


def add_noise(df: pd.DataFrame, std_vector: list, noise_factor: float) -> pd.DataFrame:
    shape = df.shape
    noise = np.random.normal(
        [0] * shape[1], np.array(std_vector, float) * noise_factor, shape
    )
    noisy_df = df + noise

    return noisy_df


def get_general_min_max(
    df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame, axis=0
) -> tuple:
    min_1, min_2, min_3 = df1.min(axis=axis), df2.min(axis=axis), df3.min(axis=axis)
    max_1, max_2, max_3 = df1.max(axis=axis), df2.max(axis=axis), df3.max(axis=axis)

    for i in range(len(min_1)):
        min_1.iloc[i] = min(min_1.iloc[i], min_2.iloc[i], min_3.iloc[i])
        max_1.iloc[i] = max(max_1.iloc[i], max_2.iloc[i], max_3.iloc[i])

    return min_1, max_1


def custom_min_max_scaling(
    df: pd.DataFrame, min_custom: pd.Series, max_custom: pd.Series, feature_range=(0, 1)
) -> pd.DataFrame:
    return (df - min_custom) / (max_custom - min_custom) * (feature_range[1] - feature_range[0]) + feature_range[0]


def gpu_train_test_split(
    data: list, df_len: int, multiplier: int, test_size: float, device: torch.device
) -> tuple:
    """
    Perform random train/test split and send data to the GPU
    """

    random.shuffle(data)

    resulting_len = df_len * multiplier
    split_num = int(resulting_len * test_size)

    df_test, df_train = data[:split_num], data[split_num:]

    df_train_tensor = torch.tensor(np.array(df_train)).to(torch.float32).to(device)
    df_test_tensor = torch.tensor(np.array(df_test)).to(torch.float32).to(device)

    del resulting_len, split_num, df_test, df_train
    return df_train_tensor, df_test_tensor


def torch_dataloading(tensor: torch.tensor, batch_size: int):
    custom_dataset = CustomDataset(tensor)
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def get_bottlenecks(external_layer_size: int) -> list:
    bottleneck_sizes = []

    size = external_layer_size
    while size > int(external_layer_size / 1000) and size > 1:
        size //= 2
        bottleneck_sizes.append(size)

    bottleneck_sizes.reverse()

    del size

    return bottleneck_sizes


def training_loop(
    device: torch.device,
    model,
    criterion,
    optimizer,
    epochs: int,
    train_data,
    test_data,
) -> tuple:
    train_loss, eval_loss = [], []

    for epoch in range(epochs):
        start_time = datetime.now()

        # train the model
        model.train()
        train_loss_epoch = []

        for batch in train_data:
            batch_noise = batch["noise_data"].to(device)
            batch_orig = batch["orig_data"].to(device)

            _, output = model(batch_noise)

            # comparison — forward
            loss_train_value = criterion(output, batch_orig)

            # change weights — backward
            optimizer.zero_grad()
            loss_train_value.backward()
            optimizer.step()

            train_loss_epoch.append(loss_train_value.detach().cpu().numpy())

        train_loss.append(np.mean(train_loss_epoch))

        # evaluate the model
        model.eval()
        eval_loss_epoch = []

        with torch.no_grad():
            for batch in test_data:
                batch_noise = batch["noise_data"].to(device)
                batch_orig = batch["orig_data"].to(device)

                _, output = model(batch_noise)

                loss_eval_value = criterion(output, batch_orig)

                eval_loss_epoch.append(loss_eval_value.detach().cpu().numpy())

        eval_loss.append(np.mean(eval_loss_epoch))

        end_time = datetime.now()
        time_spent = end_time - start_time

        logging.info(
            f"Epoch: {epoch + 1}/{epochs}, "
            f"train_loss: {np.mean(train_loss_epoch):4f}, "
            f"eval_loss: {np.mean(eval_loss_epoch):4f}, "
            f"t_spent: {time_spent}"
        )

    return train_loss, eval_loss


def draw_loss_plots(model_name: str, train_loss: list, eval_loss: list) -> None:
    train_loss[0] = None  # because usually is too big and breaks the scale

    train_loss_chart = plt.plot(train_loss, color="red")
    eval_loss_chart = plt.plot(eval_loss, color="blue")

    plt.title(f"Training dynamics of {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss value")
    plt.legend((train_loss_chart[0], eval_loss_chart[0]), ("training", "evaluation"))
    plt.grid(True)

    plt.savefig(f"loss_of_{model_name}.png")
    plt.clf()


def find_saturation_point(x: list, y: list, y_type: str) -> int:
    points = [abs((y[i] - y[i - 1]) / (x[i] - x[i - 1])) for i in range(1, len(x))]
    min_value = min(points)
    var_value = np.var(points)
    i = 0
    while i < len(points) and points[i] > min_value + var_value:
        i += 1
    if i < len(points):
        if y_type.lower() == "mse" or y_type.lower() == "r_sq" and max(y) >= 0.81:
            return x[i + 1]
        else:
            logging.critical(
                f"Model can not train on these data: max(R^2) value is {max(y)}"
            )
            sys.exit(1)
    else:
        logging.critical(
            f"Model can not train on these data: no significant {y_type} change"
        )
        sys.exit(2)


def get_z_scores(err_control: list, err_not_control: list, parameter_names: list):
    # subtract control MSEs from not control MSEs & nullify negative diffs
    err_control = np.array(err_control)
    err_not_control = np.array(err_not_control)

    err_diff = np.subtract(err_not_control, err_control)
    err_diff = np.clip(err_diff, 0.0, None)

    # summarize error values for each gene
    err_sum = np.sum(err_diff, axis=0)

    # calculate metrics
    z_score = stats.zscore(err_sum)
    p_value = stats.norm.sf(abs(z_score))
    fdr_value = fdrcorrection(p_value)[1]
    logging.info("Metrics calculated")

    p_values = pd.DataFrame(
        {
            "sum_diff_err": err_sum,
            "z-score": z_score,
            "p_value": p_value,
            "fdr": fdr_value,
        },
        index=parameter_names,
    )

    return p_values


def calculate_correlation_matrix(weight_matrix, external_layer_size):
    corr_matrix = np.empty((external_layer_size, external_layer_size))

    for i in range(external_layer_size):
        corr_matrix[i][i] = 1.0
        for j in range(i + 1, external_layer_size):
            corr = np.corrcoef(weight_matrix[i], weight_matrix[j])[0][1]
            corr_matrix[i][j], corr_matrix[j][i] = corr, corr

    return corr_matrix
