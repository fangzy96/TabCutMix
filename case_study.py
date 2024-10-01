import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def custom_distance(X, Y, numerical_cols, categorical_cols):
    distances = np.zeros(X.shape[0])

    if numerical_cols:
        X_num = X[numerical_cols].astype(float).values
        Y_num = Y[numerical_cols].astype(float).values.reshape(1, -1)
        num_diff = X_num - Y_num
        euclidean_distances = np.sqrt(np.sum(num_diff ** 2, axis=1))
        scaler = MinMaxScaler()
        normalized_distances = scaler.fit_transform(euclidean_distances.reshape(-1, 1)).flatten()
        distances += normalized_distances

    if categorical_cols:
        X_cat = X[categorical_cols].values
        Y_cat = Y[categorical_cols].values.reshape(1, -1)
        cat_diff = (X_cat != Y_cat).astype(float)
        cat_distances = cat_diff.sum(axis=1)
        distances += cat_distances

    average_distances = distances / (len(numerical_cols) + len(categorical_cols))

    return average_distances


def find_closest_samples(dataname, generated_data, train_data):
    numerical_cols = {
        'magic': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'shoppers': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'adult': [0, 2, 4, 10, 11, 12],
        'default': [0, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    }

    categorical_cols = {
        'magic': [10],
        'shoppers': [10, 11, 12, 13, 14, 15, 16, 17],
        'adult': [1, 3, 5, 6, 7, 8, 9, 13, 14],
        'default': [1, 2, 3, 5, 6, 7, 8, 9, 10, 23]
    }

    if dataname in numerical_cols:
        numerical_col_names = train_data.columns[numerical_cols[dataname]].tolist()
        categorical_col_names = train_data.columns[categorical_cols[dataname]].tolist()
    else:
        print('Invalid dataname.')
        return None

    closest_samples = []

    for index, W in generated_data.iterrows():
        distances = custom_distance(train_data, W, numerical_col_names, categorical_col_names)
        min_index = np.argmin(distances)
        min_distance = distances[min_index]
        distances[min_index] = np.inf
        second_min_index = np.argmin(distances)
        second_min_distance = distances[second_min_index]

        if second_min_distance > 0 and np.isfinite(min_distance) and np.isfinite(second_min_distance):
            ratio = min_distance / second_min_distance
        else:
            ratio = np.nan

        closest_samples.append((index, min_index, min_distance, ratio))

    return closest_samples


def perform_case_study():
    datasets = ['adult']

    for dataname in datasets:
        print(f"Case study for dataset: {dataname}")

        real_data_path = f'results/distribution/{dataname}/real.csv'
        generated_data_path = f'results/distribution/{dataname}/tabsyn.csv'

        generated_data_path_tabcutmix = f'results/distribution/{dataname}_tabcutmix/tabsyn_tabcutmix.csv'

        real_data = pd.read_csv(real_data_path)
        generated_data = pd.read_csv(generated_data_path)
        generated_data_tabcutmix = pd.read_csv(generated_data_path_tabcutmix)

        closest_tab_syn_samples = find_closest_samples(dataname, generated_data, real_data)

        closest_tab_cutmix_samples = find_closest_samples(dataname, generated_data_tabcutmix, real_data)

        count = 0
        printed_indices = set()

        for i, (gen_index, min_index, min_distance, ratio) in enumerate(closest_tab_syn_samples):
            if ratio < 1 / 3 and min_index not in printed_indices:
                print(f"\nReal sample (TabSyn memorized):")
                print(real_data.iloc[min_index])
                print(f"\nGenerated TabSyn sample:")
                print(generated_data.iloc[gen_index])


                for j, (gen_index_cutmix, min_index_cutmix, min_distance_cutmix, ratio_cutmix) in enumerate(
                        closest_tab_cutmix_samples):
                    if ratio_cutmix >= 1 / 3 and min_index_cutmix == min_index:
                        print(f"\nClosest non-memorized TabCutMix sample:")
                        print(generated_data_tabcutmix.iloc[gen_index_cutmix])
                        printed_indices.add(min_index)
                        count += 1
                        break

                if count >= 10:
                    break


if __name__ == "__main__":
    perform_case_study()
