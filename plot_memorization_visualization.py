import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.manifold import TSNE
import random

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
        Y_cat = Y[categorical_cols].astype(str).values.reshape(1, -1)
        cat_diff = (X_cat != Y_cat).astype(float)
        cat_distances = cat_diff.sum(axis=1)
        distances += cat_distances

    average_distances = distances / (len(numerical_cols) + len(categorical_cols))

    return average_distances

def cal_memorization(dataname, generated_data, train_data):
    column_indices = {
        'magic': {
            'numerical': [0,1,2,3,4,5,6,7,8,9],
            'categorical': [10]
        },
        'shoppers': {
            'numerical': [0,1,2,3,4,5,6,7,8,9],
            'categorical': [10,11,12,13,14,15,16,17]
        },
        'adult': {
            'numerical': [0,2,4,10,11,12],
            'categorical': [1,3,5,6,7,8,9,13,14]
        },
        'default': {
            'numerical': [0,4,11,12,13,14,15,16,17,18,19,20,21,22],
            'categorical': [1,2,3,5,6,7,8,9,10,23]
        }
    }

    if dataname in column_indices:
        numerical_cols = column_indices[dataname]['numerical']
        categorical_cols = column_indices[dataname]['categorical']
    else:
        print('Invalid dataname.')
        return None

    numerical_col_names = train_data.columns[numerical_cols].tolist()
    categorical_col_names = train_data.columns[categorical_cols].tolist()

    replicate_ratios = []
    memorized_flags = []
    nearest_indices = []

    for index, W in generated_data.iterrows():
        distances = custom_distance(train_data, W, numerical_col_names, categorical_col_names)
        min_index = np.argmin(distances)
        min_distance = distances[min_index]
        distances[min_index] = np.inf
        second_min_index = np.argmin(distances)
        second_min_distance = distances[second_min_index]
        ratio = min_distance / second_min_distance
        replicate_ratios.append(ratio)
        memorized_flags.append(ratio < 1/3)
        nearest_indices.append((min_index, second_min_index))

    return replicate_ratios, memorized_flags, nearest_indices

def plot_memorization_visualization(ax, dataname, generated_data, train_data, memorized_flags, nearest_indices, numerical_cols, categorical_cols, method_name, ratio, show_legend=False):
    num_memorized = int(ratio * 100)
    num_non_memorized = 100 - num_memorized

    selected_memorized_indices = [i for i in range(len(memorized_flags)) if memorized_flags[i]]
    selected_non_memorized_indices = [i for i in range(len(memorized_flags)) if not memorized_flags[i]]

    selected_memorized = random.sample(selected_memorized_indices, min(num_memorized, len(selected_memorized_indices)))
    selected_non_memorized = random.sample(selected_non_memorized_indices, min(num_non_memorized, len(selected_non_memorized_indices)))

    selected_indices = selected_memorized + selected_non_memorized
    selected_data = generated_data.iloc[selected_indices]

    X_num = selected_data[numerical_cols].values
    X_cat = selected_data[categorical_cols].values

    X_train_num = train_data[numerical_cols].values
    X_train_cat = train_data[categorical_cols].values

    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    X_cat_encoded = encoder.fit_transform(X_cat)
    X_train_cat_encoded = encoder.transform(X_train_cat)

    combined_data = np.hstack([X_num, X_cat_encoded])

    selected_nearest_indices = [nearest_indices[i] for i in selected_indices]
    nearest_data = np.vstack([np.hstack([train_data.iloc[idx[0]][numerical_cols].values, X_train_cat_encoded[idx[0]]]) for idx in selected_nearest_indices])
    second_nearest_data = np.vstack([np.hstack([train_data.iloc[idx[1]][numerical_cols].values, X_train_cat_encoded[idx[1]]]) for idx in selected_nearest_indices])

    tsne_data = np.vstack([combined_data, nearest_data, second_nearest_data])
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(tsne_data)

    selected_tsne = tsne_results[:100]
    nearest_tsne = tsne_results[100:200]
    second_nearest_tsne = tsne_results[200:300]

    memorized_tsne = selected_tsne[:num_memorized]
    non_memorized_tsne = selected_tsne[num_memorized:]

    font_size = 32

    ax.scatter(second_nearest_tsne[:, 0], second_nearest_tsne[:, 1], color='blue', marker='^', label='Second Nearest', alpha=0.5)
    ax.scatter(nearest_tsne[:, 0], nearest_tsne[:, 1], color='blue', marker='o', label='Nearest', alpha=0.5)
    ax.scatter(non_memorized_tsne[:, 0], non_memorized_tsne[:, 1], color='green', marker='x', label='Non-Memorized')
    ax.scatter(memorized_tsne[:, 0], memorized_tsne[:, 1], color='red', marker='x', label='Memorized')

    ax.set_title(f'{dataname.capitalize()} {method_name} ({ratio*100:.2f}%)', fontsize=font_size)
    ax.grid(True)
    ax.tick_params(axis='x', labelsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)

    if show_legend:
        ax.legend(fontsize=22, title_fontsize=22)
    else:
        ax.legend().set_visible(False)

def plot_memorization_distribution():
    datasets = ['adult', 'default', 'shoppers', 'magic']
    ratios = {
        'default': 0.2011,
        'shoppers': 0.2768,
        'magic': 0.8001,
        'adult': 0.2926
    }

    tabcutmix_ratios = {
        'default': 0.1686,
        'shoppers': 0.2538,
        'magic': 0.5206,
        'adult': 0.2703
    }

    fig, axes = plt.subplots(2, 4, figsize=(30, 12))
    plt.subplots_adjust(wspace=0.45, hspace=0.45)
    for i, dataname in enumerate(datasets):
        real_data_path = f'results/distribution/{dataname}/real.csv'
        generated_data_path = f'results/distribution/{dataname}/tabsyn.csv'

        real_data_path_tabcutmix = f'results/distribution/{dataname}_tabcutmix/real_100.csv'
        generated_data_path_tabcutmix = f'results/distribution/{dataname}_tabcutmix/tabsyn_tabcutmix.csv'

        real_data = pd.read_csv(real_data_path)
        generated_data = pd.read_csv(generated_data_path)
        real_data_tabcutmix = pd.read_csv(real_data_path_tabcutmix)
        generated_data_tabcutmix = pd.read_csv(generated_data_path_tabcutmix)

        replicate_ratios, memorized_flags, nearest_indices = cal_memorization(dataname, generated_data, real_data)
        replicate_ratios_tabcutmix, memorized_flags_tabcutmix, nearest_indices_tabcutmix = cal_memorization(dataname, generated_data_tabcutmix, real_data_tabcutmix)

        column_indices = {
            'magic': {
                'numerical': [0,1,2,3,4,5,6,7,8,9],
                'categorical': [10]
            },
            'shoppers': {
                'numerical': [0,1,2,3,4,5,6,7,8,9],
                'categorical': [10,11,12,13,14,15,16,17]
            },
            'adult': {
                'numerical': [0,2,4,10,11,12],
                'categorical': [1,3,5,6,7,8,9,13,14]
            },
            'default': {
                'numerical': [0,4,11,12,13,14,15,16,17,18,19,20,21,22],
                'categorical': [1,2,3,5,6,7,8,9,10,23]
            }
        }

        numerical_cols = real_data.columns[column_indices[dataname]['numerical']]
        categorical_cols = real_data.columns[column_indices[dataname]['categorical']]

        plot_memorization_visualization(axes[0, i], dataname, generated_data, real_data, memorized_flags, nearest_indices, numerical_cols, categorical_cols, 'TabSyn', ratios[dataname], show_legend=False)
        plot_memorization_visualization(axes[1, i], dataname, generated_data_tabcutmix, real_data_tabcutmix, memorized_flags_tabcutmix, nearest_indices_tabcutmix, numerical_cols, categorical_cols, 'TabCutMix', tabcutmix_ratios[dataname], show_legend=(i == 3))

    plt.tight_layout()
    plt.savefig('memorization_visualization_comparison.pdf', format='pdf')
    plt.show()

if __name__ == "__main__":
    plot_memorization_distribution()
