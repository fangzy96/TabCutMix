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

    for index, W in generated_data.iterrows():
        distances = custom_distance(train_data, W, numerical_col_names, categorical_col_names)
        min_index = np.argmin(distances)
        min_distance = distances[min_index]
        distances[min_index] = np.inf
        second_min_index = np.argmin(distances)
        second_min_distance = distances[second_min_index]
        ratio = min_distance / (second_min_distance+1e-9)
        replicate_ratios.append(ratio)

    return replicate_ratios

def plot_memorization_distribution():
    datasets = ['adult', 'default', 'shoppers', 'magic']
    ratios = {
        'default': '20.11%',
        'shoppers': '27.68%',
        'magic': '80.01%',
        'adult': '29.26%'
    }

    tabcutmix_ratios = {
        'default': '16.86%',
        'shoppers': '25.38%',
        'magic': '52.06%',
        'adult': '27.03%'
    }

    fig, axes = plt.subplots(1, 4, figsize=(28, 6))
    font_size = 32

    for i, dataname in enumerate(datasets):

        real_data_path = f'results/distribution/{dataname}/real.csv'
        generated_data_path = f'results/distribution/{dataname}/tabsyn.csv'


        generated_data_path_tabcutmix = f'results/distribution/{dataname}_tabcutmix/tabsyn_tabcutmix.csv'
        real_data_path_tabcutmix = f'results/distribution/{dataname}_tabcutmix/real_100.csv'


        real_data = pd.read_csv(real_data_path)
        generated_data = pd.read_csv(generated_data_path)
        real_data_tabcutmix = pd.read_csv(real_data_path_tabcutmix)
        generated_data_tabcutmix = pd.read_csv(generated_data_path_tabcutmix)


        replicate_ratios = cal_memorization(dataname, generated_data, real_data)
        replicate_ratios_tabcutmix = cal_memorization(dataname, generated_data_tabcutmix, real_data_tabcutmix)


        counts, bins = np.histogram(replicate_ratios, bins=20)
        normalized_counts = counts / counts.sum()  # 归一化处理
        axes[i].bar(bins[:-1], normalized_counts, width=np.diff(bins), edgecolor="black", align="edge",
                    color='coral', alpha=0.7, label=f'TabSyn (Ratio: {ratios[dataname]})')


        counts_tabcutmix, bins_tabcutmix = np.histogram(replicate_ratios_tabcutmix, bins=20)
        normalized_counts_tabcutmix = counts_tabcutmix / counts_tabcutmix.sum()  # 归一化处理
        axes[i].bar(bins_tabcutmix[:-1], normalized_counts_tabcutmix, width=np.diff(bins_tabcutmix),
                    edgecolor="black", align="edge", color='skyblue', alpha=0.7, label=f'TabCutMix (Ratio: {tabcutmix_ratios[dataname]})')


        axes[i].axvline(x=1 / 3, color='orange', linestyle='--', linewidth=3)


        axes[i].set_title(f'{dataname.capitalize()}', fontsize=font_size)
        axes[i].set_xlabel('Distance Ratio', fontsize=font_size)
        axes[i].set_ylabel('Density', fontsize=font_size)
        axes[i].grid(True)
        axes[i].tick_params(axis='x', labelsize=font_size)
        axes[i].tick_params(axis='y', labelsize=font_size)

        axes[i].legend(loc='best', fontsize=22)

    plt.tight_layout()
    plt.savefig('memorization_distribution.pdf', format='pdf')
    plt.show()

if __name__ == "__main__":
    plot_memorization_distribution()
