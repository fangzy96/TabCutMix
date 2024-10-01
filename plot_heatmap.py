import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compute_absolute_divergence(real_data, synthetic_data, numerical_cols, categorical_cols):

    real_data_num = real_data.iloc[:, numerical_cols]
    synthetic_data_num = synthetic_data.iloc[:, numerical_cols]

    real_data_cat = pd.get_dummies(real_data.iloc[:, categorical_cols], drop_first=True)
    synthetic_data_cat = pd.get_dummies(synthetic_data.iloc[:, categorical_cols], drop_first=True)


    real_data_combined = pd.concat([real_data_num, real_data_cat], axis=1)
    synthetic_data_combined = pd.concat([synthetic_data_num, synthetic_data_cat], axis=1)

    real_data_combined, synthetic_data_combined = real_data_combined.align(synthetic_data_combined, join='outer',
                                                                           axis=1, fill_value=0)

    real_corr = real_data_combined.corr().values
    synthetic_corr = synthetic_data_combined.corr().values

    divergence = np.abs(real_corr - synthetic_corr)
    return divergence

def plot_heatmaps(datasets, column_indices, real_data_paths, synthetic_data_paths):
    num_datasets = len(datasets)
    num_methods = len(synthetic_data_paths[0])


    plt.rc('font', size=24)  # controls default text sizes
    plt.rc('axes', titlesize=24)  # fontsize of the axes title
    plt.rc('axes', labelsize=22)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=24)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=24)  # fontsize of the tick labels
    plt.rc('legend', fontsize=22)  # legend fontsize
    plt.rc('figure', titlesize=24)  # fontsize of the figure title


    fig, axes = plt.subplots(num_methods, num_datasets, figsize=(20, 8))
    plt.tight_layout(rect=[0.01, 0.01, 0.98, 0.98])
    plt.subplots_adjust(wspace=0.2, hspace=0.15)

    methods = ['TabSyn', 'TabCutMix']

    for i, dataset in enumerate(datasets):
        real_data = pd.read_csv(real_data_paths[i])
        numerical_cols = column_indices[dataset]['numerical']
        categorical_cols = column_indices[dataset]['categorical']

        for j, method in enumerate(methods):
            synthetic_data = pd.read_csv(synthetic_data_paths[i][j])
            divergence = compute_absolute_divergence(real_data, synthetic_data, numerical_cols, categorical_cols)

            ax = axes[j, i]
            sns.heatmap(divergence, ax=ax, cmap='Blues')
            ax.set_title(f'{dataset.capitalize()} - {method}', fontsize=24)
            ax.axis('off')
    plt.savefig('quality_heatmap.pdf', format='pdf')
    plt.show()



# Define datasets and methods
datasets = ['adult', 'default', 'shoppers', 'magic']

# Define the file paths
real_data_paths = [
    'results/distribution/adult/real.csv',
    'results/distribution/default/real.csv',
    'results/distribution/shoppers/real.csv',
    'results/distribution/magic/real.csv'
]

synthetic_data_paths = [
    ['results/distribution/adult/tabsyn.csv', 'results/distribution/adult_tabcutmix/tabsyn_tabcutmix.csv'],
    ['results/distribution/default/tabsyn.csv', 'results/distribution/default_tabcutmix/tabsyn_tabcutmix.csv'],
    ['results/distribution/shoppers/tabsyn.csv', 'results/distribution/shoppers_tabcutmix/tabsyn_tabcutmix.csv'],
    ['results/distribution/magic/tabsyn.csv', 'results/distribution/magic_tabcutmix/tabsyn_tabcutmix.csv']
]

# Column indices for numerical and categorical data
column_indices = {
    'magic': {
        'numerical': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'categorical': [10]
    },
    'shoppers': {
        'numerical': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'categorical': [10, 11, 12, 13, 14, 15, 16, 17]
    },
    'adult': {
        'numerical': [0, 2, 4, 10, 11, 12],
        'categorical': [1, 3, 5, 6, 7, 8, 9, 13, 14]
    },
    'default': {
        'numerical': [0, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        'categorical': [1, 2, 3, 5, 6, 7, 8, 9, 10, 23]
    }
}

plot_heatmaps(datasets, column_indices, real_data_paths, synthetic_data_paths)
