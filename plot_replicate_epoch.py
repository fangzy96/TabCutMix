import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from scipy.interpolate import UnivariateSpline


models = ['Tabsyn', 'TabDDPM']


csv_files = {model: glob.glob(f'results/epoch/{model}_*.csv') for model in models}

datasets = [os.path.basename(file).split('_')[1].replace('.csv', '') for file in csv_files[models[0]]]


fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(28, 12))

font_size = 32


def smooth_data(x, y, smoothing_factor=0.1):
    spl = UnivariateSpline(x, y)
    spl.set_smoothing_factor(smoothing_factor)
    return spl(x)


for i, dataset_name in enumerate(datasets):

    ax = axes[0, i]
    file_tabsyn = [f for f in csv_files['Tabsyn'] if dataset_name in f][0]
    df_tabsyn = pd.read_csv(file_tabsyn)
    df_tabsyn.columns = ['Epoch', 'Memorization Ratio']
    df_filtered_tabsyn = df_tabsyn
    ax.plot(df_filtered_tabsyn['Epoch'], df_filtered_tabsyn['Memorization Ratio'], label='Tabsyn')
    ax.set_title(f'{dataset_name.capitalize()} - Tabsyn', fontsize=font_size)
    ax.set_xlabel('Epoch', fontsize=font_size)
    ax.set_ylabel('Memorization Ratio', fontsize=font_size)
    ax.tick_params(axis='x', labelsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)
    ax.grid(True)

    ax = axes[1, i]
    file_tabddpm = [f for f in csv_files['TabDDPM'] if dataset_name in f][0]
    df_tabddpm = pd.read_csv(file_tabddpm)
    df_tabddpm.columns = ['Epoch', 'Memorization Ratio']
    df_filtered_tabddpm = df_tabddpm#[df_tabddpm['Epoch'] <= 1000]


    if dataset_name == 'default':
        y_smooth = smooth_data(df_filtered_tabddpm['Epoch'], df_filtered_tabddpm['Memorization Ratio'],
                               smoothing_factor=0.03)
    elif dataset_name == 'adult':
        y_smooth = smooth_data(df_filtered_tabddpm['Epoch'], df_filtered_tabddpm['Memorization Ratio'],
                               smoothing_factor=0.01)
    elif dataset_name == 'magic':
        y_smooth = smooth_data(df_filtered_tabddpm['Epoch'], df_filtered_tabddpm['Memorization Ratio'],
                               smoothing_factor=0.1)
    elif dataset_name == 'shoppers':
        y_smooth = smooth_data(df_filtered_tabddpm['Epoch'], df_filtered_tabddpm['Memorization Ratio'],
                               smoothing_factor=0.15)
    ax.plot(df_filtered_tabddpm['Epoch'], y_smooth, label='TabDDPM')
    ax.set_title(f'{dataset_name.capitalize()} - TabDDPM', fontsize=font_size)
    ax.set_xlabel('Epoch', fontsize=font_size)
    ax.set_ylabel('Memorization Ratio', fontsize=font_size)
    ax.tick_params(axis='x', labelsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)
    ax.grid(True)


# for ax in axes.flatten()[:-1]:
#     legend = ax.get_legend()
#     if legend:
#         legend.remove()


# axes[1, -1].legend(title='Model', fontsize=font_size, title_fontsize=font_size)


plt.tight_layout()

plt.savefig('plot_replicate_epoch.pdf', format='pdf')
plt.show()
