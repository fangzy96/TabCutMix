import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from scipy.interpolate import UnivariateSpline


def load_and_prepare_data(path, dataset_name):
    csv_files = glob.glob(path)

    all_data = []

    for file in csv_files:
        df = pd.read_csv(file)
        df.columns = ['Epoch', 'Memorization Ratio']

        suffix = int(os.path.basename(file).split('_')[-1].replace('.csv', ''))

        df['label'] = f"{suffix}"

        all_data.append(df)

    combined_data = pd.concat(all_data)

    color_mapping = {'100': 'green', '50': 'orange', '30': 'red', '10': 'brown'}

    combined_data['label'] = pd.Categorical(combined_data['label'], ordered=True, categories=['100', '50', '30', '10'])

    return combined_data, color_mapping


def smooth_data(x, y, smoothing_factor=0.1):
    spl = UnivariateSpline(x, y)
    spl.set_smoothing_factor(smoothing_factor)
    return spl(x)


fig, axes = plt.subplots(1, 4, figsize=(28, 6))
font_size = 32

datasets = {
    'Default': "results/train_size/default/*.csv",
    'Adult': "results/train_size/adult/*.csv",
    'Shoppers': "results/train_size/shoppers/*.csv",
    'Magic': "results/train_size/magic/*.csv"
}

for ax, (dataset_name, path) in zip(axes.flatten(), datasets.items()):

    combined_data, color_mapping = load_and_prepare_data(path, dataset_name)

    for label in combined_data['label'].unique():
        subset = combined_data[combined_data['label'] == label]
        x_smooth = subset['Epoch']
        if dataset_name == 'Magic':
            y_smooth = smooth_data(subset['Epoch'], subset['Memorization Ratio'], smoothing_factor=0.01)
        elif dataset_name == 'Shoppers':
            y_smooth = smooth_data(subset['Epoch'], subset['Memorization Ratio'], smoothing_factor=0.01)
        elif dataset_name == 'Default':
            y_smooth = smooth_data(subset['Epoch'], subset['Memorization Ratio'], smoothing_factor=0.002)
        else:
            y_smooth = smooth_data(subset['Epoch'], subset['Memorization Ratio'], smoothing_factor=0.001)
        ax.plot(x_smooth, y_smooth, label=label, color=color_mapping[label])

    ax.set_title(f'{dataset_name}', fontsize=font_size)
    ax.set_xlabel('Epoch', fontsize=font_size)
    ax.set_ylabel('Memorization Ratio', fontsize=font_size)

    if dataset_name == 'Magic':
        handles, labels = ax.get_legend_handles_labels()
        order = [labels.index('100'), labels.index('50'), labels.index('30'), labels.index('10')]
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], title='Train Size Percent', loc='best',
                  fontsize=28, title_fontsize=28)
    else:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    ax.tick_params(axis='x', labelsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)

    ax.grid(True)

plt.tight_layout()
plt.savefig('plot_replicate_train_size.pdf', format='pdf')
plt.show()
