import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from scipy.interpolate import UnivariateSpline

datasets = ['default', 'adult', 'shoppers', 'magic']


fig, axes = plt.subplots(1, 4, figsize=(28, 6))

color_mapping = {'100': 'green', '70': 'orange', '50': 'red', '30': 'brown'}

font_size = 30

def smooth_data(x, y, smoothing_factor=0.1):
    spl = UnivariateSpline(x, y)
    spl.set_smoothing_factor(smoothing_factor)
    return spl(x)


for i, dataset in enumerate(datasets):

    csv_files = glob.glob(f"results/num_feature/{dataset}/*.csv")

    all_data = []

    for file in csv_files:

        df = pd.read_csv(file)

        suffix = int(os.path.basename(file).split('_')[-1].replace('.csv', ''))

        df['label'] = f"{suffix}"

        all_data.append(df)


    combined_data = pd.concat(all_data)


    combined_data['label'] = pd.Categorical(combined_data['label'], ordered=True, categories=['100', '70', '50', '30'])
    combined_data.columns = ['Epoch', 'Memorization Ratio', 'label']

    ax = axes[i]

    for label in ['100', '70', '50', '30']:
        subset = combined_data[combined_data['label'] == label]
        if not subset.empty:
            x_smooth = subset['Epoch']
            if dataset.capitalize() == 'Magic':
                y_smooth = smooth_data(subset['Epoch'], subset['Memorization Ratio'], smoothing_factor=0.001)
            elif dataset.capitalize() == 'Shoppers':
                y_smooth = smooth_data(subset['Epoch'], subset['Memorization Ratio'], smoothing_factor=0.001)
            elif dataset.capitalize() == 'Default':
                y_smooth = smooth_data(subset['Epoch'], subset['Memorization Ratio'], smoothing_factor=0.0001)
            else:
                y_smooth = smooth_data(subset['Epoch'], subset['Memorization Ratio'], smoothing_factor=0)
            # y_smooth = smooth_data(subset['Epoch'], subset['Memorization Ratio'], smoothing_factor=0.001)
            ax.plot(x_smooth, y_smooth, label=label, color=color_mapping[label])

    ax.set_title(f'{dataset.capitalize()}', fontsize=font_size)
    ax.set_xlabel('Epoch', fontsize=font_size)
    ax.set_ylabel('Memorization Ratio', fontsize=font_size)

    if i == len(datasets) - 1:
        ax.legend(title='Number of Feature Percent', fontsize=22, title_fontsize=22)
    else:

        legend = ax.get_legend()
        if legend:
            legend.remove()

    ax.tick_params(axis='x', labelsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)
    ax.grid(True)


for i in range(len(datasets), 4):
    fig.delaxes(axes[i])

plt.tight_layout()


plt.savefig('plot_replicate_feature.pdf', format='pdf')
plt.show()
