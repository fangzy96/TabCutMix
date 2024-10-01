import pandas as pd
import matplotlib.pyplot as plt
import glob
import re

datasets = ['default', 'shoppers']

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
font_size = 45

for ax, dataset in zip(axes, datasets):
    csv_files = glob.glob(f'results/cutmix_threshold/{dataset}/{dataset}_*.csv')
    print(csv_files)
    file_suffix_list = []

    for file in csv_files:
        match = re.search(rf'{dataset}_ratio_(\d+)', file)
        if match:
            numeric_suffix = int(match.group(1))
            file_suffix_list.append((file, numeric_suffix))

    file_suffix_list = sorted(file_suffix_list, key=lambda x: x[1])

    for file, suffix in file_suffix_list:

        df = pd.read_csv(file)
        df.columns = ['Epoch', 'Memorization Ratio']

        df_filtered = df#[df['Epoch'] <= 800]
        print(df_filtered)

        ax.plot(df_filtered['Epoch'], df_filtered['Memorization Ratio'], label=f'{suffix}%')


    ax.set_title(f'{dataset.capitalize()}', fontsize=font_size)
    ax.set_xlabel('Epoch', fontsize=font_size)
    ax.set_ylabel('Memorization Ratio', fontsize=font_size)
    ax.tick_params(axis='x', labelsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)
    ax.grid(True)

for ax in axes[:-1]:
    legend = ax.get_legend()
    if legend:
        legend.remove()

axes[-1].legend(title='Augmented Ratio', fontsize=30, title_fontsize=30)

plt.tight_layout()

plt.savefig('plot_replicate_epoch_cutmix_threshold.pdf', format='pdf')
