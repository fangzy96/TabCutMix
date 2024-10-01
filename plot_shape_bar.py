import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('font', size=45)  # controls default text sizes
plt.rc('axes', titlesize=50)  # fontsize of the axes title
plt.rc('axes', labelsize=45)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=45)  # fontsize of the tick labels
plt.rc('ytick', labelsize=45)  # fontsize of the tick labels
plt.rc('legend', fontsize=40)  # legend fontsize
plt.rc('figure', titlesize=55)  # fontsize of the figure title


tabsyn_folder_path = "results/shape/tabsyn/"
tabcutmix_folder_path = "results/shape/tabcutmix/"


csv_files = [f for f in os.listdir(tabsyn_folder_path) if f.endswith('.csv')]

fig, axes = plt.subplots(1, 4, figsize=(50, 16))
plt.subplots_adjust(wspace=2.5, hspace=0.45)

color_palette = {"TV": "blue", "KS": "orange"}
method_palette = {"TabSyn": "lightblue", "TabCutMix": "lightcoral"}

for i, csv_file in enumerate(csv_files):

    tabsyn_path = os.path.join(tabsyn_folder_path, csv_file)
    tabcutmix_path = os.path.join(tabcutmix_folder_path, csv_file)

    df_tabsyn = pd.read_csv(tabsyn_path)
    df_tabcutmix = pd.read_csv(tabcutmix_path)


    df_tabsyn['Metric'] = df_tabsyn['Metric'].replace({'TVComplement': 'TV', 'KSComplement': 'KS'})
    df_tabcutmix['Metric'] = df_tabcutmix['Metric'].replace({'TVComplement': 'TV', 'KSComplement': 'KS'})

    df_tabsyn['Method'] = 'TabSyn'
    df_tabcutmix['Method'] = 'TabCutMix'

    df_combined = pd.concat([df_tabsyn, df_tabcutmix])

    df_combined['Metric_Column'] = df_combined['Metric'] + " " + df_combined['Column'].astype(str)
    df_combined = df_combined.sort_values(by=['Metric', 'Score'], ascending=True)

    sns.barplot(y="Metric_Column", x="Score", hue="Method", data=df_combined, ax=axes[i],
                palette=method_palette, dodge=True)

    if i < len(csv_files) - 1:
        axes[i].legend().set_visible(False)
    else:
        axes[i].legend(loc='best')

    axes[i].set_title(f"{csv_file.replace('_shape.csv', '').capitalize()}")
    axes[i].set_xlabel('Shape Score')
    axes[i].set_ylabel('Features')

    dataset_name = csv_file.replace('_shape.csv', '').capitalize()
    if dataset_name == "Default":
        axes[i].set_xlim(0.92, 1)
    elif dataset_name == "Adult":
        axes[i].set_xlim(0.9, 1)
    elif dataset_name == "Shoppers":
        axes[i].set_xlim(0.85, 1)
    elif dataset_name == "Magic":
        axes[i].set_xlim(0.75, 1)

plt.tight_layout()
plt.savefig('plot_shape_bar.pdf', format='pdf')
plt.show()
