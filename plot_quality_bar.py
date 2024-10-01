import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_comparison(datasets):
    sns.set_context("notebook", font_scale=1.5)  # Adjust font scale to increase text size

    # Set all font sizes to 18
    plt.rc('font', size=30)  # controls default text sizes
    plt.rc('axes', titlesize=30)  # fontsize of the axes title
    plt.rc('axes', labelsize=30)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=30)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=24)  # fontsize of the tick labels
    plt.rc('legend', fontsize=18)  # legend fontsize
    plt.rc('figure', titlesize=30)  # fontsize of the figure title

    fig, axes = plt.subplots(2, len(datasets), figsize=(24, 12))

    plt.tight_layout(rect=[0.02, 0.32, 0.98, 0.95])  # Adjust the rect to reduce left and right margins
    plt.subplots_adjust(wspace=0.45, hspace=0.45)

    for i, dataname in enumerate(datasets):
        print(f"\n--- Processing dataset: {dataname} ---\n")

        # File paths
        real_data_path = f'results/distribution/{dataname}_tabcutmix/real_100.csv'
        generated_data_path_tabsyn = f'results/distribution/{dataname}/tabsyn.csv'
        generated_data_path_tabcutmix = f'results/distribution/{dataname}_tabcutmix/tabsyn_tabcutmix.csv'

        # Load the data
        real_data = pd.read_csv(real_data_path)[:50]
        tabsyn_data = pd.read_csv(generated_data_path_tabsyn)[:50]
        tabcutmix_data = pd.read_csv(generated_data_path_tabcutmix)[:50]

        # Select a numerical feature and a categorical feature for each dataset
        if dataname == 'adult':
            num_feature = 'fnlwgt'
            cat_feature = 'relationship'
        elif dataname == 'default':
            num_feature = 'BILL_AMT4'
            cat_feature = 'PAY_0'
        elif dataname == 'shoppers':
            num_feature = 'ExitRates'
            cat_feature = 'VisitorType'
        elif dataname == 'magic':
            num_feature = 'Asym'  # Example feature, adjust as needed
            cat_feature = 'class'  # Example feature, adjust as needed

        # Plot numerical feature (Density Plot)
        ax = axes[0, i]
        ax.grid()
        sns.kdeplot(real_data[num_feature], ax=ax, label='Real', color='blue', fill=True)
        sns.kdeplot(tabsyn_data[num_feature], ax=ax, label='TabSyn', color='orange', fill=True)
        sns.kdeplot(tabcutmix_data[num_feature], ax=ax, label='TabCutMix', color='green', fill=True)
        ax.set_title(f'{dataname.capitalize()}')
        ax.set_ylabel('Density')
        if i == len(datasets) - 1:  # Only show legend for the last plot in the row
            ax.legend()
        else:
            ax.legend().remove()

        # Plot categorical feature (Bar Plot)
        ax = axes[1, i]
        ax.grid()
        real_counts = real_data[cat_feature].value_counts(normalize=True)
        tabsyn_counts = tabsyn_data[cat_feature].value_counts(normalize=True)
        tabcutmix_counts = tabcutmix_data[cat_feature].value_counts(normalize=True)

        df_bar = pd.DataFrame({
            'Category': real_counts.index,
            'Real': real_counts.values,
            'TabSyn': tabsyn_counts.reindex(real_counts.index, fill_value=0).values,
            'TabCutMix': tabcutmix_counts.reindex(real_counts.index, fill_value=0).values
        })

        df_bar_melted = df_bar.melt(id_vars='Category', var_name='Model', value_name='Proportion')
        sns.barplot(x='Category', y='Proportion', hue='Model', data=df_bar_melted, ax=ax)

        ax.set_xlabel(f'{cat_feature.capitalize()}')  # Set x-axis label to feature name
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        if i == len(datasets) - 1:  # Only show legend for the last plot in the row
            ax.legend()
        else:
            ax.legend().remove()
    plt.savefig('quality_bar.pdf', format='pdf')

    plt.show()



if __name__ == "__main__":
    datasets = ['adult', 'default', 'shoppers', 'magic']
    plot_comparison(datasets)
