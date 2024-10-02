# (ICLR 2025 Submission) Unveiling Memorization in Diffusion Models for Tabular Data

Tabular data generation has attracted significant research interest in recent years, with the tabular diffusion models greatly improving the quality of synthetic data. However, while memorization—where models inadvertently replicate exact or near-identical training data—has been thoroughly investigated in image and text generation, its effects on tabular data remain largely unexplored. In this paper, we conduct the first comprehensive investigation of memorization phenomena in diffusion models for tabular data. Our empirical analysis reveals that memorization appears in tabular diffusion models and increases with larger training epochs. We further examine the influence of factors such as dataset sizes, feature dimensions, and different diffusion models on memorization. Additionally, we provide a theoretical explanation for why memorization occurs in tabular diffusion models. To address this issue, we propose TabCutMix, a simple yet effective data augmentation technique that exchanges randomly selected feature segments between random training sample pairs. Experimental results across various datasets and diffusion models demonstrate that TabCutMix effectively mitigates memorization while maintaining high-quality data generation.


## Installing Dependencies

Python version: 3.10

Create environment

```
conda create -n tabsyn python=3.10
conda activate tabsyn
```

Install pytorch
```
pip install torch torchvision torchaudio
```

or via conda
```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Install other dependencies

```
pip install -r requirements.txt
```


Create another environment for the quality metric (package "synthcity")

```
conda create -n synthcity python=3.10
conda activate synthcity

pip install synthcity
pip install category_encoders
```

## Preparing Datasets

Download raw dataset:

```
python download_dataset.py
```

Process dataset:

```
python process_dataset.py
```
uncomment line 170 for different train sizes

Process dataset for different feature dimensions:
```
python remove_features.py
python process_dataset.py
```

## Training Models

For baseline methods, use the following command for training:

```
python main.py --dataname [NAME_OF_DATASET] --method [NAME_OF_BASELINE_METHODS] --mode train
```

Options of [NAME_OF_DATASET]: adult, default, shoppers, magic
Options of [NAME_OF_BASELINE_METHODS]: stay, tabddpm

For Tabsyn, use the following command for training:

```
# train VAE first
python main.py --dataname [NAME_OF_DATASET] --method vae --mode train

# after the VAE is trained, train the diffusion model
python main.py --dataname [NAME_OF_DATASET] --method tabsyn --mode train
```

## Tabular Data Synthesis

For baseline methods, use the following command for synthesis:

```
python main.py --dataname [NAME_OF_DATASET] --method [NAME_OF_BASELINE_METHODS] --mode sample --save_path [PATH_TO_SAVE]
```

For Tabsyn, use the following command for synthesis:

```
python main.py --dataname [NAME_OF_DATASET] --method tabsyn --mode sample --save_path [PATH_TO_SAVE]
```

The default save path is "synthetic/[NAME_OF_DATASET]/[METHOD_NAME].csv"

## Memorization Ratio

```
python cal_memorization.py
```

## Evaluation
We evaluate the quality of synthetic data using metrics from various aspects.

#### Density estimation of single column and pair-wise correlation ([link](https://docs.sdv.dev/sdmetrics/reports/quality-report/whats-included))

```
python eval/eval_density.py --dataname [NAME_OF_DATASET] --model [METHOD_NAME] --path [PATH_TO_SYNTHETIC_DATA]
```


#### Alpha Precision and Beta Recall ([paper link](https://arxiv.org/abs/2102.08921))
- $\alpha$-preicison: the fidelity of synthetic data
- $\beta$-recall: the diversity of synthetic data

```
python eval/eval_quality.py --dataname [NAME_OF_DATASET] --model [METHOD_NAME] --path [PATH_TO_SYNTHETIC_DATA]
```

#### Machine Learning Efficiency

```
python eval/eval_mle.py --dataname [NAME_OF_DATASET] --model [METHOD_NAME] --path [PATH_TO_SYNTHETIC_DATA]
```

## Visualization
```
python plot_replicate_epoch.py
python plot_replicate_train_size.py
python plot_replicate_feature.py
python plot_replicate_cutmix_threshold.py
python plot_memorization_distribute_merge.py
python plot_memorization_visualization.py
python plot_heatmap.py
python plot_shape_bar.py
```

## Case Study
```
python case_study.py
```

## Acknowledgements

This project was built upon code from [TabSyn](https://github.com/amazon-science/tabsyn). We are deeply grateful for their open-source contributions, which have significantly helped shape the development of this project.

Specifically, many of the model components in this repository are based on the foundation provided by [TabSyn](https://github.com/amazon-science/tabsyn). We highly recommend checking out their work for further insights.
