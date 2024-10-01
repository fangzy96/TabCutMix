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

### Using the datasets adopted in the paper

Download raw dataset:

```
python download_dataset.py
```

Process dataset:

```
python process_dataset.py
```

## Training Models

For baseline methods, use the following command for training:

```
python main.py --dataname [NAME_OF_DATASET] --method [NAME_OF_BASELINE_METHODS] --mode train
```

Options of [NAME_OF_DATASET]: adult, default, shoppers, magic, beijing, news
Options of [NAME_OF_BASELINE_METHODS]: smote, goggle, great, stasy, codi, tabddpm

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


