# EvaNet

![alt text](https://github.com/mtsami/EvaNet/blob/master/architecture.png?raw=true)

## Abstract
<p align="justify">
High-resolution optical imagery becomes increasingly
available with the wide deployment of satellites and drones,
and accurate and timely mapping of flood extent from the
imagery plays a crucial role in disaster management such
as damage assessment and relief activities. However, the
current state-of-the-art solutions to this problem are based
on U-Net, which cannot segment the flood pixels accurately
due to the ambiguous pixels (e.g., tree canopies,
clouds) that prevent a direct judgement from only the spectral
features. Thanks to the digital elevation model (DEM)
data readily available from sources such as United States
Geological Survey (USGS), this work explores the use of
an elevation map to improve flood extent mapping. We
propose, EvaNet, an elevation-guided segmentation model
based on the encoder-decoder architecture with two novel
techniques: (1) a loss function encoding the physical law of
gravity that if a location is flooded (resp. dry), then its adjacent
locations with a lower (resp. higher) elevation must
also be flooded (resp. dry); (2) convolution operations utilizing
the elevation map to provide a location-sensitive gating
mechanism following GLU to regulate how much spectral
features flow through adjacent layers. Extensive experiments
show that EvaNet significantly outperforms the UNet
baselines, and works as a prefect drop-in replacement
for U-Net in existing solutions to flood extent mapping.
</p>

## Quantitative Results

|          | EvaNet | 4 Channel Input | U-Net  | 3 Channel Input | U-Net | 4 Channel Input |
|   :---:  | :---:  |     :---:       | :---:  |     :---:       | :---: |      :---:      |
|          |  Dry   |     Flood       |  Dry   |     Flood       |  Dry  |      Flood      |
| Accuracy | 1      |     2           |  3     |    4            |       |   6             |
| Precision | 1      |     2           |  3     |    4            |       |   6             |
| Recall | 1      |     2           |  3     |    4            |       |   6             |
| F1-Score | 1      |     2           |  3     |    4            |       |   6             |



## Installation
### Requirements
* Linux
* Anaconda Environment with Python>=3.7
* PyTorch>=1.12.1, torchvision>=0.13.1 and CUDA>=11.3

Conda environment can be created from environment.yml file: 
```
conda env create -f environment.yml
```
To activate: 
```
conda activate elev
```

## Usage
### Data Preparation
Please download dataset from https://uab.box.com/s/rjglwac0qswq13tlvi9axztqicc31lgp

Data must be organized in the following manner for all models:
```
model/
  --data/
    --repo/
      --FloodNetData/
        --Region_1_Features7Channel.npy
        --Region_1_labels.npy
        --...
```

To process data for EvaNet, run the following:
```
cd ./Eva_Net_4_channel/data
python data_maker.py
```

## Train and test models
The run.ipynb (jupyter notebook) file can be used to train and test the different models.


### Alternatively
### Training
Please change directory to the model root directory, then:
```
python main.py --mode training
               --train_region Region_1_3_TRAIN
               --epochs 350
               --lr 1e-6
               --val_freq 1
               --saved_model_epoch 0
```


### Inference
From model root directory run: 
```
python main.py --mode testing
               --test_region Region_2_TEST
               --saved_model_epoch 135
               --out_dir /output
               --save_model_dir /saved_models
```
NOTE:
- --out_dir is where the model predictions are saved.
- --save_model_dir is the folder where model weights are saved.
