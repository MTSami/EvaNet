# EvaNet

![alt text](https://ik.imagekit.io/lur4324m4/architecture.png?ik-sdk-version=javascript-1.4.3&updatedAt=1668633357848?raw=true)

## Abstract
<p align="justify">
Accurate and timely mapping of flood extent from
high-resolution satellite imagery plays a crucial
role in disaster management such as damage assess-
ment and relief activities. However, current state-
of-the-art solutions are based on U-Net, which can-
not segment the flood pixels accurately due to the
ambiguous pixels (e.g., tree canopies, clouds) that
prevent a direct judgement from only the spectral
features. Thanks to the digital elevation model
(DEM) data readily available from sources such
as United States Geological Survey (USGS), this
work explores the use of an elevation map to im-
prove flood extent mapping. We propose, EvaNet,
an elevation-guided segmentation model based on
the encoder-decoder architecture with two novel
techniques: (1) a loss function encoding the phys-
ical law of gravity that if a location is flooded
(resp. dry), then its adjacent locations with a lower
(resp. higher) elevation must also be flooded (resp.
dry); (2) a new (de)convolution operation that in-
tegrates the elevation map by a location-sensitive
gating mechanism to regulate how much spectral
features flow through adjacent layers. Extensive
experiments show that EvaNet significantly outper-
forms the U-Net baselines, and works as a perfect
drop-in replacement for U-Net in existing solutions
to flood extent mapping. Full paper with an extended appendix is available at: (https://github.com/MTSami/EvaNet/blob/master/full_version_with_appendix.pdf)
</p>

## Quantitative Results
Note: 4-Channel input refers to (RGB+Elevation) data provided as input to model, whereas 3-Channel input refers to just RGB data.

|               |Elevation 4-Channel Input |Elevation 4-Channel Input|U-Net 3-Channel Input   |U-Net 3-Channel Input|U-Net 4-Channel Input  |U-Net 4-Channel Input|
|     :---      | :---:  |     :---:       | :---:  |     :---:       | :---:  |      :---:      |
|               |  Dry   |     Flood       |  Dry   |     Flood       |  Dry   |      Flood      |
| **Accuracy**  | **95.805** |     **95.805**      | 88.076 |    88.076       | 94.927 |     94.927      |
| **Precision** | 95.018 |     **97.102**      | 92.682 |    82.075       | **95.452** |     94.116      |
| **Recall**    | **98.182** |     92.208      | 87.073 |    89.594       | 96.156 |     **93.066**      |
| **F1-Score**  | **96.574** |     **94.592**      | 89.789 |    85.669       | 95.803 |     93.588      |

## Qualitative Results
Below, the predicted flood region (red) for each model is illustrated. 
![alt text](https://ik.imagekit.io/lur4324m4/Results.png?ik-sdk-version=javascript-1.4.3&updatedAt=1668633366444?raw=true)

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
Please change directory to the model root directory. For example, to train EvaNet `cd ./Eva_Net_4_channel` then run:
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
               --saved_model_dir /saved_models
```
NOTE:
- --out_dir is where the model predictions are saved.
- --saved_model_dir is the folder where model weights are saved.
