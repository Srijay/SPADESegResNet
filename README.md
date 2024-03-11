# InstaDeep Take Home Test
# SPADESegResNet: Combining power of SPADE and ResNet for Breast Cancer Semantic Segmentation

![image](https://github.com/Srijay/SPADESegResNet/assets/6882352/14f39972-d5ba-47a0-aff0-cf322cbde712)

This repository contains code for SPADESegResNet, the model developed for semantic segmentation for breast cancer whole slide images. Please follow the instructions given below to setup the environment and execute the code.

# Set Up Environment

Clone the repository, and execute the following commands to set up the environment.

```
cd Ouroboros

# create base conda environment
conda env create -f environment.yml

# activate environment
conda activate spadesegresnet

# install PyTorch with pip
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```

# Download data and extract tiles

The breast cancer whole slide images, along with their semantic segmentation maps, can be downloaded from the provided link in the assignment document. Please place the downloaded images in the 'data/images' folder and the annotations or segmentation maps inside the 'data/labels' folder. The first step is to group labels of similar tissue regions together. For this purpose, download the gtruth_codes.tsv file from the provided data link, located inside the 'meta' folder. Insert it into the 'data' folder and execute the following command.

```
python ./data_scripts/construct_groupings.py
```

The script will create a folder named 'grouped_labels' inside the 'data' folder. To split the dataset into training and testing sets, please run the following script:

```
python ./data_scripts/split_dataset.py
```

It will create the training and testing data inside the 'data/train/' and 'data/test/' folders respectively. Now, please run the following script to extract tiles of size 768×768 pixels:

```
python ./data_scripts/extract_tiles.py
```

The script will create tiles and put inside the ./data/train/cropped/768 when applied on the training data. Similarly it can be applied on testing data and create testing images inside ./data/folder/cropped/768.


# Model Training

Now, we are set to train the model. Please update the training parameters inside the config.txt file, put mode='train' and run the following command:

```
python main.py 
```

All three models used in the paper — the proposed SPADESegResNet model and the baseline models UNet and UNet++ — are placed inside the 'model' folder. Please set the model you would like to train in the config.txt.

# Testing and Evaluation

To test the model, update the parameters inside config.txt file and execute the main file:

```
python main.py 
```

Please set mode='test_patch' if you want to compute semantic segmentation maps on tiles of the same size used for training. To generate segmentation maps of a larger size, please keep mode='test_wsi'. After executing the script, it will compute and store semantic segmentation maps in the 'pred' directory inside the output folder path given in config.txt. The script will also print the overall accuracy and store the list of Dice scores and AUC-ROC values inside the output folder with names <model_name>_dice.json and <model_name>_auc.json, respectively.

Now, to compute mean Dice score, mean AUC-ROC, their standard deviations, boxplots and p-values for statistical significant, please execute the following script by updating suitable paths:

```
python compute_statistics.py 
```

