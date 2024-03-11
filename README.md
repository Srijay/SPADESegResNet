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

The breast cancer whole slie images along with their semantic segmentation maps can be downloaded from the given link in the assignment document. Please put the downloaded images into the 'data/images' folder and annotations or segmentation maps inside the 'data/labels'. Run the following script to extract tiles of size 768×768 pixels: 

```
python ./data_scripts/extract_tiles.py
```

Next step is to group similar tissues together. For this purpose, first download the gtruth_codes.tsv file from the given data link. It's located inside the 'meta' folder. Put it inside the 'data' folder and execute the following command

```
python ./data_scripts/construct_groupings.py
```

The script will create a folder named 'grouped_labels' inside the 'data' folder. To split the dataset into training and testing sets, please run the following script:

```
python ./data_scripts/split_dataset.py
```

It will create the training and testing data inside the 'data/train/' and 'data/test' folders respectively. 



# Model Training

Update the training parameters inside the config.txt file, put mode='train' and run the following command:

```
python main.py 
```

# Testing and Evaluation
To test the model, update the parameters inside config.txt file and execute the main file:

```
python main.py 
```

Please put mode='test_patch' if want to compute semantic segmentation maps on tiles of same sized used for training. To generate segmentation maps of higher size, please keep mode='test_wsi'. After executing the script, it will compute and store semantic segmentation maps in 'pred' directory inside the output folder path given in config.txt. The script will also print the overall accuracy and store the list of Dice scores and AUC-ROC values inside the output folder with names <model_name>_dice.json and <model_name>_auc.json respectively.

```
python main.py 
```

Now, to compute mean Dice score, mean AUC-ROC, their standard deviations, boxplots and p-values for statistical significant, please execute the following script by updating suitable paths:

```
python compute_statistics.py 
```

