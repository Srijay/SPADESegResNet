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

The script will create a folder named 'grouped_labels' inside the 'data' folder which will have tiles of size 768×768 pixels from H&E ROIs. To split the dataset into training and testing sets, please run the following script:

```
python ./data_scripts/split_dataset.py
```

It will create the training and testing data inside the 'data/train/' and 'data/test' folders respectively. 

# Model Training

Update the parameters inside the config.txt file and run the following command:

```
python main.py 
```

# Testing 
To test the model, update the parameters inside config.txt file and execute the main file:

```
python main.py 
```

# Evaluation

The boxplots can be plot using the following command:

```
python ./plots/boxplot.py 
```


