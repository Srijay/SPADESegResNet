# InstaDeep Take Home Test: Breast Cancer Semantic Segmentation

This repository contains code for SPADESegResNet, the model developed for semantic segmentation for breast cancer whole slide images

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

The breast cancer whole slie images along with their semantic segmentation maps can be downloaded from the given link in the assignment document, and put it in folders named 'images' and 'masks' inside the 'data' folder. Run the script to extract patches: 

```
python extract_patches.py
```

The script will create a folder named 'grouped_masks' inside the 'data' folder which will have tiles of size 768×768 pixels from whole slide images.

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


