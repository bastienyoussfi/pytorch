from fastdownload import download_url
from fastbook import *
from time import sleep
from img import *
from fastai.vision.all import *
from fastai.vision.widgets import *
from fastcore.all import *
from embedded_bash import *
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

# It's a good idea to ensure you're running the latest version of any libraries you need.
# `!pip install -Uqq <libraries>` upgrades to the latest version of <libraries>
# NB: You can safely ignore any warnings or errors pip spits out about running as root or incompatibilities
import os
iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')

if iskaggle:
    run_script("!pip install -Uqq fastai duckduckgo_search")

searches = 'cat', 'dog'
path = Path('cat_or_dog')

# Dowload the images we want to create a training set
for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo', max_images = 50))
    #sleep(10)  # Pause between searches to avoid over-loading server
    download_images(dest, urls=search_images(f'{o} cute photo', max_images = 50))
    #sleep(10)
    download_images(dest, urls=search_images(f'{o} animal photo', max_images = 50))
    #sleep(10)
    resize_images(path/o, max_size=400, dest=path/o)

# Remove the uncorrectly downloaded images
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)

# Creating a DataBlock to store our data and create our model
cat_and_dogs = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
)

cat_and_dogs = cat_and_dogs.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())

dls = cat_and_dogs.dataloaders(path)

# Get a sample of our images and show it in a plot
# dls.show_batch(max_n=6)
#plt.show()

# Training our model with resnet18
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)

# Testing our model on test images
#urls = search_images('cat photos', max_images=1)
#dest = 'cat.jpg'
#download_url(urls[0], dest, show_progress=False)
#is_cat,_,probs = learn.predict(PILImage.create('cat.jpg'))
#print(f"This is a: {is_cat}.")
#print(f"Probability it's a {is_cat}: {probs[0]:.4f}")

# Confusion matrix
interp = ClassificationInterpretation.from_learner(learn)
#interp.plot_confusion_matrix()
interp.plot_top_losses(5, nrows=1)
plt.show()

learn.export('model.pkl')