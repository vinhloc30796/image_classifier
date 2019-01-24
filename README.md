# Data Scientist Nanodegree
# Deep Learning
## Project: Create Your Own Image Classifier

### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [PyTorch](https://pytorch.org)
- [PIL](https://PILlow.readthedocs.io/)
- [argparse](https://docs.python.org/3/library/argparse.html)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project.

### Code

Exploration code is provided in the `Image Classifier Project.ipynb` notebook file.

### Run

To run the program from the command line, use:

```bash
python train.py
```  
or
```bash
python predict.py
```

The list of arguments can be seen after running the command. The commands can also be run without specifying any argument, as all arguments already have a default.

### Data

The dataset is created by Maria-Elena Nilsback and Andrew  Zisserman, available in [this link](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html). It includes 102 categories of flowers, each containing 40 to 258 images. The images have large scale, pose and light variations. In addition, there are categories that have large variations within the category and several very similar categories. The dataset is visualized using isomap with shape and colour features.

### train.py's features
According to Udacity's rubric:
- `train.py` successfully trains a new network on a dataset of images
- The training loss, validation loss, and validation accuracy are printed out as a network trains
- The training script allows users to choose from at least two different architectures available from torchvision.models
- The training script allows users to set hyperparameters for learning rate, number of hidden units, and training epochs
- The training script allows users to choose training the model on a GPU

### predict.py's features
According to Udacity's rubric:
- The `predict.py` script successfully reads in an image and a checkpoint then prints the most likely image class and it's associated probability
- It allows users to print out the top K classes along with associated probabilities
- It allows users to load a JSON file that maps the class values to other category names
- It allows users to use the GPU to calculate the predictions
