# Imports here
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict

from PIL import Image

from train import build_model, device_agnostic

import argparse
import os
import json


# Get arguments from command line
def get_arguments():
    # Initiate parser
    parser = argparse.ArgumentParser(description='Parameters for predicting image category using a pre-trained deep learning model')

    # Define arguments:
    parser.add_argument('--image_path', type=str, default='flowers/train/1/image_06734.jpg', help='The directory to an image to be processed by this model')
    parser.add_argument('--top_k', type=int, default='1', help='The number of top classes to return')
    parser.add_argument('--category_name', type=str, default='cat_to_name.json', help='The directory to a dictionary linking category number to category name')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint.pth', help='The directory to an existing checkpoint of a pre-trained model')
    parser.add_argument('--gpu', action='store_true', default=False, help='Turn on inference on GPU - defaults to False')

    args = parser.parse_args()
    print(args)
    return args

def load_model(checkpoint_path):
    ''' Load the model from checkpoint
    '''
    checkpoint = torch.load(checkpoint_path)

    # Get model from checkpoint
    model = build_model(architecture=checkpoint['model'])

    ## Build classifier architecture
    # Define first layer
    architecture = OrderedDict([
        ('fc1', nn.Linear(checkpoint['input'], checkpoint['hidden'][0])),
        ('re1', nn.ReLU()),
        ('dr1', nn.Dropout(p=checkpoint['p_drop'])),
    ])
    ## If 1+ hidden layers
    if len(checkpoint['hidden']) > 1:
        # Define the hidden layer(s)
        for index, layer_size in enumerate(zip(checkpoint['hidden'][:-1], checkpoint['hidden'][1:])):
            architecture.update({'fc{}'.format(index+2): nn.Linear(layer_size[0], layer_size[1])})
            architecture.update({'re{}'.format(index+2): nn.ReLU()})
            architecture.update({'dr{}'.format(index+2): nn.Dropout(p=p_drop)})

        # Define the last layer
        architecture.update({'fc{}'.format(index+3): nn.Linear(checkpoint['hidden'][-1], checkpoint['output'])})
        architecture.update({'log': nn.LogSoftmax(dim=1)})
    ## If no hidden layers
    if len(checkpoint['hidden']) == 1:
        architecture.update({'fc2': nn.Linear(checkpoint['hidden'][0], checkpoint['output'])})
        architecture.update({'log': nn.LogSoftmax(dim=1)})

    classifier = nn.Sequential(architecture)
    model.classifier = classifier

    # Load state dict
    model.load_state_dict(checkpoint['state_dict'])

    # Load classifier
    optimizer = optim.SGD(params=model.classifier.parameters(), lr=checkpoint['lr'], momentum=checkpoint['momentum'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    # Opening
    im = Image.open(image_path)
    # Resizing
    im.thumbnail((256, 256))

    # Center cropping
    width, height = im.size
    new_width = 224
    new_height = 224

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    im = im.crop((left, top, right, bottom))

    # Converting PIL.Image(min=0, max=255) to NumPy.ndarray(min=0, max=1)
    np_im = np.array(im) / 255

    # Normalizing
    mean = np.array([0.485, 0.456, 0.406])
    stdev = np.array([0.229, 0.224, 0.225])
    np_im = (np_im - mean) / stdev

    # Transposing
    np_im = np_im.transpose((2, 0, 1))
    return np_im

def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Implement the code to predict the class from an image file
    im = process_image(image_path)
    tensor_im = torch.from_numpy(im).type(torch.FloatTensor)

    tensor_im, model = tensor_im.to(device), model.to(device)

    output = torch.exp(model(tensor_im.unsqueeze(0)))

    probs, labels = output.topk(topk)
    probs = probs.tolist()[0]
    labels = labels.tolist()[0]
    return probs, labels

def main():
    # Get args from command line
    args = get_arguments()

    # Device
    device = device_agnostic(args.gpu)

    # Get checkpoint from checkpoint path
    checkpoint = torch.load(args.checkpoint_path)

    # Get cat_to_name
    with open(args.category_name, 'r') as f:
        cat_to_name = json.load(f)

    # Get idx_to_class
    idx_to_class = {val: key for key, val in checkpoint['class_to_idx'].items()}

    # Load model and optimizer from checkpoint.pth
    model, optimizer = load_model(args.checkpoint_path)

    # Predict
    probs, labels = predict(args.image_path, model, device, topk=args.top_k)
    flowers = [cat_to_name[idx_to_class[label]] for label in labels]

    for index, flower in enumerate(flowers):
        print('The class that is #{} likely is: {}..\n    with a probability of {:.2%}\n'.format(index+1, flower, probs[index]))
    return probs, labels, flowers

# Runs script only from command line
if __name__ == '__main__':
    main()
