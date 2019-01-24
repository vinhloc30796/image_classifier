# Imports here
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict

import argparse
import os
import json

# Get arguments from command line
def get_arguments():
    # Initiate parser
    parser = argparse.ArgumentParser(description='Parameters for training a deep learning model')

    # Define arguments: data_dir, arch, hidden, learning_rate, momentum, epochs, p_drop, gpu, save_directory
    parser.add_argument('--data_dir', type=str, default='flowers', help='The directory of training, testing, and validating data')
    parser.add_argument('--arch', type=str, default='vgg19_bn', choices=('vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn'), help='The base architecture of the model')
    parser.add_argument('--hidden', type=int, nargs='+', default=[8192, 4096, 1024], help='The sizes of the classifier\'s hidden layers')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='The learning rate of the optimizer when training')
    parser.add_argument('--momentum', type=float, default=0.5, help='The momentum of the optimizer when training')
    parser.add_argument('--epochs', type=int, default=5, help='The number of epochs through which the model will train')
    parser.add_argument('--p_drop', type=float, default=0.25, help='The probability that an input feature will be dropped in a layer while training')
    parser.add_argument('--gpu', action='store_true', default=False, help='Turn on training on GPU - defaults to False')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='The directory to save the model\'s checkpoint')

    args = parser.parse_args()
    print(args)
    return args

# Figure out which device to run the network on
def device_agnostic(gpu=True):
    if gpu:
        # Device-agnostic code
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print('Working on CUDA')
        else:
            device = torch.device('cpu')
            print('No CUDA available - defaulting to working on CPU')
    else:
        device = torch.device('cpu')
        print('Working on CPU')
    return device

# Define transformations on images from these directories
def process_data(train_dir, test_dir, valid_dir):
    # Define transforms for the training, validation, and testing sets
    # Train transforms include randomly rotating, scaling, cropping, and flipping
    # Then converting images to tensors and normalizing
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Test & Validate transforms include resizing and center-cropping
    # Then convertinng images to tensors and normalizing
    test_n_valid_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(root=test_dir, transform=test_n_valid_transforms)
    valid_data = datasets.ImageFolder(root=valid_dir, transform=test_n_valid_transforms)

    return train_data, test_data, valid_data

# Turn data into loader
def get_dataloader(train_data, test_data, valid_data):
    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)

    return trainloader, testloader, validloader

# Get the correct model from args.arch
def build_model(architecture='vgg19_bn'):
    # Get the corresponding model from torchvision.models
    if architecture == 'vgg11_bn':
        model = models.vgg11_bn(pretrained=True)
    if architecture == 'vgg13_bn':
        model = models.vgg13_bn(pretrained=True)
    if architecture == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=True)
    if architecture == 'vgg19_bn':
        model = models.vgg19_bn(pretrained=True)

    # Freeze the parameters
    for param in model.parameters():
        param.requires_grad = False

    return model

# Build the classifier to replace original classifier
def build_classifier(model, hidden=[8192, 4096, 1024], p_drop=0.25):
    n_input = model.classifier[0].in_features
    n_output = 102
    # Define first layer
    architecture = OrderedDict([
        ('fc1', nn.Linear(n_input, hidden[0])),
        ('re1', nn.ReLU()),
        ('dr1', nn.Dropout(p=p_drop)),
    ])
    ## If 1+ hidden layers
    if len(hidden) > 1:
        # Define the hidden layer(s)
        for index, layer_size in enumerate(zip(hidden[:-1], hidden[1:])):
            architecture.update({'fc{}'.format(index+2): nn.Linear(layer_size[0], layer_size[1])})
            architecture.update({'re{}'.format(index+2): nn.ReLU()})
            architecture.update({'dr{}'.format(index+2): nn.Dropout(p=p_drop)})

        # Define the last layer
        architecture.update({'fc{}'.format(index+3): nn.Linear(hidden[-1], n_output)})
        architecture.update({'log': nn.LogSoftmax(dim=1)})
    ## If no hidden layers
    if len(hidden) == 1:
        architecture.update({'fc2': nn.Linear(hidden[0], n_output)})
        architecture.update({'log': nn.LogSoftmax(dim=1)})

    classifier = nn.Sequential(architecture)
    print(classifier)
    return classifier

# Train the model
def train_model(model, optimizer, criterion, trainloader, validloader, device,
                lr=0.1, momentum=0.5, epochs=5):
    model = model.to(device)
    # Initialize optimizer and criterion
    optimizer = optim.SGD(params=model.classifier.parameters(), lr=lr, momentum=momentum)
    criterion = nn.NLLLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # Define some variables for the training process
    steps = 0
    print_every = 25

    for e in range(epochs):
        # Set running loss to 0
        running_loss = 0
        # Put model in training mode
        model.train()
        for image, label in trainloader:
            # Move image and label to CUDA if available
            image, label = image.to(device), label.to(device)

            # Feed image forward through model
            train_output = model(image)

            # Clear out optimizer
            optimizer.zero_grad()

            # Get training loss by comparing training output with label
            train_loss = criterion(train_output, label)

            # Feed training loss backward through model
            train_loss.backward()

            # Take one step for optimizer
            optimizer.step()
            steps += 1

            # Increment running loss
            running_loss += train_loss.item()

            # Evaluate and print results every 100 loops
            if steps % print_every == 0:
                # Put model in evaluation mode
                model.eval()

                # Turn off gradients for validation to speed up
                with torch.no_grad():
                    accuracy = 0
                    valid_loss = 0
                    for image, label in validloader:
                        # Move image and label to CUDA if available
                        image, label = image.to(device), label.to(device)

                        # Feed image forward through model
                        valid_output = model(image)

                        # Increment evaluation loss
                        valid_loss += criterion(valid_output, label).item()

                        # Accuracy
                        # Take exponential of log-softmax to get the probabilities ,
                        ps = torch.exp(valid_output)
                        # Class with highest probability is our predicted class, compare with true label
                        top_p, top_class = ps.topk(1, dim=1)
                        equality = (label.view(*top_class.shape) == top_class)
                        # Take the mean of equality to get accuracy
                        accuracy += equality.type_as(torch.FloatTensor()).mean()

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.2%}".format(accuracy/len(validloader)))

                scheduler.step(valid_loss)
    return model, optimizer, scheduler

# Use the model on test data
def test_model(model, criterion, testloader, device):
    with torch.no_grad():
        model = model.to(device)
        accuracy = 0
        test_loss = 0
        for image, label in testloader:
            # Move image and label to CUDA if available
            image, label = image.to(device), label.to(device)

            # Feed image forward through model
            test_output = model(image)

            # Increment evaluation loss
            test_loss += criterion(test_output, label).item()

            # Accuracy
            # Take exponential of log-softmax to get the probabilities,
            ps = torch.exp(test_output)
            # Class with highest probability is our predicted class, compare with true label
            top_p, top_class = ps.topk(1, dim=1)
            equality = (label.view(*top_class.shape) == top_class)
            # Take the mean of equality to get accuracy
            accuracy += equality.type_as(torch.FloatTensor()).mean()

        print("Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
              "Test Accuracy: {:.2%}".format(accuracy/len(testloader)))

def main():
    # Get cat_to_name
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # TO DO: Get arguments
    args = get_arguments()

    # Define the device
    device = device_agnostic(args.gpu)

    # Define the directories
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'
    valid_dir = data_dir + '/valid'

    train_data, test_data, valid_data = process_data(train_dir, test_dir, valid_dir)
    trainloader, testloader, validloader = get_dataloader(train_data, test_data, valid_data)

    # Get the correct model
    model = build_model(args.arch)

    # Put the classifier into the model
    classifier = build_classifier(model, args.hidden, args.p_drop)
    model.classifier = classifier

    # Define the criterion
    criterion = nn.NLLLoss()

    # Train model
    model, optimizer, scheduler = train_model(model, classifier, criterion, trainloader=trainloader, validloader=validloader, device=device,
                                               lr=args.learning_rate, momentum=args.momentum, epochs=args.epochs)

    # Test model
    test_model(model, criterion, testloader, device)

    # Save checkpoint
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'model': args.arch,
                  'input': model.classifier[0].in_features,
                  'output': 102,
                  'hidden': args.hidden,
                  'p_drop': args.p_drop,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'epoch': args.epochs,
                  'print_every': 25,
                  'class_to_idx': model.class_to_idx,
                  'lr': args.learning_rate,
                  'momentum': args.momentum,
                 }

    torch.save(checkpoint, args.save_dir)

    print('Training complete. Model saved in {}'.format(args.save_dir))

# Runs script only from command line
if __name__ == '__main__':
    main()
