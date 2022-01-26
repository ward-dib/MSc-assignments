from torchvision import datasets, transforms
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import math
import torch as to
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import seaborn; seaborn.set()

# sklearn modules
from sklearn import model_selection

# define a class for the forward definition
# nn.Module is inherited
# Names matter. Extensive use of scoping in PyTorch means
# specifc names have to be used
# A function forward has to be defined for the forward pass
# capitalise the f and it will not work

class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        
        # this defines the forward propagation
        # inputs are propagated to 64 nodes of the 
        # first hidden layer
        self.linear1 = nn.Linear(784, 64)
        # propagation from the 1st hidden layer to the 2nd
        self.linear2 = nn.Linear(64, 64)
        # from the 2nd to the 3rd
        self.linear3 = nn.Linear(64, 64)
        # from the 3rd to the output layer
        self.linear4 = nn.Linear(64, 10)
        # sigmoid function for the output
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        # after every propagation ReLU is used to get rid
        # of negative weights
        y_pred = self.linear1(x)
        y_pred = nn.ReLU()(y_pred)
        y_pred = self.linear2(y_pred)
        y_pred = nn.ReLU()(y_pred)
        y_pred = self.linear3(y_pred)
        y_pred = nn.ReLU()(y_pred)
        y_pred = self.linear4(y_pred)
        # convert the output using the sigmoid function
        y_pred = self.softmax(y_pred)
        return y_pred
    
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
# Download the MNIST set and store it in a subfolder.

trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)


# Limit to 3000 samples.

trainset, _ = model_selection.train_test_split(trainset, train_size = 3000)

# Create loaders allowing to process the images in batches.
# Shuffle determines whether data is shuffled or not.
# False is the default.

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()

#Print(images.shape)
#print(labels.shape)

plt.imshow(images[0].numpy().squeeze(), cmap='gray_r');

figure = plt.figure()
num_of_images = 60
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
    

#################network 1#################
# Start of the optimisation.

# Start the training loop over 50 iterations.
losses = np.zeros((20,50))
iter = []
nbatches = 0

criterion = nn.NLLLoss(reduction = 'sum')
for i in range(20):
    model = network()
    optimizer = to.optim.SGD(model.parameters(), lr = 1e-4, momentum = 0.5)
    for j in range(50):
        add_loss = 0
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)
            y_pred = model(images)
            loss = criterion(y_pred, labels)
            add_loss = add_loss + loss.item()
            nbatches = nbatches + images.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses[i,j] = add_loss
    print(i)

#print(losses)
mean = np.mean(losses, axis=0)
#print(mean)
mean.shape
std = np.std(losses, axis = 0)
#print(std)

# Plot losses as function of iteration number.
plt.plot(losses[0]);


#################network 2#################
# Start of the optimisation.

# Start the training loop over 50 iterations.
losses = np.zeros((20,100))
iter = []
nbatches = 0

criterion = nn.NLLLoss(reduction = 'sum')
for i in range(20):
    model = network()
    optimizer = to.optim.SGD(model.parameters(), lr = 1e-4, momentum = 0.5)
    for j in range(100):
        add_loss = 0
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)
            y_pred = model(images)
            loss = criterion(y_pred, labels)
            add_loss = add_loss + loss.item()
            nbatches = nbatches + images.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses[i,j] = add_loss
    print(i)

#print(losses)
mean = np.mean(losses, axis=0)
#print(mean)
mean.shape
std = np.std(losses, axis = 0)
#print(std)

# Plot losses as function of iteration number.
plt.plot(losses[0]);



#################network 3#################
# Start of the optimisation.

# Start the training loop over 50 iterations.
losses = np.zeros((20,80))
iter = []
nbatches = 0

criterion = nn.NLLLoss(reduction = 'sum')
for i in range(20):
    model = network()
    optimizer = to.optim.SGD(model.parameters(), lr = 1e-3, momentum = 0.4)
    for j in range(80):
        add_loss = 0
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)
            y_pred = model(images)
            loss = criterion(y_pred, labels)
            add_loss = add_loss + loss.item()
            nbatches = nbatches + images.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses[i,j] = add_loss
    print(i)

#print(losses)
mean = np.mean(losses, axis=0)
#print(mean)
mean.shape
std = np.std(losses, axis = 0)
#print(std)

# Plot losses as function of iteration number.
plt.plot(losses[0]);

















