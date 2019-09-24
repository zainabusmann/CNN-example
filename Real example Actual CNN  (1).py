
#imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models # add models to the list
from torchvision.utils import make_grid
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")



#test and train data transforms
train_transform = transforms.Compose([
        transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize(224),             # resize shortest side to 224 pixels so all are same
        transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
        transforms.ToTensor(),              # into tensors
        transforms.Normalize([0.485, 0.456, 0.406], #already knew
                             [0.229, 0.224, 0.225])
    ])

test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])



#grab data and put into folders and load
root = '../Data/CATS_DOGS' #top path first

train_data = datasets.ImageFolder(os.path.join(root, 'train'), transform=train_transform)
test_data = datasets.ImageFolder(os.path.join(root, 'test'), transform=test_transform)

torch.manual_seed(42) # can play around
train_loader = DataLoader(train_data, batch_size=10, shuffle=True) #can play with batch size
test_loader = DataLoader(test_data, batch_size=10, shuffle=True)

class_names = train_data.classes

print(class_names)
print(f'Training images available: {len(train_data)}') 
print(f'Testing images available:  {len(test_data)}')



# Grab the first batch of 10 images (both dogs and cats with rotations and at random)
                for images,labels in train_loader: 
                    break

                # Print the labels
                print('Label:', labels.numpy())
                print('Class:', *np.array([class_names[i] for i in labels]))

                im = make_grid(images, nrow=5)  # the default nrow is 8

                # Inverse normalize the images, know from before
                inv_normalize = transforms.Normalize(
                    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                    std=[1/0.229, 1/0.224, 1/0.225]
                )
                im_inv = inv_normalize(im)

                # Print the images
                plt.figure(figsize=(12,4))
                plt.imshow(np.transpose(im_inv.numpy(), (1, 2, 0)));



#define the model
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)  # input (rgb so 3), output, filter, stride , (padding opt)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)  # can play with filters alot (depending on time) but now input needs to be output of last
        self.fc1 = nn.Linear(54*54*16, 120) #calc for 54 (dependant on pooling layer). 224-2(padding) /2 pooling layer. another conv -2 and anothe pooling /2 and 54.5 so around off to 54. 16 is # of filters from last. can also play around with 120 neurons
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2) # classes are 2 so final output is 2

    def forward(self, X):
        X = F.relu(self.conv1(X)) #rec linear conv 1
        X = F.max_pool2d(X, 2, 2) # pooling layer 2x2 with stride 2. its common
        X = F.relu(self.conv2(X)) 
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 54*54*16) #now go into linear units, flatten out now. -1 cause we want to keep same batch size
        X = F.relu(self.fc1(X)) #rec linear being passed to fc
        X = F.relu(self.fc2(X))
        X = self.fc3(X)         #last one
        return F.log_softmax(X, dim=1)



#loss and optimization function set
torch.manual_seed(101) #101 is just random
CNNmodel = ConvolutionalNetwork() #created object and then the instance of conv network 
criterion = nn.CrossEntropyLoss() #cho0se cross entropy cause its classification
optimizer = torch.optim.Adam(CNNmodel.parameters(), lr=0.001) #Adam optimizer, 0.001 is usually good but can play around
CNNmodel



# number of parameters for CNN
for p in CNNmodel.parameters():
    print(p.numel())
    
    
#ps flattening out gives huge number of parameters 5m but rememeber with ANN would be even more



#training
import time #to see how long training time is taking
start_time = time.time()

epochs = 3torch.save(CNNmodel.state_dict(), 'CustomImageCNNModel.pt') #set epochs

#puts a limit to num of batch only for time purposes (optional)
max_trn_batch = 800  #each batach has 10 hence 8000 here out of 18k. set variables
max_tst_batch = 300

train_losses = [] #keeping trackers
test_losses = [] 
train_correct = []
test_correct = []

for i in range(epochs): #keeping trackers
    trn_corr = 0
    tst_corr = 0
    
    # run the training batches
    for b, (X_train, y_train) in enumerate(train_loader):
        
        # limit the number of batches (optional). kinda to test your data
        if b == max_trn_batch:
            break
        b+=1
        
        # apply the model
        y_pred = CNNmodel(X_train)
        loss = criterion(y_pred, y_train)
 
        # tally the number of correct predictions
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr
        
        # update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print interim results
        if b%200 == 0:  #at 200 print out following
            print(f'epoch: {i:2}  batch: {b:4} [{10*b:6}/8000]  loss: {loss.item():10.8f}  accuracy: {trn_corr.item()*100/(10*b):7.3f}%')

    train_losses.append(loss)
    train_correct.append(trn_corr)

    # Run the testing batches while training
    with torch.no_grad(): #dont update
        for b, (X_test, y_test) in enumerate(test_loader):
            # Limit the number of batches
            if b == max_tst_batch:
                break

            # Apply the model (grab predicted)
            y_val = CNNmodel(X_test)

            # Tally the number of correct predictions
            predicted = torch.max(y_val.data, 1)[1] 
            tst_corr += (predicted == y_test).sum()

    loss = criterion(y_val, y_test) #calc loss
    test_losses.append(loss)
    test_correct.append(tst_corr)

print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed




#always good to save the model for future since it took a lil time
torch.save(CNNmodel.state_dict(), 'CustomImageCNNModel.pt')


#evaluate model performance by plotting it
plt.plot(train_losses, label='training loss')
plt.plot(test_losses, label='validation loss')
plt.title('Loss at the end of each epoch')
plt.legend();


#evaluate at each epoch end too
plt.plot([t/80 for t in train_correct], label='training accuracy')
plt.plot([t/30 for t in test_correct], label='validation accuracy')
plt.title('Accuracy at the end of each epoch')
plt.legend();


#print accuracy
print(test_correct)
print(f'Test accuracy: {test_correct[-1].item()*100/3000:.3f}%') # last item


