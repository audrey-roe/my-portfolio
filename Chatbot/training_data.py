import json
from nltk_utilities import tokenize, stem, bag_of_words
import numpy as np

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models import NeuralNetwork

with open('intents.json', 'r') as f: # opens the file intents.json in 'r' read mode in f
    intents = json.load(f) # loads the intents file stored in f

all_words = [] # creating an empty list for all the words
tags = [] # creates and empty list for all the tags in intents.json file
xy = [] # will later hold patterns and tags
for intent in intents ['intents']: # loop over the intents.json
    tag = intent['tag'] # this is stating that this is the tag key from the json file
    tags.append(tag) # this appends to the tags list earlier created
    for pattern in intent['patterns']: #looping over patterns key in the intent.json file
        w = tokenize(pattern) #tokenize the pattern
        all_words.extend(w) # put into the all words array using extend not append. because w is an array already so that we dont put array of w into array of all words instead we extend
        xy.append((w, tag)) #this is a tuple, it will know the pattern and the corresponing tags

ignore_words = ['?', ',', '!', '.']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))  # to remove duplicate elements 
tags = sorted(set(tags)) # to remove duplicate elements 

X_train = []
Y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    Y_train.append(label) #1 hot encoded vector? or CrossEntropyLoss?

X_train = np.array(X_train) # converts it to a numpy array
Y_train = np.array(Y_train) # converts it to a numpy array

 # creating a pytorch data set from the training data for iteration and perform batch training

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train) # we store the lenght of x train in the number of samples
        self.x_data = X_train # we store the data of x train 
        self.y_data = Y_train # we store the data of x train 

    #dataset[idx]; this is so that we can later access the data set using index
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index] # return as a tuple

    def __len__ (self):
        return self.n_samples

# hyperparameters
batch_size = 8 # for batch training
hidden_size = 8 #this value can be changed but the input and output sizes must remain the same
output_size = len(tags) # this is the length of classess or tags we have
input_size = len(X_train[0]) # this is the number of each bag of words (length of all words (x train))
learning_rate = 0.001
num_epochs = 1000 # can try out different values

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0) #num of workers is for multi threading or multi processing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # to check for gpu is avaiable otherwise we use the cpu
model = NeuralNetwork(input_size, hidden_size, output_size).to(device) # push the model to device if it is available

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #lr is the learning rate defined in the hyperparameter 

# training loop
for epoch in range(num_epochs):
    for (words, labels) in train_loader: # unpacking the words and the labels into the training loader   
        words = words.to(device) # push it to the device
        labels = labels.to(device) 

        # forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backward pass and optimizer step
        optimizer.zero_grad()    # first empty the gradients
        loss.backward()     # then call the loss to calculate the back propagartion
        optimizer.step()    #for optimizing
    
    if (epoch +1) % 100 == 0: #for every 100 steps then
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}') # to print the current epoch (epoch+1) and all 
                                                                     # epochs(num_epoch) then the loss for all items
                                                                     # .4f => formatting to print for 4 decimal values

print(f'final loss, loss={loss.item():.4f}')                      # print the final loss 

# creating a dictionary to save different things
data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags": tags
        }    


FILE = "data.pth" #pth => pytorch
torch.save(data, FILE) #serialize and save to a pickeled

print('training complete, file saved to {FILE}')
