# -*- coding: utf-8 -*-

# Import dependencies.
from ast import literal_eval  # For evaluating string literals as Python expressions
import csv # Manipulating csv files
import numpy as np  # Numerical computing library
import os  # Operating system interface
from pandas import read_csv  # Data manipulation and analysis library
from shutil import copyfile  # File copying utility
from time import time  # Time-related functions
import torch  # Deep learning library
from torch.nn import Module, Linear, Sigmoid, MSELoss, ReLU, Tanh  # Neural network components
from torch.optim import SGD  # Stochastic gradient descent optimizer
from torch.utils.data import Dataset, DataLoader, random_split  # Data loading and handling


t1 = time()

# Define the dataset.
class CSVDataset(Dataset):
    """
    Dataset class for loading a CSV file.
    
    Arguments
    ---------
    path : str
        String representing the path to the CSV file.
        
    Attributes
    ----------
    X : (torch.Tensor)
        The input data.
    y : (torch.Tensor)
        The target data.
        
    Methods
    -------
    __len__() : Returns the number of rows in the dataset.
    __getitem__(idx) : Returns a single data sample at the given index.
    get_splits(n_test=0.2) : Splits the dataset into train and test sets.
    
    """
    
    def __init__(self, path):
        """
        Initialize the CSVDataset.
        
        Parameters
        ---------
        path : str 
            The path to the CSV file.
        
        """
        df = read_csv(path)
        # store the inputs and outputs
        self.X = df.values[:, :3]
        self.y = df.values[:, -2:]
        # convert data from strings to lists
        for i, el in enumerate(self.y):
            for j in range(2):
                self.y[i][j] = literal_eval(self.y[i][j])
                self.y[i][j] = torch.tensor([float(val) for val in self.y[i][j]], dtype=torch.float32)
    
        # stack the elements of self.y
        self.y = torch.stack([torch.stack([self.y[i][0], self.y[i][1]]) for i in range(len(self.y))])
    
        # ensure input data is floats
        self.X = self.X.astype(np.float32)
    
    
    # number of rows in the dataset
    def __len__(self):
        """
        Returns the number of rows in the dataset.
        
        Returns
        -------
        int 
            The number of rows in the dataset.
        
        """
        return len(self.X)
    
    
    # get a row at an index
    def __getitem__(self, idx):
        """
        Returns a single data sample at the given index.

        Parameters
        ----------
        idx : int
            The index of the data sample.

        Returns
        -------
        tuple 
            A tuple containing the input data and the target data.

        """
        return self.X[idx], self.y[idx]
    
    
    # get indexes for train and test rows
    def get_splits(self, n_test=0.2):
        """
        Splits the dataset into train and test sets.
        
        Parameters
        ----------
        n_test : float
            The proportion of data to be used for testing (default = 0.2).
        
        Returns
        -------
        tuple
            A tuple containing the training dataset and the testing dataset.
        
        """
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])
    
    
# Definition of the model.
class MLP(Module):
    """
    Multilayer Perceptron (MLP) model for regression.
    
    Arguments
    ----------
    n_inputs : int
        Integer representing the number of input features.
        
    Attributes
    ----------
    hidden1 : (torch.nn.Linear)
        The first hidden layer.
    act1 : (torch.nn.Sigmoid)
        The activation function for the first hidden layer.
    hidden2 : (torch.nn.Linear) 
        The second hidden layer.
    act2 : (torch.nn.ReLU)
        The activation function for the second hidden layer.
    hidden3 : (torch.nn.Linear)
        The third hidden layer.
    act3 : (torch.nn.Sigmoid)
        The activation function for the third hidden layer.
    hidden4 : (torch.nn.Linear)
        The fourth hidden layer.
    act4 : (torch.nn.Sigmoid)
        The activation function for the fourth hidden layer.
    hidden5 : (torch.nn.Linear)
        The fifth hidden layer and output layer.
    act5 : (torch.nn.Sigmoid)
        The activation function for the fifth hidden layer and output layer.

    Methods
    -------
    forward : X 
        Forward propagates the input through the layers.


    """
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 10)
        self.act1 = Tanh()
        # second hidden layer
        self.hidden2 = Linear(10, 500)
        self.act2 = Tanh()
        # third hidden layer
        self.hidden3 = Linear(500, 3000)
        self.act3 = Tanh()
        # fourth hidden layer
        self.hidden4 = Linear(3000, 2000)
        self.act4 = Tanh()
        # fifth hidden layer and output
        self.hidden5 = Linear(2000, 2*512) 
        self.act5 = Tanh()

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer 
        X = self.hidden3(X)
        X = self.act3(X)
        # fourth hidden layer 
        X = self.hidden4(X)
        X = self.act4(X)
        # fifth hidden layer and output
        X = self.hidden5(X)
        X = self.act5(X)
        return X


# Prepare the dataset.
def prepare_data(path, batch_size):
    """
    Loads the data and splits it into testing and training data.

    Parameters
    ----------
    path : str
        String containing the path of the dataset.
    batch_size : int
        Integer indicating the batch size.

    Returns
    -------
    train_dl : (torch.utils.data.dataloader.DataLoader)
        DataLoader object that provides the training data.
    test_dl : (torch.utils.data.dataloader.DataLoader)
        DataLoader object that provides the testing data.

    """
    dataset = CSVDataset(path)
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test, batch_size=2000, shuffle=False)
    return train_dl, test_dl


# Train the model.
def train_model(train_dl, model, num_epochs, device, learning_rate):
    """
    Trains the model using the training data set.

    Parameters
    ----------
    train_dl : (torch.utils.data.dataloader.DataLoader)
        DataLoader object that provides the training data.
    model : MLP
        The MLP model to be trained.
    num_epochs : int
        Integer indicating the number of epochs in training.
    device : str
        String indicating whether code is to be run on cpu or gpu.
    learning_rate : int
        Integer representing the learning rate to beused for training.

    Returns
    -------
    None.

    """
    # define the optimization
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # enumerate epochs
    for epoch in range(num_epochs):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl): 
            # clear the gradients
            optimizer.zero_grad()
            
            # get data to cuda if possible (use GPU if possible)
            inputs = inputs.to(device=device)
            targets = targets.to(device=device)
            
            # compute the model output (forwards step)
            yhat = model(inputs)
            yhat = yhat.view(yhat.size(0), 2, 512)
            
            # calculate loss
            loss = criterion(yhat, targets)
            
            # credit assignment (backwards step)
            loss.backward()
            
            # update model weights (gradient descent)
            optimizer.step()


# Evaluate the model.
def evaluate_model(test_dl, model):
    """
    Evaluates the model by calculating the mean squared error
    and returning it.

    Parameters
    ----------
    test_dl : (torch.utils.data.dataloader.DataLoader)
        DataLoader object that provides the testing data.
    model : MLP
        The trained model that is to be evaluated.

    Returns
    -------
    float 
        Returns the mean squared error.

    """
    criterion = MSELoss()
    predictions, actuals = list(), list()
    with torch.no_grad():
        for inputs, targets in test_dl:
            # get data to cuda if possible (use GPU if possible)
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # compute the model output
            yhat = model(inputs)
            yhat = yhat.view(yhat.size(0), 2, 512)
            
            # append predictions and actuals
            predictions.extend(yhat.tolist())
            actuals.extend(targets.tolist())
        
        # calculate the mean squared error
        mse = criterion(torch.tensor(predictions), torch.tensor(actuals))
        return mse.item()


def save_copy():
    """
    Creates a copy of the current file. 
    The new file name is entered by user.
    Returns the new file name

    Returns
    -------
    filename : str
        Returns the new file name inputted by user.
    """
    # source file path
    current_file_path = os.path.abspath(__file__)
    
    # get the parent directory
    parent_directory = os.path.dirname(current_file_path) + "\configs"
    
    # input file name
    filename = input("Enter file name: ")
    
    # destination file path
    destination_file = parent_directory + f"\{filename}.py"
    
    # copy the code file
    copyfile(current_file_path, destination_file)
    
    print("Code copy saved successfully.")

    return filename


def save_run_info(mse, n_inputs, learning_rate, batch_size, num_epochs, filename=os.path.basename(__file__)):
    row = [mse, n_inputs, learning_rate, batch_size, num_epochs, filename]
    path = os.path.dirname(__file__) + "/configs/runs.csv" 
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row) 
        

def leaderboard():
    # implement in new py file
    ...



# Set device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters.
n_inputs = 3
learning_rate = 0.001
batch_size = 32
num_epochs = 1

# Get train and test data.
path = "test_train_data.csv"
train_dl, test_dl = prepare_data(path, batch_size)
t2 = time()
print(f"Loading data: {t2-t1} secs")    

# Define the network.
model = MLP(n_inputs).to(device)

# Train the model.
train_model(train_dl, model, num_epochs, device, learning_rate)
t3 = time()
print(f"Training the model: {t3-t2} secs")        

# Evaluate the model.
mse = evaluate_model(test_dl, model)
t4 = time()
print(f"Evaluating the model: {t4-t3} secs")    
print(f"Overall time: {time()-t1} secs")
print(f"Mean Squared Error: {mse:.4f}")



# Saving code copy.
while True:
    ask = input("Save copy (y/n)? ")
    if ask.lower() == "y" or ask.lower() == "yes":
        filename = save_copy()
        save_run_info(mse, n_inputs, learning_rate, batch_size, num_epochs,filename)
        break
    elif ask.lower() == "n" or ask.lower() == "no":
        save_run_info(mse, n_inputs, learning_rate, batch_size, num_epochs)
        break

