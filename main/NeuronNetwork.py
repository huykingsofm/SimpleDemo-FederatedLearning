import os
import copy
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class FLModule(nn.Module):
    def __init__(self):
        self.__version__ = 0
        super(FLModule, self).__init__()

    def nextVersion(self):
        self.__version__ += 1

    def checkVersion(self, filename):
        """
        checkVersion == True if the file has same version as current object \n
        checkVersion == False if the file has another version with current object
        """
        if os.path.isfile(filename):
            f = open(filename, "rb")
            version, _ = pickle.load(f)
            f.close()
            if version == self.__version__:
                return True
            return False
        else:
            return False

    def read(self, filename):
        f = open(filename, "rb")
        data = f.read()
        f.close()
        self.deserialize(data)

    def write(self, filename):
        f = open(filename, "wb")
        f.write(self.serialize())
        f.close()

    def serialize(self):
        obj = (self.__version__, self.state_dict())
        return pickle.dumps(obj)

    def deserialize(self, obj):
        self.__version__, W = pickle.loads(obj)
        self.load_state_dict(W)
 
    def averaging(self, models_list, scale_list = None):
        if scale_list == None:
            scale_list = [1/len(models_list)] * len(models_list)

        assert abs(sum(scale_list) - 1) < 1e-5
        
        # CREATE INITIAL WEIGHT
        # Get current weight architecture of model
        global_W = self.state_dict()
        
        # Set all elements of all tensors to 0s
        local_W0 = models_list[0].state_dict()
        for key in global_W.keys():
            global_W[key] -= global_W[key]

        # AVERAGING MODEL
        for local_model, scale in zip(models_list, scale_list):
            local_W = local_model.state_dict()
            for key in local_W.keys():
                global_W[key] += scale * local_W[key]

        self.load_state_dict(global_W)

class EncryptedModel(FLModule):
    def __init__(self, state_dict) -> None:
        self.__state_dict__ = state_dict

    def forward(self, data):
        raise Exception("Encrypted Model cannot forward")

    def serialize(self):
        return pickle.dumps((self.__version__, self.__state_dict__))

    def deserialize(self, obj):
        self.__version__, self.__state_dict__ = pickle.loads(obj)
        
    def state_dict(self):
        return self.__state_dict__

    def averaging(self, models_list, scale_list):
        raise Exception("Encrypted Model cannot averaging")

class SimpleNeuronNetwork(FLModule):
    def __init__(self):
        super(SimpleNeuronNetwork, self).__init__()
        self.__model__ = nn.Sequential(
            nn.Linear(in_features=2, out_features=40, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=40, out_features=30, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=30, out_features=20, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=1, bias=True)
        )

    def forward(self, input):
        return self.__model__(input)

class SimpleDataset(Dataset):
    def __init__(self, data: list, label: list) -> None:
        assert len(data) == len(label)
        self.__data__ = data
        self.__label__ = label
    

    def __len__(self) -> int:
        return len(self.__data__)

    def __getitem__(self, index: int):
        return {"X": self.__data__[index], "Y": self.__label__[index]}

def train(model: nn.Module, dataset: Dataset, loss_fn, optimizer, nepochs, batchsize, brief_print = False):
    dataloader = DataLoader(dataset, batchsize, shuffle= True)
    
    for i in range(nepochs):
        if brief_print:
            print("\r[{:3d}/{:3d}] ...".format(i + 1, nepochs), end = "")
        else:
            print("[{:3d}/{:3d}] ...".format(i + 1, nepochs), end = "")
        sum_loss = 0
        for batch in dataloader:
            # Set gradient to zero 
            # Because after calculating gradient by ``loss.backward()``, the gradient will be cummulative
            optimizer.zero_grad()

            # Predict the label of this batch
            Ypredict = model(batch["X"])
            
            # Calculate the loss function and the gradient
            loss = loss_fn(Ypredict, batch["Y"])
            loss.backward()

            # For printing the average loss of entire dataset
            sum_loss += loss.item()
            
            # Using optimizer to update weight of model
            optimizer.step()
        if brief_print:
            print("\r[{:3d}/{:3d}] Loss = {:.5f}".format(i + 1, nepochs, sum_loss/len(dataset)), end = "")
        else:
            print("\r[{:3d}/{:3d}] Loss = {:.5f}".format(i + 1, nepochs, sum_loss/len(dataset)))
    print()

def test(model, size):
    X = torch.rand(size) * 10
    print(X, "-->", model(X))