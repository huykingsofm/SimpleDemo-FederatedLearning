import os
import torch
from main import Client, MODEL


def f(X):
    W = torch.Tensor([1, 2]).reshape(2, -1)
    b = -3
    return torch.mm(X ** 2, W) + b + torch.randn((len(X), 1))


N = torch.randint(30, 50, (1,))
DATA = torch.rand((N, 2)) * 10

if __name__ == "__main__":
    print("We have {} data points".format(len(DATA)))
    x = MODEL()
    print(x.state_dict())
    """ client = Client(
        server_address=("127.0.0.1", 1999),
        architecture= MODEL, 
        data= (DATA, f(DATA)),
        directory= os.sys.argv[1],
        verbosities= {
            "user": ["notification", "warning", "error"],
            "dev": ["debug", "error"]
            }
    )
    client.start() """
