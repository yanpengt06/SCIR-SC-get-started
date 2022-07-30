
import torch

if __name__ == '__main__':
    a = torch.zeros(3)
    b = torch.ones(3)
    print(torch.stack([a,b],dim=0))

