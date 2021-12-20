import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import Net
from utils import show_confusion_plot


torch.manual_seed(777)
np.random.seed(777)


def main(model_file):
    # hyper-params
    in_channels = 1
    num_fmaps = 32

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # loading dataset
    test_data = datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=transform
    )
    test_loader = DataLoader(test_data, batch_size=100, shuffle=True)

    # load model
    model = Net(in_channels, num_fmaps).to(device)
    checkpoint = torch.load(model_file, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    targets = []
    predictions = []
    with torch.no_grad():
        for data, labels in test_loader:
            out = model(data.to(device)).cpu()
            predictions.append(out)
            targets.append(labels)

    # calc. test accuracy
    targets = torch.cat(targets, dim=0)
    predictions = torch.cat(predictions, dim=0)
    _, predictions = torch.max(predictions, dim=1)
    corrects = (predictions == targets).sum()
    print(f'\n\ntest accuracy: {corrects / len(targets): .2%}\n')

    target_names = test_data.classes
    show_confusion_plot(targets.numpy(), predictions.numpy(), target_names, True)







if __name__ == '__main__':
    model_file = './model_e15.pt'
    if len(sys.argv) > 1:
        model_file = sys.argv[1]

    main(model_file)
