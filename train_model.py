import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from model import Net


torch.manual_seed(777)
np.random.seed(777)


def main():
    # hyper-params
    epochs = 15
    batch_size = 64
    lr = 1e-3
    in_channels = 1
    num_fmaps = 32

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # transformation for images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # loading Fashion MNIST dataset
    training_data = datasets.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=transform
    )
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    print(
        f'\nnumber of training samples: {len(training_data)}, '
        f'number of batches: {len(train_loader)}'
    )

    # creating model
    model = Net(in_channels, num_fmaps).to(device)
    print(f'\n\n{model}\n\n')

    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.99, 0.999))
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    fig, ax_loss = prepare_plot()
    loss_line = None
    losses = []

    expr_dir = Path(f'logs/expr_{round(time.time())}')
    expr_dir.mkdir(parents=True)
    # create a tensorboard summary writer
    writer = SummaryWriter(expr_dir)
    # add model graph to logs
    dataiter = iter(train_loader)
    data, labels = dataiter.next()
    writer.add_graph(model, data.to(device))
    writer.add_text('Model', model.__repr__(), 1)
    writer.flush()
    plot_sample_images(data, labels)

    # training loop
    for e in range(epochs):
        targets = []
        predictions = []
        print(f'\nepoch #{e + 1}:')

        for batch_idx, (data, labels) in enumerate(train_loader):
            out = model(data.to(device))
            # calc. loss
            loss = loss_fn(out, labels.to(device))
            # optimize model
            optim.zero_grad()
            loss.backward()
            optim.step()

            targets.append(labels)
            predictions.append(out.cpu())

            # add to tensorboard
            writer.add_scalar(
                'Train Loss', loss.item(),
                e * len(train_loader) + batch_idx + 1,
                new_style=True
            )
            writer.flush()

            # print some logs and update loss plot
            if batch_idx == 0 or (batch_idx + 1) % 50 == 0:
                print(f'\tbatch #{batch_idx + 1},\tloss: {loss.item():.5f}')
                losses.append(loss.item())
                # plot loss
                if loss_line:
                    loss_line[0].remove()
                loss_line = ax_loss.plot(losses, color='dodgerblue', lw=1.5, label='Train Loss')
                ax_loss.set_xlim(left=0., right=len(losses))
                # refresh figure
                fig.canvas.draw()
                fig.canvas.flush_events()
                # add log to tensorboard
                writer.add_text(
                    'Train Loss',
                    f'\nepoch #{e + 1}, batch #{batch_idx + 1},\tloss: {loss.item():.5f}',
                    e * len(train_loader) + batch_idx + 1
                )
        # end of epoch
        # calc. epoch accuracy
        targets = torch.cat(targets, dim=0)
        predictions = torch.cat(predictions, dim=0)
        _, predictions = torch.max(predictions, dim=1)
        corrects = (predictions == targets).sum()

        print(f'\taccuracy: {corrects / len(targets): .2%}')
        writer.add_scalar(
            f'Epoch {e + 1} Accuracy', corrects * 100 / len(targets),
            e + 1, new_style=True
        )
        writer.flush()
    # end of training

    writer.close()

    # saving the trained model
    torch.save(model.state_dict(), expr_dir.joinpath(f'./model_e{epochs}.pt'))
    fig.savefig(expr_dir.joinpath('train_loss.png'))


def prepare_plot():
    fig = plt.figure(figsize=(14, 5.7), tight_layout=True)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title('Train Loss')
    ax1.grid(alpha=0.7)
    plt.show(block=False)

    return fig, ax1


def plot_sample_images(imgs, labels):
    fig = plt.figure(figsize=(11, 9), tight_layout=True)
    ax1 = fig.add_subplot(1, 1, 1)
    grid = make_grid(nrow=8, tensor=imgs)
    print(f'image tensor: {imgs.shape}')
    print(f'class labels: {labels}\n')
    ax1.imshow(np.transpose(grid, axes=(1, 2, 0)), cmap='gray')
    ax1.set_title('Image Samples')
    plt.show(block=False)




if __name__ == '__main__':
    main()
