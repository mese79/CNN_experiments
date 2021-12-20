import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score
)


def show_confusion_plot(targets: np.ndarray,
                        predictions: np.ndarray,
                        target_names: list,
                        block=True):

    num_classes = len(target_names)
    class_labels = [i for i in range(num_classes)]

    conf_mat = confusion_matrix(targets, predictions)
    print(f'\nconfusion matrix:\n{conf_mat}\n')

    report = classification_report(
        targets, predictions,
        labels=class_labels, target_names=target_names,
        zero_division=1
    )
    print(f'\nclassification report:\n{report}')

    # plot
    fig: plt.Figure = plt.figure(3, figsize=(13, 10), tight_layout=True)
    fig.canvas.set_window_title('Confusion Matrix Plot')
    ax: plt.Axes = fig.add_subplot(1, 1, 1)
    ax.set_title('Confusion Matrix\n', fontsize=10, fontweight='bold')

    n = 128
    colors = np.ones((n, 4))
    colors[:, 0] = np.linspace(236 / 256, 20 / 256, n)
    colors[:, 1] = np.linspace(240 / 256, 150 / 256, n)
    colors[:, 2] = np.linspace(210 / 256, 24 / 256, n)
    new_map = ListedColormap(colors)
    im = ax.imshow(conf_mat, cmap=new_map)
    fig.colorbar(im)

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(target_names)
    ax.set_yticklabels(target_names)
    ax.xaxis.tick_top()

    # turn spines off and create grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(num_classes + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(num_classes + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="thistle", linestyle='-', linewidth=3)

    # annotations
    threshold = conf_mat.max() // 2 + 50
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            color = 'white'
            if conf_mat[i, j] < threshold:
                color = 'black'
            ax.text(
                j, i, conf_mat[i, j],
                color=color, fontsize=9.5, fontweight='bold',
                horizontalalignment='center', verticalalignment='center'
            )
            # ax.text(
            #     j, i - 0.15, f'{conf_mat[i, j] / conf_mat.sum():.2%}',
            #     color=color, fontsize=7, fontweight='normal',
            #     horizontalalignment='center', verticalalignment='center'
            # )
            if (i == j):
                if conf_mat[:, j].sum() > 0:
                    precision = conf_mat[i, j] / conf_mat[:, j].sum()
                    recall = conf_mat[i, j] / conf_mat[i, :].sum()
                else:
                    precision = 0
                    recall = 0

                ax.text(
                    j, i + 0.35, f'p: {precision:.2%}\nr: {recall:.2%}',
                    color='#181395', fontsize=7, fontweight='semibold',
                    horizontalalignment='center', verticalalignment='center'
                )

    accuracy = conf_mat.trace() / conf_mat.sum()

    x = (num_classes / 2) - 1.5
    y = num_classes
    ax.text(
        x, y, f'overall accuracy: {accuracy:.2%}',
        fontsize=10, fontweight='medium', fontfamily='monospace',
        horizontalalignment='left', verticalalignment='bottom'
    )
    bal_acc = balanced_accuracy_score(targets, predictions)
    ax.text(
        x, y + 0.17, f'balanced accuracy: {bal_acc:.2%}',
        fontsize=10, fontweight='medium', fontfamily='monospace',
        horizontalalignment='left', verticalalignment='bottom'
    )

    plt.show(block=block)
    plt.pause(0.001)
