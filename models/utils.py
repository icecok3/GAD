import numpy as np
import matplotlib.pyplot as plt

def res_plot(data, label_true, label_pred, nrows, ncols, title=None):

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(8,8))
    axes = ax.flatten()
    for i in range(nrows*ncols):
        axes[i].imshow(data[i])
        axes[i].set_title("{} -> {}".format(label_true[i], label_pred[i]))
    if title is not None:
        plt.suptitle(title)
    plt.show()
