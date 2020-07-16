import matplotlib.pyplot as plt


def plot_history(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'], '')
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend([metric, f'val_{metric}'])
    plt.show()
