import matplotlib.pyplot as plt
import numpy as np


def plot_search(energies, true_val=None, save_name=None):
    plt.figure()
    plt.ylabel('Energy')
    plt.xlabel('Iteration')
    plt.title('Ground State Energy vs. Iteration')

    if true_val is not None:
        plt.plot(range(len(energies)), energies, label='Predicted Value')
        plt.plot(range(len(energies)), [true_val for _ in range(len(energies))], dashes=[4, 2], label='Exact Value')
        plt.legend()
    else:
        plt.plot(range(len(energies)), energies)

    if save_name is not None:
        plt.savefig(save_name)

    plt.show()


def plot_search_error(energies, true_val, save_name=None):
    errors = np.abs(energies - true_val)

    plt.figure()
    plt.loglog(range(len(errors)), errors)
    plt.ylabel('Error')
    plt.xlabel('Iteration')
    plt.title('Ground State Energy Error vs. Iteration')
    if save_name is not None:
        plt.savefig(save_name)
    plt.show()


def plot_different_lrs(lrs, energies, true_val=None, save_name=None):

    plt.figure()
    for lr, energy in zip(lrs, energies):
        plt.plot(range(len(energy)), energy, label='lr = %0.1f' % lr)


    if true_val is not None:
        plt.plot(range(len(energies)), [true_val for _ in range(len(energies))], dashes=[4, 2], label='Exact Value')
    plt.legend()
    plt.ylabel('Energy')
    plt.xlabel('Iteration')
    plt.title('Ground State Energy vs. Iteration')
    if save_name is not None:
        plt.savefig(save_name)
    plt.show()


def plot_alpha_error(alphas, energies, true_val, save_name=None):
    errors = [np.abs(energy - true_val) for energy in energies]

    plt.figure()
    for alpha, error in zip(alphas, errors):
        plt.loglog(range(len(error)), error, label='$\\alpha = %d$' % alpha)
    plt.legend()
    plt.ylabel('Error')
    plt.xlabel('Iteration')
    plt.title('Ground State Energy Error vs. Iteration')
    if save_name is not None:
        plt.savefig(save_name)
    plt.show()

def plot_correlations(correlations, save_name=None):

    correlations = np.array([np.roll(correlations, int(len(correlations)/2))])

    plt.figure(figsize=(4, 1))

    plt.imshow(correlations, cmap='RdBu', vmin=-1, vmax=1, interpolation='nearest', extent=[0, 5, 0, 1])

    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_ticklabels([])
    cur_axes.axes.get_yaxis().set_ticklabels([])
    cur_axes.axes.get_xaxis().set_ticks([])
    cur_axes.axes.get_yaxis().set_ticks([])

    if save_name is not None:
        plt.savefig(save_name)
    plt.show()

def plot_correlations_legend(save_name=None):

    values = np.array([np.linspace(-1, 1, 100)])
    plt.figure(figsize=(4, 1))

    plt.imshow(values, cmap='RdBu', vmin=-1, vmax=1, interpolation='bilinear', extent=[-1, 1, 0, 0.05])

    cur_axes = plt.gca()
    # cur_axes.axes.get_xaxis().set_ticks([0, 20])
    # cur_axes.axes.get_xaxis().set_ticklabels([])
    cur_axes.axes.get_yaxis().set_ticks([])
    cur_axes.axes.get_yaxis().set_ticklabels([])

    if save_name is not None:
        plt.savefig(save_name)
    plt.show()
