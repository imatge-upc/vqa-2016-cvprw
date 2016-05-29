import argparse
import sys

import h5py
import matplotlib.pyplot as plt

sys.path.append('..')

from vqa.model.library import ModelLibrary

# ------------------------------ GLOBALS ------------------------------

DEFAULT_MODEL = max(ModelLibrary.get_valid_model_nums())
PLOT_TYPES = ['epochs', 'batches']


# ------------------------------- SCRIPT FUNCTIONALITY -------------------------------

def main(model_num, plot_type):
    with h5py.File('../results/train_losses_{}.h5'.format(model_num)) as f:
        losses = f['/train_losses'].value

    # Save plot
    plt.ioff()
    if plot_type == 'epochs':
        plt.plot(losses[::19403])
        plt.xlabel('Epoch number')
    elif plot_type == 'batches':
        plt.plot(losses)
        plt.xlabel('Batch number')
    else:
        raise ValueError('Plot type {} does not exist'.format(plot_type))
    plt.ylabel('Loss')
    plt.title('Loss curve for the training set')
    plt.savefig('../results/train_losses_{}_{}.png'.format(plot_type, model_num))


# ------------------------------- ENTRY POINT -------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Main entry point to interact with the VQA module')
    parser.add_argument(
        '-m',
        '--model',
        type=int,
        choices=ModelLibrary.get_valid_model_nums(),
        default=DEFAULT_MODEL,
        help='Specify the model architecture to interact with. Each model architecture has a model number associated.'
             'By default, the model will be the last architecture created, i.e., the model with the biggest number'
    )
    parser.add_argument(
        '-t',
        '--type',
        choices=PLOT_TYPES,
        default='epochs',
        help='Specify the plot type, i.e., the plot content'
    )
    # Start script
    args = parser.parse_args()
    main(args.model, args.type)
