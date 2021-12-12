from typing import NamedTuple
from matplotlib import pyplot as plt
import argparse
import re
import numpy as np


# Parse loss from a file, assuming the style of checkpoints/loss_log.txt but with only a single run in it with no headers
def parse_loss(file):
    # regex pattern
    p = r'\(epoch: (\d+), iters: (\d+), time: (\S+), data: (\S+)\) D_A: (\S+) G_A: (\S+) cycle_A: (\S+) idt_A: (\S+) D_B: (\S+) G_B: (\S+) cycle_B: (\S+) idt_B: (\S+)\s*'

    with open(file) as f:
        # get a list of lists of all parsed out fields
        m = re.findall(p, f.read())
        # transpose the list of lists to get them by column
        m = np.array(m).T.astype(np.float)
        # take each list and put them into a dict with the name as the key
        labels = ['epoch', 'iters', 'time', 'data', 'D_A', 'G_A',
                  'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        lossDict = {k: v for k, v in zip(labels, m)}
        return lossDict


# Take Dictionary of Losses from parse_loss and save graph files
def graph_loss(ld, args):

    # calculate X-axis value combining epoch and iter
    x = ld['epoch'] + ld['iters'] / np.max(ld['iters'])

    plt.figure()
    plt.title(f'{args.type} Graph')
    plt.xlabel(f'Epochs')
    plt.ylabel(f'{args.type}ing Loss')
    # Graph field we care about
    for y in ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']:
        plt.plot(x, ld[y], label=y)

    plt.grid()
    plt.legend()
    plt.savefig(f'graphs/{args.outputFile}.png')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Parse a single run from checkpoints/loss_log.txt')
    parser.add_argument('file', type=str, help='input file')
    parser.add_argument('--type', default='Train',
                        help='train or test(graph labeling purposes only')
    parser.add_argument('--outputFile', default='train_graph',
                        type=str, help='output file name(no extension)')
    args = parser.parse_args()

    lossDict = parse_loss(args.file)
    graph_loss(lossDict, args)
