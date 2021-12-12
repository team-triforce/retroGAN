from typing import NamedTuple
from matplotlib import pyplot as plt
import argparse
import re
import numpy as np

class lossParser:
    def __init__():
        self.epoch = np.array()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parse a single run from checkpoints/loss_log.txt')
    parser.add_argument('file', type=str, help='input file')
    parser.add_argument('--type', default='train', help='train or test(graph labeling purposes only')
    args = parser.parse_args()

    # pattern
    p = r'\(epoch: (?P<epoch>\d+), iters: (?P<iters>\d+), time: (?P<time>|\S+), data: (?P<data>\S+)\) D_A: (?P<D_A>\S+) G_A: (?P<G_A>\S+) cycle_A: (?P<cycle_A>\S+) idt_A: (?P<idt_A>\S+) D_B: (?P<D_B>\S+) G_B: (?P<G_B>\S+) cycle_B: (?P<cycle_B>\S+) idt_B: (?P<idt_B>\S+)\s*'

    with open(args.file) as file:
        m = re.findall(p, file.read())
        m = np.array(m)
        print(m.shape)
        print(m[0])
        m = m.T
        print(m.shape)
        print(m[0])
        # lines = file.readlines()
        # for line in lines:
        #     m = re.match(p, line)
        #     print(m.group('epoch'))

