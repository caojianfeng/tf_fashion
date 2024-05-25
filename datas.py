#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv


def save_datas(indexes, accs, losses, tofile):
    columns = ['epochs', 'accs', 'losses']
    with open(tofile, 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(columns)
        for _index, _acc, _loss in zip(indexes, accs, losses):
            writer.writerow([_index, _acc, _loss])


def load_datas(fromfile):
    epochs = []
    accs = []
    losses = []

    with open(fromfile, 'r') as infile:
        reader = csv.reader(fromfile)
        for epoch, acc, loss in reader:
            epochs.append(epoch)
            accs.append(acc)
            losses.append(loss)

    return epochs, accs, losses
