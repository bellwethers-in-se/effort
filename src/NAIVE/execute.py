from __future__ import print_function, division
import os
import sys

root = os.path.join(os.getcwd().split('src')[0], 'src/effort')
if root not in sys.path:
    sys.path.append(root)

import warnings
from prediction.model import logistic_model, rf_model
from py_weka.classifier import classify
from utils import *
from metrics.abcd import abcd
from metrics.recall_vs_loc import get_curve
from pdb import set_trace
import numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas
from plot.effort_plot import effort_plot
from tabulate import tabulate
from datasets.handler import get_all_datasets


def weight_training(test_instance, training_instance):
    head = training_instance.columns
    new_train = training_instance[head[:-1]]
    try:
        new_train = (new_train - test_instance[head[:-1]].mean()) / test_instance[head[:-1]].std()
    except:
        set_trace()
    new_train[head[-1]] = training_instance[head[-1]]
    new_train.dropna(axis=1, inplace=True)
    tgt = new_train.columns
    new_test = (test_instance[tgt[:-1]] - test_instance[tgt[:-1]].mean()) / (
        test_instance[tgt[:-1]].std())
    # set_trace()
    new_test[tgt[-1]] = test_instance[tgt[-1]]
    new_test.dropna(axis=1, inplace=True)
    columns = list(set(tgt[:-1]).intersection(new_test.columns[:-1]))+[tgt[-1]]
    return new_train[columns], new_test[columns]



def predict_defects(train, test):
    actual = test[test.columns[-1]].values
    predicted = rf_model(train, test)
    return actual, predicted


def bellw(source, target, n_rep=12):
    """
    TNB: Transfer Naive Bayes
    :param source:
    :param target:
    :param n_rep: number of repeats
    :return: result
    """
    result = dict()
    for tgt_name, tgt_path in target.iteritems():
        stats = []
        print("{} \r".format(tgt_name[0].upper() + tgt_name[1:]))
        for src_name, src_path in source.iteritems():
            if not src_name == tgt_name:
                src = pandas.read_csv(src_path)
                tgt = pandas.read_csv(tgt_path)

                for _ in xrange(n_rep):
                    # _train, __test = weight_training(test_instance=tgt, training_instance=src)
                    columns = list(set(src.columns[:-1]).intersection(tgt.columns[:-1])) + [tgt.columns[-1]]
                    _train, __test = src[columns], tgt[columns]
                    actual, predicted = predict_defects(train=_train, test=__test)
                    # set_trace()
                    mmre = abs((actual - predicted) * 100 / actual)

                stats.append([src_name, int(np.mean(mmre)), int(np.std(mmre))])  # ,

        stats = pandas.DataFrame(sorted(stats, key=lambda lst: lst[1], reverse=False),  # Sort by G Score
                             columns=["Name", "MMRE (Mean)", "MMRE (Std)"])  # ,
        print(tabulate(stats,
                   headers=["Name", "MMRE (Mean)", "MMRE (Std)"],
                   showindex="never",
                   tablefmt="fancy_grid"))

        result.update({tgt_name: stats})
    return result


def tnb_jur():
    all = get_all_datasets()
    bellw(all, all, n_rep=10)
    set_trace()


if __name__ == "__main__":
    tnb_jur()
