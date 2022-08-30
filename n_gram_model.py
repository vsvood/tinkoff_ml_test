"""This module provides NGramModel class"""
from collections import defaultdict
import numpy as np


def zero():
    """return 0. use as default for default
    dict due to lambdas can't be pickled"""
    return 0.


def zero_dict():
    """return default dict with default value 0. use as default
    for default dict due to lambdas can't be pickled"""
    return defaultdict(zero)


class NGramModel:
    """This class represents n-gram language model and provides
    methods to learn model and predict next word according to it """
    def __init__(self, n):
        self.n = n
        self.model = defaultdict(zero_dict)
        self.alpha = defaultdict(zero)

    def fit(self, data: tuple[str]):
        """This function trains model with given dataset"""
        count = defaultdict(zero)
        for i in range(0, len(data) - self.n - 1):
            self.model[data[i:i + self.n]][data[i + self.n]] += 1
            count[data[i:i + self.n]] += 1
        for key in self.model.keys():
            for subkey in self.model[key].keys():
                self.model[key][subkey] /= count[key]
        count = 0
        for word in data:
            self.alpha[word] += 1
            count += 1
        for key in self.alpha.keys():
            self.alpha[key] /= count

    def generate(self, sample: tuple[str]) -> str:
        """This function generates next word according to given sample and model"""
        if sample[-self.n:] in self.model:
            return np.random.choice(list(self.model[sample[-self.n:]].keys()),
                                    p=list(self.model[sample[-self.n:]].values()))
        return np.random.choice(list(self.alpha.keys()), p=list(self.alpha.values()))
