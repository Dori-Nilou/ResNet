'''
Hyperparameters
'''
import re
import json

import numpy as np


class Hyperparameter:
    '''
    Contains hyperparameter settings
    '''
    pattern = r'[A-Z_]+'
    encoder_registry = {}
    estimator_registry = {}
    separator_registry = {}
    classifer_registry = {}
    ozer_registry = {}
    dataset_registry = {}

    def __init__(self):
        pass


    def load(self, di):
        '''
        load from a dict
        Args:
            di: dict, string -> string
        '''
        assert isinstance(di, dict)
        pat = re.compile(self.pattern)
        for k,v in di.items():
            if None is pat.fullmatch(k):
                raise NameError
            assert isinstance(v, (str, int, float, bool, type(None)))
        self.__dict__.update(di)

    def load_json(self, file_):
        '''
        load from JSON file

        Args:
            file_: string or file-like
        '''
        if isinstance(file_, (str, bytes)):
            file_ = open(file_, 'r')
        di = json.load(file_)
        self.load(di)


hparams = Hyperparameter()



