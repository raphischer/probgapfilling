import inspect
import multiprocessing as mp
import sys
import time

from probgf.random_fields import RandomField
from probgf.methods_simple import LIN, NN, HANTS
from probgf.nspi import NSPI


def get_method(methodname, config, data_handler, console, emiters):
    methods = [cl for cl in inspect.getmembers(sys.modules[__name__], inspect.isclass) if 'method_id' in dir(cl[1])]
    method_ids = [(method, method.method_id()) for _, method in methods if method.method_id() != '']
    for method, method_id in method_ids:
        if method_id.lower() == methodname.lower():
            if config == 'default':
                config = method.default_config()
            meth = method(config, data_handler.handler.dates, console, emiters)
            return MethodHandler(meth, data_handler, console)
    raise NotImplementedError('Specified method "{}" not (yet) implemented. Currently featured methods:\n\n'.format(methodname) + \
        '\n'.join(['{} - {}'.format(method.__name__, method.__doc__) for method, _ in method_ids]))


class MethodHandler:

    
    def __init__(self, method, data_handler, console):
        self.method = method
        self.data_handler = data_handler
        self.cons = console


    def run(self, kfolds, cpus):
        if cpus == 1:
            t_train, predicted = self.run_sequential(kfolds)
        elif cpus > 1:
            t_train, predicted = self.run_parallel(kfolds, cpus)
        else:
            raise RuntimeError('Invalid given number "{}" of CPUs to use!'.format(cpus))
        self.data_handler.evaluate_prediction(t_train, predicted, self.method.name().lower())


    def run_parallel(self, kfolds, cpus): # multi core
        with mp.Manager() as manager: # allows to use global lists for progress
            progr_train = manager.list() # for training progress
            progr_pred = manager.list() # for prediction progress
            split_data = []
            for split in range(kfolds):
                progr_pred.append(0.0)
                progr_train.append(0.0)
                split_data.append((split, progr_train, progr_pred))
            with mp.Pool(cpus) as pool:
                self.cons.centered('TRAIN ' + self.method.name(), emptystart=True)
                self.cons.clear_for_prog()
                t_train = pool.map(self.run_training, split_data)
                self.cons.centered('RUN ' + self.method.name(), emptystart=True)
                self.cons.clear_for_prog()
                predicted = pool.map(self.run_prediction, split_data)
        return t_train, predicted 


    def run_sequential(self, kfolds): # single core
        t_train = []
        predicted = []
        for split in range(kfolds):
            progr_train = [0.0 for _ in range(kfolds)]
            progr_pred = [0.0 for _ in range(kfolds)]
            self.cons.centered('TRAIN {}/{} {}'.format(split + 1, kfolds, self.method.name()), emptystart=True)
            self.cons.clear_for_prog()
            t_train.append(self.run_training((split, progr_train, progr_pred)))
            self.cons.centered('RUN {}/{} {}'.format(split + 1, kfolds, self.method.name()), emptystart=True)
            self.cons.clear_for_prog()
            predicted.append(self.run_prediction((split, progr_train, progr_pred)))
        return t_train, predicted


    def run_training(self, split_data):
        split, progr_train, _ = split_data
        data_train, obs_train = self.data_handler.get_data(split, 'train', self.method.slice_shape)
        start = time.time()
        self.method.run_training(data_train, obs_train, split, progr_train)
        end = time.time()
        return end - start


    def run_prediction(self, split_data):
        split, _, progr_pred = split_data
        data_predict, obs_predict = self.data_handler.get_data(split, 'predict', self.method.slice_shape)
        start = time.time()
        pred = self.method.run_prediction(data_predict, obs_predict, split, progr_pred)
        end = time.time()
        return pred, end - start
