"""main application for filling gaps in spatio-temporal data via command line"""
import os
import sys
import argparse
import json

import numpy as np

from probgf.data_access import get_data_handler
from probgf.helpers import ConsoleOutput, cleanup
from probgf.validation import Validation
from probgf.methods import get_method


def run(run_config):
    cons = ConsoleOutput(run_config.kfolds, run_config.cpus)
    if run_config.seed is not None:
        np.random.seed(run_config.seed)
    # try:
    if run_config.em_iters < 1:
        raise RuntimeError('Invalid value "{}" for em_iters, has to be positive!'.format(run_config.em_iters))
    cons.centered('LOADING DATA', emptystart=True)
    data_handler = get_data_handler(run_config.data, run_config.range.lower())
    valid_handler = Validation(data_handler, run_config.kfolds, run_config.shuffle, run_config.hide_amount, run_config.hide_method, cons)
    method_handler = get_method(run_config.method, run_config.config, valid_handler, cons, run_config.em_iters)
    method_handler.run(run_config.kfolds, run_config.cpus)
    cons.centered('FINISHED', emptystart=True)
    # except Exception as error:
    #     cons.centered('!! ERROR !!', emptystart=True)
    #     print(str(error))
    #     sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    # DATA PARAMETERS
    parser.add_argument('-d', '--data', default='Chess',
                        help='file or directory with data which shall be accessed (pass "C" to clean up the current directory, see README).')
    parser.add_argument('-r', '--range', default='full',
                        help='allows to select a spatial subpart of the data')
    # EVALUATION PARAMETERS
    parser.add_argument('-k', '--kfolds', type=int, default=2,
                        help='number of splits for validation')
    parser.add_argument('-H', '--hide_amount', type=float, default=0.2,
                        help='amount of data that should be hidden for testing (0 <= t <= 1)')
    parser.add_argument('-M', '--hide_method', default='spatial_clouds',
                        help='method for hiding test data')
    parser.add_argument('-s', '--shuffle', type=int, default=0,
                        help='enables data shuffling for the validation splitting')
    parser.add_argument('-S', '--seed', type=int, default=0,
                        help='fixes seed for reproducibility (set to None to disable)')
    # METHOD PARAMETERS
    parser.add_argument('-m', '--method', default='LIN',
                        help='method that shall be run')
    parser.add_argument('-c', '--config', default='default',
                        help='method configuration')
    parser.add_argument('-I', '--em_iters', type=int, default=2,
                        help='number of EM iterations (only used when training data is incomplete)')
    parser.add_argument('-C', '--cpus', type=int, default=1,
                        help='number of cpus that shall be used for parallel execution of cross-validation')

    new_args = parser.parse_args()
    if new_args.data == 'C':
        cleanup()
        sys.exit(0)
    if not os.path.isfile('.config'):
        args = new_args
        with open('.config', 'w') as conf_file:
            json.dump(vars(args), conf_file)
    else:
        with open('.config') as conf_file:
            data = json.load(conf_file)
        args = argparse.Namespace(**data)
        args.cpus = new_args.cpus
        args.em_iters = new_args.em_iters
        args.hide_amount = new_args.hide_amount
        args.hide_method = new_args.hide_method
        args.method = new_args.method
        args.config = new_args.config
    run(args)
