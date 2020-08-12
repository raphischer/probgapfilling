from probgf.data_access import Chess_Handler
from probgf.validation import Validation
from probgf.helpers import ConsoleOutput, cleanup
from probgf.methods import get_method


kfolds = 2
shuffle = False
hide_amount = 0.2
hide_method = 'spatial_gaps'
nr_samples = 0
cpus = 1
config = 'default'
em_iters = 1

data = Chess_Handler(None, None)
cons = ConsoleOutput(kfolds, cpus)


def test_full_run():
    valid_handler = Validation(data, 2, shuffle, hide_amount, hide_method, cons)
    for method in ['NN', 'LIN', 'RandomField', 'HANTS']:
        method_handler = get_method(method, config, valid_handler, cons, em_iters)
        method_handler.run(kfolds, cpus)
    cleanup()
