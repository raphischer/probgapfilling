import numpy as np
from sklearn.metrics import mean_squared_error

from probgf.data_access import get_data_handler
from probgf.discretization import Discretization
from probgf.visualization import visualize_rgb

data = get_data_handler('/gapfilling/dortmund_from_space_2018/', 'full')

mask = data.mask
data = data.data

original = np.array(data[mask])

for k in [2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48]:
    to_discretize = np.array(original)
    discret = Discretization(str(k))
    labels = discret.discretize(0, to_discretize)
    discretized = discret.continuize(0, labels)
    error = np.sqrt(mean_squared_error(original, discretized))
    print(k, error)
    data[mask] = discretized
    if k in [4, 8, 16]:
        for t in range(data.shape[0]):
            img = data[t].reshape(1000, 1000, 3)
            visualize_rgb(img, 'test_{}_{}.png'.format(k, t), np.array([0, 1, 2]), np.array([0, 0, 0]), np.array([3000,3000, 3000]))
