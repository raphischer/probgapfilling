import os
import numpy as np

from probgf.validation import cv_foldername
from probgf.helpers import load_obj, save_obj


class Discretization:
    def __init__(self, k):
        if not k.isdigit() or int(k) < 2:
            raise RuntimeError('Valid discretization need desired number of discrete values k (int > 1)!')
        self.k = int(k)
        self.discretizer = MiniBatchKMeansDiscretizer
        self.info = self.discretizer.info(self.k)
        self.discretizers = {} # later used to store discretizers for the individual splits


    def discretize(self, split, data):
        assert(len(data.shape) == 2) # data shape has to be (n, D)
        if split not in self.discretizers:
            self.discretizers[split] = self.discretizer(split, self.k, data)
        d_labels = self.discretizers[split].predict_discrete(data)
        return d_labels


    def continuize(self, split, data, dst=None):
        if not isinstance(data, int) and not (len(data.shape) == 1):
            raise RuntimeError('continuize requires data to be an integer or ndarray of shape (n,)!')
        if split not in self.discretizers:
            self.discretizers[split] = self.discretizer(split, self.k, data)
        if dst is None: # return the continuus data
            return self.discretizers[split].predict_cont(data)
        # or just copy them to the destination array
        np.copyto(dst, self.discretizers[split].predict_cont(data).reshape(dst.shape))
        return None


class MiniBatchKMeansDiscretizer:


    @classmethod
    def info(cls, k):
        return 'K{}Means'.format(k)


    def __init__(self, split, k=None, data=None):
        """load previously trained discretizer or train and store a new one"""
        disc_file = self.get_filename(split, k)
        if os.path.isfile(disc_file):
            self.load(disc_file)
        else:
            self.train_and_store(disc_file, k, data)


    def load(self, disc_file):
        self.discretizer = load_obj(disc_file)


    def train_and_store(self, disc_file, k, data):
        """discretizes by running D-dimensional MiniBatchKMeans clustering"""
        from sklearn.cluster import MiniBatchKMeans
        self.discretizer = MiniBatchKMeans(n_clusters=k)
        self.discretizer.fit(data)
        self.discretizer.cluster_centers_ = self.discretizer.cluster_centers_.astype(data.dtype)
        save_obj(self.discretizer, disc_file)


    def predict_discrete(self, data):
        """returns the centroid indices for values"""
        return self.discretizer.predict(data)


    def predict_cont(self, data):
        """returns the centroid values for indices"""
        return self.discretizer.cluster_centers_[data]


    def get_filename(self, split, k):
        return os.path.join(cv_foldername(split), '.' + self.info(k).lower().replace(' ', '_'))
