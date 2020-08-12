import inspect
import os
import re
import sys
import warnings
import numpy as np

from probgf.helpers import find_method, load_obj, save_obj, draw_structure
from probgf.tex_output import plot
from probgf.discretization import Discretization
from probgf.validation import cv_foldername, HIDE_VAL
from probgf.methods_simple import SpatioTemporalBase


class RandomField(SpatioTemporalBase):
    """
    Deploys gap filling based on Markov random fields
    Implementation of probabilistic inference used in form of the pxpy library
    More info:
    Nico Piatkowski.
    "Exponential Families on Resource-Constrained Systems"
    (2018)
    https://pypi.org/project/pxpy/
    https://randomfields.org/
    """


    @classmethod
    def default_config(cls):
        return '0.01r,chain,noprior:noprior,0.1,map,8'


    def __init__(self, config, dates, console, emiters):
        super().__init__(config, dates, console, emiters)
        self.mode = 'mrf'


    def name(self):
        return 'MRF_S{}_{}_{}_{}_{}_em{}_{}'.format(str(self.stop_crit).replace('.', '_'), self.prior_name,
                                                       str(self.lam).replace('.', '_'), self.shape,
                                                       self.pred_name, self.emiters, self.discret.info)


    def configure(self, config):
        try:
            prior_methods = [(reg.split('_', 1)[1], mem) for (reg, mem) in inspect.getmembers(self, inspect.ismethod) if reg.startswith('prior_')]
            pred_methods = [(pred.split('_', 1)[1], mem) for (pred, mem) in inspect.getmembers(self, inspect.ismethod) if pred.startswith('predict_')]
            shape_methods = [(shape.split('_', 1)[1], mem) for (shape, mem) in inspect.getmembers(self, inspect.ismethod) if shape.startswith('shape_')]
            if len(config.split(',')) < 6:
                raise RuntimeError('Not enough values given!')
            stop, self.shape, self.prior_name, lam, self.pred_name, disc_config = config.split(',')
            # check chosen stop criteria
            try:
                st_val, st_flag = re.findall(r'([\d\.]*)(\D*)', stop)[0]
                if st_flag == 't':
                    self.stop_crit = int(st_val)
                elif st_flag == 'r':
                    self.no_improve = 0
                    self.stop_crit = float(st_val)
                else:
                    raise ValueError
                if self.stop_crit <= 0: raise ValueError
            except (IndexError, ValueError):
                raise RuntimeError('Config value for stop has to be an int followed by "t" (total number of iterations) or a float followed by "r" (stopping after 100 iterations with lower improvement)! ("{}" given)'.format(stop))
            # check chosen slice shape
            try:
                name, width = re.findall(r'(\D+)(\d*)', self.shape.lower())[0]
            except IndexError:
                raise RuntimeError('Config value for shape needs to start with a string! ("{}" given)'.format(self.shape))
            sh_meth = find_method(name, shape_methods, 'Slice shape')
            positions = sh_meth(width)
            draw_structure(self.edges, positions, 'fig_mrf_cis_{}.png'.format(self.shape.lower()))
            # check chosen prior
            try:
                pr_tmp, pr_spat = self.prior_name.split(':')
            except ValueError:
                raise RuntimeError('Two priors are required for temporal and spatial edges, seperated by a ":"! ("{}" given)'.format(self.prior_name))
            self.prior_name = self.prior_name.replace(':', '_')
            self.temp_prior_method = find_method(pr_tmp, prior_methods, 'Temporal prior regularization')
            self.spat_prior_method = find_method(pr_spat, prior_methods, 'Spatial prior regularization')
            # check chosen lambda
            try:
                self.lam = float(lam)
            except ValueError:
                raise RuntimeError('Config value for lambda has to be float value! ("{}" given)'.format(lam))
            # check chosen prediction
            self.predict = find_method(self.pred_name.lower(), pred_methods, 'Prediction')
            # check chosen discretization
            self.discret = Discretization(disc_config)
        except RuntimeError as error:
            raise RuntimeError('Invalid config "{}".\n{} needs a comma-seperated list of the following values:\n'.format(config, self.__class__.method_id()) + \
            '  stop       : stop criteria (supported: "(int)t" or "(float)r" \n' + \
            '  shape      : shape of slices (supported: {})\n'.format(', '.join([name for name, _ in shape_methods])) + \
            '  priors     : priors for temporal and spatial edges (seperated by a ":") (supported: {})\n'.format(', '.join([name for name, _ in prior_methods])) + \
            '  lambda     : max regularization weight\n' + \
            # '  lambda flag: controls usage of a (f)ixed or (a)dapative lambda\n' + \
            # '               (adaptive calculates values between [0,lambda], depending on the amount of observed information at adjacent nodes)\n' + \
            # '  stateshare : enables (1) state sharing for all vertices (supported: 0 or 1)\n' + \
            '  prediction : method for filling (supported: {})\n'.format(', '.join([name for name, _ in pred_methods])) + \
            '  disc       : discretization clusters\n{}'.format(str(error)))
        except ImportError:
            raise RuntimeError('Import error, please make sure that "pxpy" is correctly installed for using {}!'.format(self.__class__.method_id()))
        self.lam_flag = 'a'
        self.shared = 1


    def shape_chain(self, width):
        if width != '':
            raise RuntimeError('"chain" shape does not support usage with specified width, please simply pass "chain" as argument!')
        self.edges = np.array([np.array([t, t+1]) for t in range(len(self.dates) - 1)], dtype=np.uint64)
        pos = {}
        for vtx in range(len(self.dates)):
            pos[vtx] = [0, vtx]
        return pos


    def shape_cross(self, width):
        if not width.isdigit():
            raise RuntimeError('shape cross also requires width information (e.g. cross3)!')
        width = int(width)
        self.slice_shape = np.zeros((width, width), dtype=bool)
        self.slice_shape[width // 2] = True
        self.slice_shape[:, width // 2] = True
        s_size = np.count_nonzero(self.slice_shape)
        T = len(self.dates)
        v_cent = [t * (s_size) + s_size // 2 for t in range(T)]
        temps = [v_cent[idx:idx + 2] for idx in range(len(v_cent) - 1)]
        spat_v1 = sorted(v_cent * (s_size - 1))
        spat_v2 = [vtx for vtx in range(T * s_size) if vtx not in spat_v1]
        self.edges = np.array(temps + list(zip(spat_v1, spat_v2)), dtype=np.uint64)
        pos = {} # necessary for plotting the structure
        rel_pos = np.array([0.01, 0.15])
        for vtx in range(len(self.dates)):
            middle = np.array([0, vtx])
            for s_vtx in range(s_size): # there are four branches of the cross
                dist = width // 2 - s_vtx % (width // 2) if s_vtx < width else (s_vtx - 1) % (width // 2) + 1
                if s_vtx < s_size // 4: # branch 1
                    pos[vtx * s_size + s_vtx] = middle + rel_pos * dist
                elif s_vtx < s_size // 2: # branch 2
                    pos[vtx * s_size + s_vtx] = middle + rel_pos * [dist, -dist]
                elif s_vtx == s_size // 2: # center node
                    pos[vtx * s_size + s_vtx] = middle
                elif s_vtx < s_size // 4 + s_size // 2 + 1: # branch 3
                    pos[vtx * s_size + s_vtx] = middle + rel_pos * [-dist, dist]
                else: # branch 4
                    pos[vtx * s_size + s_vtx] = middle - rel_pos * dist
        return pos


    def setup_for_cv(self, split, data, obs):
        self.prior_values = None
        self.split = split
        dirname = os.path.join(cv_foldername(self.split), '.px_models')
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        # model filename should not contain prediction method and total em iters
        filename_parts = os.path.join(dirname, '.mod_' + self.name()).split('_')
        self.model_filename = '_'.join(filename_parts[:-3] + filename_parts[-1:])
        # prepare the data and prior
        data, _ = self.discretize(data, obs, split)
        self.temp_prior_matrix = self.temp_prior_method(data)
        self.spat_prior_matrix = self.spat_prior_method(data, spatial=True)
        data = data.reshape(data.shape[0], np.prod(data.shape[1:])) # to n, T * V
        self.calculate_lam_for_edge(data)
        self.map_vertex_states(data)
        return data


    def run_training(self, data, obs, split, progr_train):
        import pxpy as px
        data = self.setup_for_cv(split, data, obs)
        missing = data == HIDE_VAL
        overall_loss = []
        for emiter in range(self.emiters):
            self.obj = sys.maxsize
            loss = []
            new_modelname = self.model_filename + '_{}'.format(emiter)
            if not os.path.isfile(new_modelname):
                if emiter != 0: # load the previous model and fill data gaps with gibbs
                    data[missing] = HIDE_VAL
                    with warnings.catch_warnings(record=True):
                        warnings.simplefilter('ignore')
                        prev_model = px.load_model(self.model_filename + '_{}'.format(emiter - 1))
                        self.predict_gibbs(prev_model, data)
                else:
                    prev_model = None
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter('ignore')
                    model = px.train(data=data, iters=sys.maxsize, graph=px.create_graph(self.edges), 
                                     mode=getattr(px.ModelType, self.mode), shared_states=bool(self.shared),
                                     in_model=prev_model, opt_regularization_hook=self.regularize,
                                     opt_progress_hook=(lambda x, em=emiter, loss=loss : self.check_progress(x, progr_train, em, loss)))
                    model.save(new_modelname)
                    model.graph.delete()
                    model.delete()
                overall_loss.append(('EM Iter ' + str(emiter), loss))
            progr_train[self.split] = (100.0 / self.emiters) * (emiter + 1)
            self.cons.progress(progr_train, self.split)
        self.plot_convergence(overall_loss)
        super().run_training(data, obs, split, progr_train) # for final console output


    def plot_convergence(self, loss):
        conv_plot_name = 'fig_' + cv_foldername(self.split) + os.path.basename(self.model_filename)[4:]
        if not loss == []:
            obj = [(emiter, [(it, obj) for it, obj, _ in data]) for emiter, data in loss]
            reg = [(emiter, [(it, reg) for it, _, reg in data]) for emiter, data in loss]
            loss = [(emiter, [(it, reg + obj) for it, obj, reg in data]) for emiter, data in loss]
            plot(fname=conv_plot_name + '_obj', title='Convergence', xlabel='Iteration', ylabel=r'$\frac{1}{n}\sum_{i=1}^n \log \mP_{\bt}(\bx^i)$', data=obj)
            plot(fname=conv_plot_name + '_reg', title='Convergence', xlabel='Iteration', ylabel=r'$\lambda R(\bt)$', data=reg)
            plot(fname=conv_plot_name + '_loss', title='Convergence', xlabel='Iteration', ylabel=r'$\text{Loss }\ell(\bt)$', data=loss)


    def check_progress(self, state_p, progr, emiter, loss):
        if self.prior_values is not None and (isinstance(self.lam_values, np.ndarray) or self.lam_values != 0): # compute R(theta)
            reg = self.lam * np.square(np.linalg.norm(state_p.contents.best_weights - self.prior_values))
        else: # there is no regularization
            reg = 0
        if state_p.contents.iteration > 1:
            loss.append((state_p.contents.iteration, state_p.contents.best_obj, reg))
        obj_diff = np.abs(self.obj - (state_p.contents.best_obj + reg))
        self.obj = state_p.contents.best_obj + reg
        if isinstance(self.stop_crit, float): # check for relative improvement stopping
            progr[self.split] = (np.exp(state_p.contents.iteration / -100.0) - 1) * -self.per_iter + self.per_iter * emiter
            self.cons.progress(progr, self.split)
            if obj_diff > self.stop_crit:
                self.no_improve = 0
            else:
                if self.no_improve == 100:
                    state_p.contents.iteration = state_p.contents.max_iterations
                    self.no_improve = 0
                else:
                    self.no_improve += 1
        else: # check for total number of iteration stopping
            if state_p.contents.iteration == self.stop_crit:
                state_p.contents.iteration = state_p.contents.max_iterations
            progr[self.split] = float(state_p.contents.iteration) / self.stop_crit * self.per_iter + self.per_iter * emiter
            self.cons.progress(progr, self.split)


    def run_prediction(self, to_pred, obs, split, progr_pred):
        import pxpy as px
        data_disc = self.setup_for_cv(split, to_pred, obs)
        to_pred = to_pred.reshape(data_disc.shape + to_pred.shape[-1:]) # to n, T * V, D    
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('ignore')
            model = px.load_model(self.model_filename + '_{}'.format(self.emiters - 1))
            self.predict(model, data_disc, progr_pred, to_pred)
            model.graph.delete()
            model.delete()
        super().run_prediction(to_pred, obs, self.split, progr_pred) # for final console output
        to_pred = to_pred.reshape(obs.shape)
        if self.slice_shape is not None: # only return the centered slice pixel series
            to_pred = to_pred[:, :, np.count_nonzero(self.slice_shape) // 2]
            to_pred = np.expand_dims(to_pred, 2)
        return to_pred


    def predict_map(self, model, data, progr_pred=None, pred=None):
        """predicts the (conditional) maximum a-posterior states"""
        nr_batches = min(data.shape[0], 1000)
        predicted_at = data == HIDE_VAL
        for idx, data_batch in enumerate(np.array_split(data, nr_batches)): # batch processing
            if progr_pred is not None:
                progr_pred[self.split] = float(idx) / nr_batches * 100
                self.cons.progress(progr_pred, self.split)
            model.predict(data_batch)
        if pred is not None:
            self.map_vertex_states(data, revert=True) # revert to regular states
            pred[predicted_at] = self.discret.continuize(self.split, data[predicted_at])


    def predict_sup(self, model, data, progr_pred=None, pred=None):
        """predicts superposition states, i.e. mixtures of the discrete values based on (conditional) vertex marginals"""
        states = np.array([self.discret.continuize(self.split, k) for k in range(self.discret.k)])
        probs = np.zeros(self.discret.k)
        if pred is None:
            raise RuntimeError('SUP prediction not implemented without target prediction array!')
        for idx, val in enumerate(data):
            if progr_pred is not None:
                if idx % 10 == 0: # do not write in every iteration
                    progr_pred[self.split] = float(idx) / data.shape[0] * 100
                    self.cons.progress(progr_pred, self.split)
            model.infer(observed=val) # has to be called before probs!
            for vertex in range(val.size):
                if val[vertex] == HIDE_VAL:
                    if self.shared:
                        for state in range(model.states[vertex]):
                            probs[state] = model.prob(vertex, state)
                    else:
                        for px_state, state in enumerate(self.vertex_state_maps[vertex]):
                            probs[state] = model.prob(vertex, px_state)
                    pred[idx, vertex] = np.dot(probs, states)


    def predict_gibbs(self, model, data, progr_pred=None, pred=None):
        """samples from the (conditional) MRF distribution (based on Gibbs sampling)"""
        import pxpy as px
        nr_batches = min(data.shape[0], 1000)
        for idx, data_batch in enumerate(np.array_split(data, nr_batches)): # batch processing
            if progr_pred is not None:
                progr_pred[self.split] = float(idx) / nr_batches * 100
                self.cons.progress(progr_pred, self.split)
            model.sample(observed=data_batch, sampler=px.SamplerType.gibbs)
        if pred is not None:
            self.map_vertex_states(data, revert=True) # revert to regular states
            self.discret.continuize(self.split, data.flatten(), pred)


    def map_vertex_states(self, data, revert=False):
        """px uses vertex states from 0 to k_v if share_states is turned off, so the states need to be mapped"""
        assert(len(data.shape) == 2)
        if not self.shared:
            fname = self.model_filename + '_projected_vertex_states'
            if os.path.isfile(fname):
                self.vertex_state_maps = load_obj(fname) # available states for each vertex
            else:
                self.vertex_state_maps = []
                for vertex in range(data.shape[1]):
                    states = np.sort(np.unique(data[:, vertex]))
                    if HIDE_VAL in states: states = states[:-1] # HIDE_VAL is not a regular state
                    self.vertex_state_maps.append(states) # stored internally for later use
                save_obj(self.vertex_state_maps, fname)
            for vertex in range(data.shape[1]):
                states = self.vertex_state_maps[vertex]
                data_v = data[:, vertex]
                if revert:
                    if states.size == 0: # map is empty, i.e. no observed values on this vertex
                        data_v[data_v != HIDE_VAL] = 0
                    else:
                        data_v[data_v != HIDE_VAL] = states[data_v[data_v != HIDE_VAL]]
                else:
                    if states.size == 0: # map is empty, i.e. no observed values on this vertex
                        data_v[:] = HIDE_VAL
                    else:
                        for idx, state in enumerate(states):
                            data_v[data_v == state] = idx
                        state = states[-1]
                        data_v[data_v > state] = HIDE_VAL
                data[:, vertex] = data_v


    def calculate_lam_for_edge(self, data):
        """calculates a lambda for each edge, depending on the amount of observed information"""
        self.lam_for_edge = {}
        for idx, (v_start, v_end) in enumerate(self.edges):
            miss_start = data[:, v_start] == HIDE_VAL
            miss_end = data[:, v_end] == HIDE_VAL
            miss = np.logical_or(miss_start, miss_end)
            self.lam_for_edge[idx] = self.lam * np.count_nonzero(miss) / miss.size


    def construct_state_similarity(self, nonlinear=False, logscale=False):
        try:
            from scipy.spatial.distance import pdist, squareform
            from scipy.interpolate import interp1d
        except Exception:
            raise RuntimeError('Import error, please make sure that "SciPy" is correctly installed for using {}!'.format(self.prior_name))
        cont_states = self.discret.continuize(self.split, np.arange(self.discret.k))
        dists = squareform(pdist(cont_states))
        if nonlinear:
            dists = np.square(dists) / np.max(dists)
        if logscale:
            dists = dists + 0.1 # laplace amoothing to avoid zeros
            rescaling = np.vectorize(interp1d([np.min(dists), np.max(dists)], [np.max(dists), np.min(dists)])) # highest distance = lowest similarity
            scaled_dists = rescaling(dists)
            return np.log(scaled_dists / scaled_dists.sum(axis=1)[:, np.newaxis])
        else:
            rescaling = np.vectorize(interp1d([np.min(dists), np.max(dists)], [1, 0])) # highest distance = lowest similarity
            return rescaling(dists)


    def prior_noprior(self, data, spatial=False):
        """no prior, no regularization"""


    def prior_l2(self, data, spatial=False):
        """L2 zero prior (standard L2 regularization)"""
        return np.zeros((self.discret.k, self.discret.k))


    def prior_es(self, data, spatial=False):
        """Euclidean similarity prior"""
        return self.construct_state_similarity()


    def prior_tp(self, data, spatial=False):
        """(empirical) transition probability prior"""
        counts = np.full((self.discret.k, self.discret.k), 1) # adding one results in non zero entries, necessary for log
        if spatial:
            if self.slice_shape is None:
                return None
            for idx1 in range(data.shape[2]):
                for idx2 in range(data.shape[2]):
                    if any([idx1 in edge and idx2 in edge and idx1 < idx2 for edge in self.edges]):
                        for (idx_n, idx_t), val1 in np.ndenumerate(data[:, :, idx1]):
                            val2 = data[idx_n, idx_t, idx2]
                            if HIDE_VAL not in [val1, val2]: # otherwise transition should not be counted
                                counts[val1, val2] += 1
        else:
            for t in range(data.shape[1] - 1):
                for (idx_n, idx_v), val1 in np.ndenumerate(data[:, t, :]):
                    val2 = data[idx_n, t + 1, idx_v]
                    if HIDE_VAL not in [val1, val2]: # otherwise transition should not be counted
                        counts[val1, val2] += 1
        probs = counts / np.sum(counts)
        return np.log(probs / probs.sum(axis=1)[:, np.newaxis]) # normalize rows and take log

    
    def construct_prior(self, state):
        prior_file = os.path.join(cv_foldername(self.split), '.px_models', 
                                  '.prior_{}_{}_s{}_{}.npy'.format(self.shape, self.prior_name,
                                                                   self.shared, self.discret.info))
        if os.path.isfile(prior_file):
            prior = np.load(prior_file)
        else:
            prior = np.zeros_like(state.weights)
            for edge in range(state.model.graph.edges):
                self.construct_prior_for_edge(edge, state, prior)
            np.save(prior_file, prior)
        return prior


    def construct_prior_for_edge(self, edge, state, prior):
        if edge < len(self.dates) - 1: # there are only T - 1 temporal edges
            matrix = self.temp_prior_matrix
        else:
            matrix = self.spat_prior_matrix
        v0 = self.edges[edge][0] # start vertex
        v1 = self.edges[edge][1] # end vertex
        if not self.shared and (self.vertex_state_maps[v0].size == 0 or 
                                self.vertex_state_maps[v1].size == 0): # no observed values
            prior_values = state.model.slice_edge(edge, prior)
            np.copyto(prior_values, matrix.max()) # edge gets highest prior
        else:
            edge_st = state.model.edge_statespace(edge)
            for s0, s1 in edge_st: # there is a parameter / prior for each state of edge
                if self.shared: # no state projection
                    ps0, ps1 = s0, s1
                else:
                    ps0 = self.vertex_state_maps[v0][s0]
                    ps1 = self.vertex_state_maps[v1][s1]
                prior_value = state.model.slice_edge_state(edge, s0, s1, prior)
                np.copyto(prior_value, matrix[ps0, ps1]) # copy the prior value


    def regularize(self, state_p):
        """computes squared L2 prior regularization"""
        if 'noprior' in self.prior_name:
            return
        state = state_p.contents
        if self.prior_values is None: # will be cached
            if self.lam_flag == 'a': # build the vector of adaptive lambdas
                self.lam_values = np.zeros_like(state.weights)
                for edge in range(state.model.graph.edges):
                    edge_lams = state.model.slice_edge(edge, self.lam_values)
                    np.copyto(edge_lams, self.lam_for_edge[edge]) # the already computed adaptive lambda value
                if np.all(self.lam_values == 0):
                    self.lam_values = 0
            else:
                self.lam_values = self.lam
            self.prior_values = self.construct_prior(state)
        # update gradient, gradient norm and stepsize
        np.copyto(state.gradient, state.gradient + 2.0 * self.lam_values * (state.weights - self.prior_values))
        state.norm = np.linalg.norm(state.gradient, ord=np.inf)
        if state.iteration == 1:
            state.min_stepsize = 1.0/(1.0/state.min_stepsize + 2.0 * self.lam) # lipschitz constant is upper bounded so maximum lambda is fine
