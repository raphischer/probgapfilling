"""contains classes and methods for spatio-temporal data handling"""
import os
import re
import sys
import inspect
import datetime

import numpy as np

from probgf.helpers import project_value
from probgf.media import cloudshape
import probgf.visualization as visualization


FULL = 'full'


def get_shape(width, shape):
    """returns array indicies of a circle with a specified width"""
    from PIL import Image, ImageDraw
    if shape == 'circle':
        shape_img = Image.new('1', (width, width))
        draw = ImageDraw.Draw(shape_img)
        draw.rectangle((0, 0, width, width), fill='BLACK', outline='BLACK')
        draw.ellipse((0, 0, width, width), fill='WHITE', outline='WHITE')
    elif shape == 'cloud':
        shape_img = cloudshape.convert(mode='1').resize((width, width))
    mask = np.array(shape_img).astype(bool) # bool array with shape being True values
    return mask


def create_outline(shape, width):
    """creates outline of all True values in the array"""
    outlines = np.zeros_like(shape)
    for idx_t, idx_x, idx_y in np.argwhere(shape):
        x_start = max([idx_x - width, 0])
        x_end = min([idx_x + width + 1, outlines.shape[1]])
        y_start = max([idx_y - width, 0])
        y_end = min([idx_y + width + 1, outlines.shape[2]])
        outlines[idx_t, x_start:x_end, y_start:y_end] = True
    outlines[shape] = False
    return outlines


def get_data_handler(data, subrange):
    """returns the corresponding data handler based on the given data"""
    handlers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    hand_ids = [(hand, hand.data_id()) for _, hand in handlers if hand.data_id() != '']
    for hand, h_id in hand_ids:
        if h_id in data:
            return hand(data, subrange)
    msg = '''No implemented handler for the specified dataset.
        Implemented handlers (with string data_ids that need to be found in -d argument):\n\n'''
    handlers = '\n'.join(['{} (data_id: "{}") - {}'.format(
        hand.__name__, h_id, hand.__doc__) for hand, h_id in hand_ids])
    raise NotImplementedError(msg + handlers)


class SingleSampleImageTimeSeriesHandler:
    """
    Handles spatio-temporal data in form of images over time
    As only one instance of data is available, the data is sliced along the spatial dimension
    Subranging is possible by passing minimum x and y coordinates
    The images and masks are visualized and spatial gaps can be placed for evaluation
    """
    @classmethod
    def data_id(cls):
        if len(cls.__name__.split('_')) == 1: # no specialized handler
            return ''
        return '_'.join(cls.__name__.split('_')[:-1]) # data_id is first substring


    def __init__(self, data, mask, dates, subrange, gap_width, minmax, vis_bands):
        self.N = 1
        if len(data.shape) != 4:
            raise RuntimeError('Single sample image time series has to have shape (T, W, H, D)!')
        self.T, self.W, self.H, self.D = data.shape
        if subrange != FULL:
            x_start, x_end, y_start, y_end = self.extract_subrange(subrange)
            data = data[:, x_start:x_end, y_start:y_end, :]
            mask = mask[:, x_start:x_end, y_start:y_end]
            self.W, self.H = x_end - x_start, y_end - y_start
        self.V = self.W * self.H
        self.data = data.reshape((self.T, self.W * self.H, self.N, self.D)) # reshape to T V N D
        self.mask = mask.reshape((self.T, self.W * self.H, self.N))
        self.gap_width = gap_width
        self.visualizer = visualization.Visualizer(minmax, vis_bands)
        
        if dates.dtype != int:
            raise RuntimeError('Dates must be ints!')
        if dates.size != self.T:
            raise RuntimeError('There must be a date for each t <= T!')
        if not np.all(dates == np.sort(dates)):
            raise RuntimeError('Dates must be ordered!')
        if np.unique(dates).size != self.T:
            raise RuntimeError('Dates must be unique!')
        self.dates = dates

        self.prediction = np.array(self.data)

        self.visualizer.visualize_all(self.data.reshape(self.T, self.W, self.H, self.D),
                                      os.path.join('imgs', 'original'))
        self.visualizer.visualize_all(self.mask.reshape(self.T, self.W, self.H),
                                      os.path.join('imgs', 'mask'))
        data_obs, data_masked = np.zeros_like(self.data), np.zeros_like(self.data)
        data_obs[self.mask] = self.data[self.mask]
        data_masked[np.invert(self.mask)] = self.data[np.invert(self.mask)]
        self.visualizer.visualize_all(data_obs.reshape(self.T, self.W, self.H, self.D),
                                      os.path.join('imgs', 'original_observed'))
        self.visualizer.visualize_all(data_masked.reshape(self.T, self.W, self.H, self.D),
                                      os.path.join('imgs', 'original_masked'))


    def extract_subrange(self, subrange):
        """Extraction of data subrange based on (X Start, X End, Y Start, Y End) values"""
        try:
            if len(subrange.split(',')) != 4:
                raise RuntimeError('''You have to pass four comma-seperated values!''')
            if not all([part.isdigit() for part in subrange.split(',')]):
                raise RuntimeError('''You have to pass integer values!''')
            x_start, x_end, y_start, y_end = [int(idx) for idx in subrange.split(',')]
            if not self.valid_subrange(x_start, x_end, y_start, y_end):
                raise RuntimeError('''Invalid values for (X Start, X End, Y Start, Y End).
                                      They must be >= zero and < {} (X) and {} (Y) '''
                                      .format(self.W, self.H))
        except RuntimeError as error:
            raise RuntimeError('{} failed!\nInvalid given subrange "{}"!\n{}'.format(
                self.extract_subrange.__doc__, subrange, str(error)))
        return x_start, x_end, y_start, y_end


    def valid_subrange(self, x_start, x_end, y_start, y_end):
        if x_start < 0 or x_end >= self.W or x_start >= x_end:
            return False
        if y_start < 0 or y_end >= self.H or y_start >= y_end:
            return False
        return True


    def idcs(self):
        """returns the number of possible different data samples"""
        return self.V


    def extr_data(self, rows, slice_shape=None):
        """
        extracts specified parts of the data
        allows boolean ndarray shapes for slicing
        """
        if slice_shape is not None and not isinstance(slice_shape, np.ndarray):
            raise NotImplementedError('slice_shape "{}" has not been implemented for {}!'.format(slice_shape, self.__class__.__name__))
        if slice_shape is None:
            return np.moveaxis(self.data[:, rows], 1, 0) # (n, T, V, D)
        off = slice_shape.shape[0] // 2
        shape_x, shape_y = np.where(slice_shape)
        d_slice = np.zeros((len(rows), self.T, shape_x.size, self.D), dtype=self.data.dtype)
        pos_x, pos_y = np.unravel_index(rows, (self.W, self.H))
        for shape_idx in range(shape_x.size):
            idc = np.ravel_multi_index((pos_x + shape_x[shape_idx] - off, pos_y + shape_y[shape_idx] - off),
                                       (self.W, self.H), 'wrap') # overflow will be handled by mask
            d_slice[:, :, shape_idx, :] = np.moveaxis(self.data[:, idc, 0], 1, 0)
        return d_slice


    def extr_mask(self, rows, slice_shape=None):
        """
        extracts specified parts of the data mask
        allows boolean ndarray shapes for slicing
        """
        if slice_shape is not None and not isinstance(slice_shape, np.ndarray):
            raise NotImplementedError('slice_shape "{}" has not been implemented for {}!'.format(slice_shape, self.__class__.__name__))
        if slice_shape is None:
            return np.moveaxis(self.mask[:, rows], 1, 0) # (n, T, V, D)
        off = slice_shape.shape[0] // 2
        shape_x, shape_y = np.where(slice_shape)
        m_slice = np.zeros((len(rows), self.T, shape_x.size), dtype=self.mask.dtype)
        pos_x, pos_y = np.unravel_index(rows, (self.W, self.H))
        for shape_idx in range(shape_x.size):
            idc_w = np.ravel_multi_index((pos_x + shape_x[shape_idx] - off, pos_y + shape_y[shape_idx] - off),
                                         (self.W, self.H), 'wrap')
            idc_c = np.ravel_multi_index((pos_x + shape_x[shape_idx] - off, pos_y + shape_y[shape_idx] - off),
                                         (self.W, self.H), 'clip')
            idc_overflow = idc_c != idc_w # all overflow pixels (diff of clip and wrap) shall be unobserved
            m_slice[:, :, shape_idx] = np.moveaxis(self.mask[:, idc_c, 0], 1, 0)
            for t in range(self.T):
                m_slice_t = m_slice[:, t, shape_idx]
                m_slice_t[idc_overflow] = False
                m_slice[:, t, shape_idx] = m_slice_t
        return m_slice


    def hide(self, rows, hide_method, amount, fold):
        """hides a certain amount of data and visualizes the results"""
        methods = [(name.split('_', 1)[1], method)
                   for name, method in inspect.getmembers(self, inspect.ismethod)
                   if name.startswith('hide_')]
        for name, method in methods:
            if hide_method.lower() == name:
                test_mask = method(rows, amount)
                break
        else:
            raise NotImplementedError('''Invalid hide method "{}".
                All implemented methods:\n\n'''.format(hide_method) + '\n'.join(
                    ['{} - {}'.format(name, method.__doc__) for name, method in methods]))
        split_imgs = np.zeros((self.V), dtype=bool)
        split_imgs[rows] = True
        split_hidden = np.zeros((self.T, self.V, self.N), dtype=bool)
        split_hidden[:, rows, :] = np.moveaxis(test_mask, 0, 1)
        self.visualizer.visualize_all(split_hidden.reshape(self.T, self.W, self.H),
                                      os.path.join('imgs', 'cv_{}_test'.format(fold)),
                                      fname=hide_method.lower())
        visualization.visualize_bw(split_imgs.reshape(self.W, self.H),
                                   os.path.join('imgs', 'cv_{}_test'.format(fold), 'test.png'))
        return test_mask


    def hide_latest(self, rows, amount):
        """hides the latest vertices"""
        observed = self.extr_mask(rows)
        mask_hidden = np.zeros_like(observed)
        to_hide = int(mask_hidden[observed].size * amount)
        t = self.T - 1
        while to_hide > 0: # iteratively hide data
            mask_at_t = mask_hidden[:, t]
            mask_at_t[observed[:, t]] = True
            mask_hidden[:, t] = mask_at_t
            to_hide = int(mask_hidden[observed].size * amount) - np.count_nonzero(mask_hidden)
            t -= 1
        return mask_hidden


    def hide_random(self, rows, amount):
        """hides random vertices"""
        observed = self.extr_mask(rows)
        mask_hidden = np.zeros_like(observed)
        hidable = mask_hidden[observed] # only the observed areas can be hidden
        hide_idc = np.random.choice(np.arange(0, hidable.size),
                                    size=int(hidable.size * amount),
                                    replace=False) # idc that will be hidden
        hidable[hide_idc] = True
        mask_hidden[observed] = hidable
        return mask_hidden


    def hide_random_spatials(self, rows, amount):
        """hides random vertices over the whole time"""
        observed = self.extr_mask(rows)
        mask_hidden = np.zeros_like(observed)
        to_hide = int(mask_hidden[observed].size * amount)
        vtc = np.random.permutation(len(rows)) # all possible tested vertices
        idx = 0
        while to_hide > 0: # iteratively hide temporal data for random vertices
            mask_at_v = mask_hidden[vtc[idx]]
            mask_at_v[observed[vtc[idx]]] = True
            mask_hidden[vtc[idx]] = mask_at_v
            to_hide = int(mask_hidden[observed].size * amount) - np.count_nonzero(mask_hidden)
            idx += 1
        return mask_hidden


    def hide_spatial_clouds(self, rows, amount):
        """hides spatial gaps in shape of clouds of predefined width"""
        return self.spatial_shape_hiding(rows, amount, get_shape(self.gap_width, 'cloud'))


    def hide_spatial_gaps(self, rows, amount):
        """hides round spatial gaps of predefined width"""
        return self.spatial_shape_hiding(rows, amount, get_shape(self.gap_width, 'circle'))


    def spatial_shape_hiding(self, rows, amount, shape_mask):
        """hides spatial gaps of predefined width and shape"""
        circ_x = np.argwhere(shape_mask)[:, 0] # shape x indices
        circ_y = np.argwhere(shape_mask)[:, 1] # shape y indices
        mask_hidden = np.array(self.mask).reshape((self.T, self.W, self.H, self.N))
        placeable = np.zeros_like(mask_hidden, dtype=bool)
        xrows, yrows = np.unravel_index(rows, (self.W, self.H)) # pylint: disable=unbalanced-tuple-unpacking
        placeable[:, xrows, yrows] = mask_hidden[:, xrows, yrows] # only the observed are 1
        mask_hidden[:] = False # will now store the artificial gaps
        to_hide = int(placeable[:, xrows, yrows][self.mask[:, rows]].size * amount)
        vtc = np.random.permutation(rows.size * self.T) # all possible gap positions (shuffled)
        idx = 0
        while to_hide > 0: # iteratively place spatial gaps
            if idx == vtc.size:
                raise RuntimeError('''Not enough spatial gaps could be placed.
                    Make sure that shuffle is turned off and try decreasing
                    the number of CV folds or the amount of data to hide.''')
            v_idx, time = np.divmod(vtc[idx], self.T)
            pos_x, pos_y = np.unravel_index(rows[v_idx], (self.W, self.H)) # pylint: disable=unbalanced-tuple-unpacking
            circ_pos_x = pos_x + circ_x
            circ_pos_y = pos_y + circ_y
            if (circ_pos_x.max() < self.W and circ_pos_y.max() < self.H
                    and np.all(placeable[time, circ_pos_x, circ_pos_y])): # all entries observed?
                placeable[time, circ_pos_x, circ_pos_y] = False # not available for following gaps
                mask_hidden[time, circ_pos_x, circ_pos_y] = True # store gap
                to_hide -= circ_x.size # update number of remaining pixels to hide
            idx += 1
        mask_hidden = mask_hidden.reshape(self.mask.shape)[:, rows]
        return np.moveaxis(mask_hidden, 1, 0)


    def store_prediction(self, complete_prediction, pred_name):
        """stores and visualizes the prediction"""
        all_hidden = np.zeros_like(self.mask)
        for prediction, testidcs, hidden in complete_prediction:
            prediction = np.moveaxis(prediction, 0, 1)
            hidden = np.moveaxis(hidden, 0, 1)
            tmp = all_hidden[:, testidcs]
            tmp[hidden] = True
            all_hidden[:, testidcs] = tmp
            self.prediction[:, testidcs] = prediction
        # visualize prediction
        self.visualizer.visualize_all(self.prediction.reshape(self.T, self.W, self.H, self.D),
                                      os.path.join('imgs', 'pred_hidden_' + pred_name))
        # visualize prediction with gap outlines
        all_hidden = all_hidden.reshape(self.T, self.W, self.H)
        outlines = create_outline(all_hidden, max(self.gap_width // 20, 3))
        outline_color = np.zeros(self.D)
        outline_color[self.visualizer.bands] = (self.visualizer.max_v - self.visualizer.min_v) // 2 # grey
        # = [self.visualizer.max_v[0], self.visualizer.min_v[1], self.visualizer.min_v[2]] # red
        # = self.visualizer.max_v # white
        output = np.array(self.prediction.reshape(self.T, self.W, self.H, self.D))
        output[outlines] = outline_color
        self.visualizer.visualize_all(output, os.path.join('imgs', 'pred_outline_' + pred_name))
        output = np.array(self.data.reshape(self.T, self.W, self.H, self.D))
        output[outlines] = outline_color
        self.visualizer.visualize_all(output, os.path.join('imgs', 'original_outline'))
        # visualize prediction error
        output = np.mean(np.abs(self.prediction - self.data), axis=3)
        output[np.invert(self.mask)] = 0
        output = output.reshape(self.T, self.W, self.H)
        self.visualizer.visualize_all(output, os.path.join('imgs', 'pred_error_' + pred_name))
        # visualize prediction without artificial gaps
        output = np.array(self.prediction)
        output[self.mask] = self.data[self.mask]
        output = output.reshape(self.T, self.W, self.H, self.D)
        self.visualizer.visualize_all(output, os.path.join('imgs', 'pred_' + pred_name))


class Chess_Handler(SingleSampleImageTimeSeriesHandler):
    """Creates synthetic chessboard data"""
    def __init__(self, filename, subrange):
        from probgf.visualization import MAXINTENS
        w_board = 4
        w_field = 16
        w_total = w_board * w_field
        times = 4
        dim = 3
        gap_width = w_field * 2
        minmax = (np.array([0, 0, 0]), np.array([MAXINTENS, MAXINTENS, MAXINTENS]))
        bands = [0, 1, 2]
        dates = np.arange(times)

        board = np.kron([[1, 0] * (w_board // 2), [0, 1] * (w_board // 2)] * (w_board // 2),
                        np.ones((w_field, w_field)))
        board_series = np.repeat(board, times).reshape(w_total, w_total, times)
        board_series = np.moveaxis(board_series, 2, 0) # T, W, H
        board_series[1] = np.invert(board.astype(bool))
        board_series[3] = np.invert(board.astype(bool))
        mask = np.ones(board_series.shape, dtype=bool)
        board_series = np.repeat(board_series, dim)
        board_series = board_series.reshape(times, w_board * w_field, w_board * w_field, dim)
        for t, colours in enumerate([[1.0, 0, 0], [0, 1.0, 0], [1.0, 0, 0], [0, 0, 1.0]]):
            for d, color in enumerate(colours):
                s_idx = int(0.5 * (w_total / 2))
                e_idx = s_idx * 3
                board_series[t, s_idx:e_idx, s_idx:e_idx, d] = color
        board_series[:, :w_field, :w_field] = [0.5, 0, 0.5]
        board_series[:, -w_field:, -w_field:] = [0.5, 0, 0.5]
        board_series = board_series * MAXINTENS
        super().__init__(board_series, mask, dates, FULL, gap_width, minmax, bands)


    def hide(self, rows, hide_method, amount, fold):
        """hides a certain amount of data on the second chessboard"""
        tmp = np.array(self.mask)
        self.mask[[0, 2, 3]] = False
        test_mask = super().hide(rows, hide_method, amount, fold)
        self.mask = tmp
        return test_mask


class Chess2_Handler(Chess_Handler):
    """Creates synthetic chessboard data, with t=2 being totally hidden"""
    def __init__(self, filename, subrange):
        super().__init__(filename, subrange)
        self.mask[1] = False


class dortmund_from_space_2018_Handler(SingleSampleImageTimeSeriesHandler):
    """
    Handler for Dortmund from Space 2018 data
    Downloadable from https://www.dropbox.com/sh/ohbb4zpae9djb3z/AADi5qGbsPB2peLGg2-gh8LWa
    It features surface reflectance of a region in Germany from 2018
    The data is an extract of Landsat 8 OLI/TIRS C1 Level-2 data, distributed by USGS
    More info in README that comes with the data
    """
    def __init__(self, dirname, subrange):
        if (os.path.isfile('.dates.npy') and
                os.path.isfile('.data.npy') and
                os.path.isfile('.masks.npy')): # faster import
            dates = np.load('.dates.npy')
            data = np.load('.data.npy')
            masks = np.load('.masks.npy')
        else: # load csv and prepare faster import
            mask_data = np.genfromtxt(os.path.join(dirname, 'mask.csv'), delimiter=',')
            masks = np.moveaxis(mask_data[1:].astype(bool).reshape(1000, 1000, 21), 2, 0)
            dates = mask_data[0].astype(int)
            data = np.genfromtxt(os.path.join(dirname, 'data.csv'), delimiter=',', dtype=np.int16)[1:]
            data = np.moveaxis(data.reshape(1000, 1000, 3, 21), 3, 0)
            np.save('.dates.npy', dates)
            np.save('.data.npy', data)
            np.save('.masks.npy', masks)
        minmax = (np.array([0, 0, 0]), np.array([3000, 3000, 3000]))
        cloud_width = 70
        super().__init__(data, masks, dates, subrange, cloud_width, minmax, None)
