import os
import re
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from skimage.metrics import structural_similarity as ssim

from probgf.tex_output import plot


HIDE_VAL = np.uint16(-1)
name_comp_report = 'report_compare.csv'


def cv_foldername(k):
    return 'CV_' + str(k)


def move_to_cv():
    '''moves files to dedicated CV folders'''
    for entry in os.listdir():
        if os.path.isfile(entry):
            match = re.match(r'(.*)(CV_\d*)(.*)', entry)
            if match is not None:
                os.rename(entry, os.path.join(match.group(2), match.group(1)[:-1] + match.group(3)))


def weighted_avg_and_std(values, weights):
    """computes the weighted average and standard deviation for a set of values"""
    average = np.average(values, weights=weights)
    variance = np.average((values - average)**2, weights=weights)
    return (average, np.sqrt(variance))


def compare(prediction, original, hidden):
    """compares the artificially hidden parts of a prediction with original data"""
    cnt_hid = np.count_nonzero(hidden)
    if cnt_hid == 0:
        return 0, 0, 0, 0
    y_true, y_pred = original[hidden], prediction[hidden]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return cnt_hid, mae, rmse, r2


def calc_ssim(prediction, data, hidden, testidcs, W, H):
    # restructure to image data
    idx_r, idx_c = np.unravel_index(testidcs, (W, H))
    img1 = np.zeros((idx_r.max() - idx_r.min() + 1, idx_c.max() - idx_c.min() + 1, data.shape[2]))
    img2 = np.zeros((idx_r.max() - idx_r.min() + 1, idx_c.max() - idx_c.min() + 1, data.shape[2]))
    idx_c = idx_c - idx_c.min()
    idx_r = idx_r - idx_r.min()
    for idx in range(data.shape[0]):
        if hidden[idx, data.shape[1] // 2]:
            img2[idx_r[idx], idx_c[idx], :] = prediction[idx , data.shape[1] // 2, :]
        else:
            img2[idx_r[idx], idx_c[idx], :] = data[idx, data.shape[1] // 2, :]
        img1[idx_r[idx], idx_c[idx], :] = data[idx, data.shape[1] // 2, :]
    # calculate ssim
    return ssim(img1, img2, multichannel=True)


def slices_to_img(data, idcs, W, H, inverse=False):
    idx_r, idx_c = np.unravel_index(idcs, (W, H))
    idx_c = idx_c - idx_c.min()
    idx_r = idx_r - idx_r.min()
    if inverse:
        res = np.zeros((idcs.size, data.shape[2], 1, data.shape[3]), dtype=data.dtype)
    else:
        res = np.zeros((idx_r.max() - idx_r.min() + 1, idx_c.max() - idx_c.min() + 1, data.shape[1], data.shape[3]), dtype=data.dtype)
    for idx in range(idcs.size):
        if inverse:
            res[idx, :, 0] = data[idx_r[idx], idx_c[idx]]
        else:
            res[idx_r[idx], idx_c[idx]] = data[idx, :, 0]
    del data
    return res


def write_comparison_report(out):
    """reads the last line of reports in a directory and aggregates them into a comparing report"""
    regex = 'report_(.*).csv'
    preds = [(os.path.join(out, fname), re.match(regex, fname)) for fname in os.listdir(out) if 'compare' not in fname]
    preds = [(fname, match.group(1)) for (fname, match) in sorted(preds) if match is not None]
    versions = []
    with open(os.path.join(out, name_comp_report), 'w') as report_compare:
        report_compare.write('version,obs,miss,hid,mae,sd_mae,rmse,rmse_std,r2,r2_std,ssim,ssim_std,t_train,t_pred\n')
        for fname, version in preds:
            versions.append(os.path.basename(fname))
            report = np.genfromtxt(fname, delimiter=',')
            if report.shape[1] == 11:
                obs, miss, hid, mae, rmse, r2, ssim, t_train, t_predict = report[-1, 2:]
                mae_std = rmse_std = r2_std = ssim_std = 'n.a.'
            else:
                if report.shape[1] != 15:
                    raise RuntimeError('Quality reports are malformed!')
                obs, miss, hid, mae, mae_std, rmse, rmse_std, r2, r2_std, ssim, ssim_std, t_train, t_predict = report[-1, 2:]
            report_compare.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(version, int(obs), int(miss), int(hid), mae, mae_std, rmse, rmse_std, r2, r2_std, ssim, ssim_std, t_train, t_predict))
    return versions


def plot_error_over_time(reports, cumulated=True):
    errors = [('mae', 5), ('rmse', 7), ('r2', 9), ('ssim', 11)]
    report_data = []
    for report in reports:
        report_data.append((report[7:], np.genfromtxt(report, delimiter=',')[1:, :]))
    for err, err_idx in errors:
        if cumulated:
            fname = 'fig_over_time_{}_cum'.format(err)
            y_label = 'Cumulative {}'.format(err.upper())
        else:
            fname = 'fig_over_time_{}'.format(err)
            y_label = '{}'.format(err.upper())
        plot_data = []
        for name, data in report_data:
            error = data[:-1, err_idx]
            hidden = error != 0
            plot_days = data[:-1, 1][hidden]
            plot_error = error[hidden]
            if cumulated:
                plot_error = np.cumsum(plot_error) / np.sum(plot_error) * data[-1, err_idx]# / np.iinfo(np.uint16).max
            plot_data.append((name, list(zip(plot_days, plot_error))))
        plot(fname, title='Method Quality Comparison', xlabel='Day t', ylabel=y_label, data=plot_data)


def plot_error_for_prior(reports):
    models = {}
    all_lambs = set()
    for report in reports: # extract errors for all models and prior configs
        errs = np.genfromtxt(report, delimiter=',')[-1, [5, 7, 9, 11]]
        nameparts = re.findall(r'report_mrf_(s[\d]*_[\d]*)_([a-z\d]*_[a-z\d]*)_(\d+_[\d]*)_([a-z\d]*)_([a-z]*)_(em\d*)_([a-z\d]*)_(.*)\.csv', report)
        if len(nameparts) != 1: # no mrf model
            models[report[7:-4]] = errs
        else:
            stop, prior, lam, shape, predict, em, discret, hide = nameparts[0]
            remaining = '_'.join(['mrf', stop, prior, shape, predict, em, discret, hide])
            if remaining not in models:
                models[remaining] = {}
            if lam in models[remaining]:
                print('There was an issue with plotting method comparison for priors')
                return
            lam = float(lam.replace('_', '.'))
            models[remaining][lam] = errs
            all_lambs.add(lam)
    for idx, err in enumerate(['mae', 'rmse', 'r2', 'ssim']): # plot a prior error graph for each error
        fname = 'fig_method_comparison_for_prior_{}'.format(err)
        plot_data = []
        for model in models:
            if isinstance(models[model], dict):
                lambs = sorted(models[model].keys())
                errs = [models[model][lam][idx] for lam in lambs]
            else:
                if bool(all_lambs):
                    lambs = [min(all_lambs), max(all_lambs)]
                else:
                    lambs = [0, 1]
                errs = [models[model][idx], models[model][idx]]
            plot_data.append((model, list(zip(lambs, errs))))
        plot(fname, title='Method Quality Comparison', xlabel=r'$\lambda$', ylabel=err.upper(), data=plot_data, xlog=True)


def plot_cloud_stats():
    content = [name for name in os.listdir() if name.startswith('report_')]
    plot_data = []
    if len(content) < 2:
        raise RuntimeError('Quality reports are malformed!')
    if content[0] != name_comp_report:
        stats = np.genfromtxt(content[0], delimiter=',', dtype=int)[1:-1, [1, 2, 3, 4]]
    else:
        stats = np.genfromtxt(content[1], delimiter=',', dtype=int)[1:-1, [1, 2, 3, 4]]
    for idx, coverage in enumerate(['observed', 'missing', 'hidden']):
        plot_data.append((coverage, list(zip(list(stats[:, 0]), list(stats[:, 1 + idx])))))
    plot('fig_cloud_coverage', title='Cloud Coverage', xlabel='Time', ylabel='Number of pixels', data=plot_data, barplot=True)


class Validation:
    """ 
    constructs a k-fold cross-validation of the data
    data can be shuffled for the split
    setting nr_samples to anything else then 0 results in bootstrapped test and validation sets
    bootstrap samples are chosen from the data sets that would be used with regular CV (ie different sets for tast and train!)
    the number of samples denotes the size of the combined train and test set
    """


    def __init__(self, handler, kfolds, shuffle, hide_a, hide_method, cons):
        self.handler = handler
        self.hide_method = hide_method
        self.hide_a = hide_a
        self.cons = cons
        self.kfolds = kfolds
        if kfolds < 2: raise NotImplementedError('{} kfolds is invalid, please use at least a 2-fold cross validation'.format(kfolds))
        if kfolds > handler.idcs(): raise NotImplementedError('{} kfolds is invalid without bootstrapping as only {} data samples are available'.format(kfolds, self.handler.N))
        self.test = {}
        self.train = {}
        self.hidden = {}
        self.reports = {}
        self.cons.centered('SETUP VALIDATION', emptystart=True)
        self.cons.centered('({} folds, {}% {})'.format(kfolds, int(hide_a * 100), hide_method))
        if shuffle:
            idcs = np.random.permutation(handler.idcs()) # all available idcs
        else:
            idcs = np.arange(handler.idcs())
        test_size = handler.idcs() / kfolds
        for fold in range(kfolds):
            name = cv_foldername(fold)
            if not os.path.isdir(name):
                os.makedirs(name)
            if not os.path.isfile(os.path.join(name, '.test_idcs.npy')): # setup is stored to allow running the software multiple times
                idx_s = int(fold * test_size)
                idx_e = int((fold + 1) * test_size)
                # simply split the whole dataset
                idcs_test = idcs[idx_s:idx_e]
                idcs_train = np.append(idcs[:idx_s], idcs[idx_e:])
                np.save(os.path.join(name, '.test_idcs'), idcs_test)
                np.save(os.path.join(name, '.train_idcs'), idcs_train)
            else:
                idcs_test = np.load(os.path.join(name, '.test_idcs.npy'))
                idcs_train = np.load(os.path.join(name, '.train_idcs.npy'))
            self.test[name] = idcs_test
            self.train[name] = idcs_train
            if not os.path.isfile(os.path.join(name, '.' + hide_method.lower() + str(hide_a) + '.npy')):
                hidden_test = handler.hide(idcs_test, hide_method, hide_a, fold)
                np.save(os.path.join(name, '.' + hide_method.lower() + str(hide_a)), hidden_test)
            else:
                hidden_test = np.load(os.path.join(name, '.' + hide_method.lower() + str(hide_a) + '.npy'))
            self.hidden[name] = hidden_test
    

    def get_data(self, split, datatype, slice_shape):
        """
        returns training or test data for a specified CV split, as selected by split and datatype
        """
        if slice_shape == 'img':
            slice_shape = None
            format_to_img = True
        else:
            format_to_img = False
        name = cv_foldername(split)
        if datatype == 'predict':
            idcs = self.test[name]
            hidden = self.hidden[name]
        else: # datatype == 'train'
            idcs = self.train[name]
            hidden = False # no data was artificially hidden
        data = self.handler.extr_data(idcs, slice_shape) # shape will be (n, T, V, D)
        mask = self.handler.extr_mask(idcs, slice_shape)
        obs = np.logical_xor(mask, hidden)
        obs = np.repeat(obs[:, :, :, np.newaxis], self.handler.D, axis=3)
        assert(data.shape == obs.shape)
        assert(len(data.shape) == 4)
        data[np.invert(obs)] = HIDE_VAL # hide all not observed values
        if format_to_img:
            data_old = np.array(data)
            obs_old = np.array(obs)
            data = slices_to_img(data, idcs, self.handler.W, self.handler.H)
            obs = slices_to_img(obs, idcs, self.handler.W, self.handler.H)
        return data.astype(float), obs


    def evaluate_prediction(self, time_train, predicted, pred_name):
        """
        evaluates predicted data for a specified CV split
        pred_name is chosen by method and is used to name evaluation output
        """
        for_storing = []
        self.cons.centered('EVALUATING PREDICTION', emptystart=True)
        for split in range(self.kfolds):
            prediction, time_pred = predicted[split]
            time_train[split] = np.round(time_train[split], 4)
            time_pred = np.round(time_pred, 4)
            full_pred_name = '{}_{}{}'.format(pred_name, self.hide_method.lower(), str(self.hide_a).replace('.', '_'))
            name = cv_foldername(split)
            testidcs = self.test[name]
            data = self.handler.extr_data(testidcs)
            if data.shape != prediction.shape:
                prediction = slices_to_img(prediction, testidcs, self.handler.W, self.handler.H, inverse=True)
            mask = self.handler.extr_mask(testidcs)
            hidden = self.hidden[name]
            with open(os.path.join(name, 'report_{}.csv'.format(full_pred_name)), 'w') as report:
                report.write('t,day,obs,miss,hid,mae,rmse,r2,ssim,t_train,t_pred\n')
                hids = []
                miss = []
                ssims = []
                for t in range(self.handler.T):
                    n_obs = np.count_nonzero(mask[:, t])
                    n_mis = mask[:, t].size - n_obs
                    n_hid, mae, rmse, r2 = compare(prediction[:, t].astype(float), data[:, t].astype(float), hidden[:, t])
                    ssim = calc_ssim(prediction[:, t], data[:, t], hidden[:, t], testidcs, self.handler.W, self.handler.H)
                    if n_hid > 0:
                        hids.append(n_hid)
                        miss.append(n_mis)
                        ssims.append(ssim)
                    report.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(t, self.handler.dates[t], n_obs - n_hid, n_mis, n_hid,
                                                                             mae, rmse, r2, ssim, time_train[split], time_pred))
                hid_total = np.sum(np.array(hids))
                miss_total = np.sum(np.array(miss))
                if hids != []:
                    _, mae_total, rmses_total, r2s_total = compare(prediction.astype(float), data.astype(float), hidden)
                    ssim_total = np.mean(np.array(ssims))
                else:
                    mae_total = 0
                    rmses_total = 0
                    r2s_total = 0
                    ssim_total = 0
                report.write('all,all,{},{},{},{},{},{},{},{},{}\n'.format(hidden.size - hid_total - miss_total, miss_total, hid_total,
                                                                           mae_total, rmses_total, r2s_total, ssim_total, time_train[split], time_pred))
            for_storing.append((prediction, testidcs, hidden))
            versions = write_comparison_report(name)
            self.reports[name] = versions
        self.write_cv_reports()
        move_to_cv()
        self.cons.centered('STORING PREDICTION', emptystart=True)
        self.handler.store_prediction(for_storing, full_pred_name)


    def write_cv_reports(self):
        """aggregates the reports in the individual CV splits"""
        split_versions = list(self.reports.values())
        if split_versions == []:
            return
        if all([versions == split_versions[0] for versions in split_versions]): # might not be true if some methods are run at the same time
            for version in split_versions[0]:
                report_data = np.zeros((self.handler.T + 1, 11, len(self.reports)))
                for split in range(len(self.reports)):
                    report_data[:, :, split] = np.genfromtxt(os.path.join(list(self.reports.keys())[split], version), delimiter=',')[1:, :]
                with open(version, 'w') as report:
                    report.write('t,day,obs,miss,hid,mae,sd_mae,rmse,rmse_std,r2,r2_std,ssim,ssim_std,t_train,t_pred\n')
                    t_train, t_pred = np.mean(report_data[-1, -2:, :], axis=1)
                    for t in range(self.handler.T):
                        hid_l = report_data[t, 4, :]
                        obs = int(np.sum(report_data[t, 2, :]))
                        mis = int(np.sum(report_data[t, 3, :]))
                        hid = int(np.sum(hid_l))
                        if np.sum(hid_l) == 0:
                            mae = mae_sd = rmse = rmse_std = r2 = r2_std = ssim = ssim_std = 0
                        else:
                            weights = hid_l / np.sum(hid_l)
                            mae, mae_sd = weighted_avg_and_std(report_data[t, 5, :], weights)
                            rmse, rmse_std = weighted_avg_and_std(report_data[t, 6, :], weights)
                            r2, r2_std = weighted_avg_and_std(report_data[t, 7, :], weights)
                            ssim, ssim_std = weighted_avg_and_std(report_data[t, 8, :], weights)
                        report.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(t, self.handler.dates[t], obs, mis, hid, mae, mae_sd, rmse, rmse_std, r2, r2_std, ssim, ssim_std, t_train, t_pred))
                    hid_l = report_data[-1, 4, :]
                    obs = int(np.sum(report_data[-1, 2, :]))
                    mis = int(np.sum(report_data[-1, 3, :]))
                    hid = int(np.sum(hid_l))
                    if np.sum(hid_l) == 0:
                        weights = None
                    else:
                        weights = hid_l / np.sum(hid_l)
                    mae, mae_sd = weighted_avg_and_std(report_data[-1, 5, :], weights)
                    rmse, rmse_std = weighted_avg_and_std(report_data[-1, 6, :], weights)
                    r2, r2_std = weighted_avg_and_std(report_data[-1, 7, :], weights)
                    ssim, ssim_std = weighted_avg_and_std(report_data[-1, 8, :], weights)
                    report.write('all,all,{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(obs, mis, hid, mae, mae_sd, rmse, rmse_std, r2, r2_std, ssim, ssim_std, t_train, t_pred))
            write_comparison_report(os.getcwd())
            plot_error_over_time(split_versions[0])
            plot_error_over_time(split_versions[0], cumulated=False)
            plot_error_for_prior(split_versions[0])
            plot_cloud_stats()
