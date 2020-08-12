import math

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from probgf.methods_simple import SpatioTemporalBase


class NSPI(SpatioTemporalBase):
    """
    Implementation of the Neighborhood Similar Pixel Interpolator
    Most code is taken from the authors code ("NSPI_time_series_Python-updated_20200312.zip")
    More info:
    Jin Chena, Xiaolin Zhua, James E. Vogelmannc, Feng Gaod, and Suming Jinf
    "A simple and effective method forfilling gaps in Landsat ETM+ SLC-off images"
    (2011)
    https://xiaolinzhu.weebly.com/open-source-code.html
    https://doi.org/10.1016/j.rse.2010.12.010
    """


    @classmethod
    def default_config(cls):
        return '30,4,0.0,10000.0'


    def __init__(self, config, dates, console, emiters):
        super().__init__(config, dates, console, emiters)
        self.slice_shape = 'img'


    def name(self):
        return 'NSPI_MINSIM{}_NCLASS{}_DNMIN{}_DNMAX{}'.format(
            self.min_similar,
            self.num_class,
            self.DN_min,
            self.DN_max,
        ).replace('.', '_')


    def configure(self, config):
        "instead of parsing the parameters from the yaml config file, we here allow to set parameters via command line argument"
        parts = config.split(',')
        try:
            self.min_similar = int(parts[0])  # set the minimum sample size of similar pixels, 20~40 recommended
            self.num_class   = int(parts[1])  # set the number of classes (given empirically)
            # num_band = int(parts[2] not needed
            self.DN_min      = float(parts[2])  # set the range of DN value of the image,If byte, 0 and 255
            self.DN_max      = float(parts[3])
            # patch_long = param['patch_long'] not needed
            # DOY = param['DOY'] not needed
        except (ValueError, IndexError):
            raise RuntimeError('Invalid config "{}".\n{} needs a comma-seperated list of the following values:\n'.format(config, self.__class__.method_id()) + \
            '  min_similar : minimum sample size of similar pixels, 20~40 recommended (int)\n' + \
            '  num_class   : number of classes (given empirically) (int)\n' + \
            '  DN_min      : range min of DN value of the image (float)\n' + \
            '  DN_max      : range max of DN value of the image (float)')
        

    def run_training(self, data, obs, split, progr_train):
        super().run_training(data, obs, split, progr_train)


    def run_prediction(self, to_pred, obs, split, progr_pred):
        """
        NSPI logic, taken from the original "NSPI_TIMESERIES.py" script (removed boilerplate code with comments)
        Implementation was also changed to not subprocess smaller patches of the data, but instead predict the whole thing
        """
        # reshape to NSPI format+
        orig_shape = to_pred.shape
        mask = np.moveaxis(obs[:, :, :, 0], 2, 0)
        mask = np.invert(mask)
        to_pred = np.moveaxis(to_pred.reshape(obs.shape[0], obs.shape[1], obs.shape[2] * obs.shape[3]), 2, 0)


        # ******************************************************************************************************
        # *******************************Set parameters and read input data*************************************

        # starttime = datetime.datetime.now()  # initial time

        # root = tk.Tk()
        # root.withdraw()

        # Please set the appropriate parameters for your own data in parameters_timeseries.yaml before open this file!
        # f = open(filedialog.askopenfilename(title=u"Open the parameter settings file:"))
        # param = yaml.safe_load(f)
        min_similar = self.min_similar # param['min_similar']  # set the minimum sample size of similar pixels, 20~40 recommended
        num_class = self.num_class # param['num_class']  # set the number of classes (given empirically)
        num_band = to_pred.shape[0] // mask.shape[0] # param['num_band']  # set the number of bands in each image
        DN_min = self.DN_min # param['DN_min']  # set the range of DN value of the image,If byte, 0 and 255
        DN_max = self.DN_max # param['DN_max']
        DOY = self.dates # param['DOY']  # Day of year of each image in the time-series

        # # set the folder for storing temporary output results
        # temp_file = filedialog.askdirectory(title=u"Set the temporary folder")

        # # open the SLC-off ETM+ time-series image
        # path1 = filedialog.askopenfilename(title=u"Open the SLC_off ETM+ time-series image:")
        # suffix = os.path.splitext(path1)[-1]
        # nl, ns, data_off = read_raster(path1)
        # orig_ns = ns
        # orig_nl = nl
        # fp = gdal.Open(path1)
        # nb = fp.RasterCount

        # # divide the whole scene into blocks
        # n_nl = math.ceil(nl / patch_long)
        # n_ns = math.ceil(ns / patch_long)
        # n_image = math.ceil(nb / num_band)
        n_image = to_pred.shape[0] // num_band

        # ind_patch1 = np.zeros((n_nl * n_ns, 4), dtype=np.int)
        # ind_patch = np.zeros((n_nl * n_ns, 4), dtype=np.int)
        # location = np.zeros((n_nl * n_ns, 4), dtype=np.int)

        # for i_ns in range(0, n_ns):
        #     for i_nl in range(0, n_nl):
        #         ind_patch1[n_ns * i_nl + i_ns, 0] = i_ns * patch_long
        #         ind_patch[n_ns * i_nl + i_ns, 0] = np.max([0, ind_patch1[n_ns * i_nl + i_ns, 0] - 10])
        #         location[n_ns * i_nl + i_ns, 0] = ind_patch1[n_ns * i_nl + i_ns, 0] - ind_patch[n_ns * i_nl + i_ns, 0]

        #         ind_patch1[n_ns * i_nl + i_ns, 1] = np.min([ns - 1, (i_ns + 1) * patch_long - 1])
        #         ind_patch[n_ns * i_nl + i_ns, 1] = np.min([ns - 1, ind_patch1[n_ns * i_nl + i_ns, 1] + 10])
        #         location[n_ns * i_nl + i_ns, 1] = ind_patch1[n_ns * i_nl + i_ns, 1] - ind_patch1[n_ns * i_nl + i_ns, 0] + location[n_ns * i_nl + i_ns, 0]

        #         ind_patch1[n_ns * i_nl + i_ns, 2] = i_nl * patch_long
        #         ind_patch[n_ns * i_nl + i_ns, 2] = np.max([0, ind_patch1[n_ns * i_nl + i_ns, 2] - 10])
        #         location[n_ns * i_nl + i_ns, 2] = ind_patch1[n_ns * i_nl + i_ns, 2] - ind_patch[n_ns * i_nl + i_ns, 2]

        #         ind_patch1[n_ns * i_nl + i_ns, 3] = np.min([nl - 1, (i_nl + 1) * patch_long - 1])
        #         ind_patch[n_ns * i_nl + i_ns, 3] = np.min([nl - 1, ind_patch1[n_ns * i_nl + i_ns, 3] + 10])
        #         location[n_ns * i_nl + i_ns, 3] = ind_patch1[n_ns * i_nl + i_ns, 3] - ind_patch1[ n_ns * i_nl + i_ns, 2] + location[n_ns * i_nl + i_ns, 2]

        # tempoutname = temp_file + '\\temp_target'

        # for isub in range(0, n_nl * n_ns):
        #     col1 = ind_patch[isub, 0]
        #     col2 = ind_patch[isub, 1]
        #     row1 = ind_patch[isub, 2]
        #     row2 = ind_patch[isub, 3]
        #     data = data_off[:, row1:row2 + 1, col1:col2 + 1]
        #     out_name = tempoutname + str(isub + 1) + suffix
        #     fp = path1
        #     writeimage(data, out_name, fp)

        # # open the mask of SLC-off ETM+ image: 0 is good pixels, 1 gaps 2 shadows 3 clouds
        # path2 = filedialog.askopenfilename(title=u"Open the mask image):")
        # _, _, data_gap = read_raster(path2)

        # tempoutname = temp_file + '\\temp_mask'

        # for isub in range(0, n_nl * n_ns):
        #     col1 = ind_patch[isub, 0]
        #     col2 = ind_patch[isub, 1]
        #     row1 = ind_patch[isub, 2]
        #     row2 = ind_patch[isub, 3]
        #     data = data_gap[:, row1:row2 + 1, col1:col2 + 1]
        #     out_name = tempoutname + str(isub + 1) + suffix
        #     fp = path1
        #     writeimage(data, out_name, fp)

        # # **********************************************************************************************************************
        # # ***************************************begin process the gap for each block*******************************************

        # print('there are total', n_ns * n_nl, ' blocks')

        # for isub in range(0, n_nl * n_ns):

        # open each block image
        # FileName = temp_file + '\\temp_target' + str(isub + 1) + suffix
        # nl, ns, fine1_all = read_raster(FileName)
        nl = to_pred.shape[1]
        ns = to_pred.shape[2]
        fine1_all = to_pred
        # place the new image value
        # fine0 = fine1_all[:, location[isub, 2]:location[isub, 3] + 1, location[isub, 0]:location[isub, 1] + 1]
        fine0 = np.array(fine1_all)

        # FileName = temp_file + '\\temp_mask' + str(isub + 1) + suffix
        # _, _, mask = read_raster(FileName)

        # place the mark value
        mark = np.zeros(mask.shape, dtype=int)

        # classify all images by k-means
        classes = np.zeros([n_image, nl, ns]).astype(int)
        for i_img in range(0, n_image):
            imagei = fine1_all[i_img * num_band:(i_img * num_band + num_band), :, :]
            gapi = mask[i_img, :, :]
            ind_bad = np.where(gapi != 0)
            num_bad = int(int(np.size(ind_bad)) / len(ind_bad))
            if num_bad > 0:
                for iband in range(0, num_band):
                    temp = imagei[iband, :, :]
                    temp[ind_bad] = 0.0
                    imagei[iband, :, :] = temp

            # tempnamei = temp_file + '\\imagei' + suffix
            # fp = path1
            # writeimage(imagei, tempnamei, fp)

            # _, _, imagei_new = read_raster_new(tempnamei)
            imagei_new = np.moveaxis(np.array(imagei), 0, 2)
            imagei_new = np.maximum(imagei_new, 0)

            # parameter of kmeans
            CHANGE_THRESH = .02
            NUM_CLASSES = num_class + 1
            ITERATIONS = 20
            # out_bname = 'kmeans'
            # out_name = temp_file + '\\class_kmeans'
            new_shape = (imagei_new.shape[0] * imagei_new.shape[1], num_band)
            new_imagei = imagei_new[:, :, :num_band].reshape(new_shape)
            clf = MiniBatchKMeans(n_clusters=NUM_CLASSES, random_state=0, max_iter=ITERATIONS, reassignment_ratio=0.02)
            class_img = clf.fit(new_imagei).labels_
            labels = class_img.reshape(imagei_new[:, :, 0].shape)
            classes[i_img, :, :] = labels + 1

        # process each image
        for i_img in range(0, n_image):
            fine1 = fine1_all[i_img * num_band:(i_img * num_band + num_band), :, :]
            gap = mask[i_img, :, :]
            temp_dis = np.abs(np.array(DOY) - DOY[i_img])
            order = np.argsort(temp_dis)
            note_finish = 0
            i_input = 1

            while note_finish != 1 and i_input <= n_image - 1:

                fine2 = fine1_all[order[i_input] * num_band:(order[i_input] * num_band + num_band), :, :]
                mask2 = mask[order[i_input], :, :]
                class2 = classes[order[i_input], :, :]

                similar_th_band = np.zeros(num_band).astype(float)
                ind_nogrand = np.where(mask2 == 0)
                c_nogrand = int(np.size(ind_nogrand) / len(ind_nogrand))
                if c_nogrand > 0:
                    for iband in range(0, num_band):
                        # compute the threshold of similar pixel
                        fine2_ind = fine2[iband, :, :]
                        similar_th_band[iband] = np.std(fine2_ind[ind_nogrand]) * 2.0 / float(num_class)
                similar_th = np.mean(similar_th_band)

                # for i in range(location[isub, 0], location[isub, 1] + 1):
                #     for j in range(location[isub, 2], location[isub, 3] + 1):
                for i in range(to_pred.shape[2]):
                    for j in range(to_pred.shape[1]):

                        if 0 < gap[j, i] < 10:
                            if mask2[j, i] == 0:

                                classij = class2[j, i]
                                extent = np.ceil(0.5 * (np.sqrt(min_similar) - 1))  # compute the minimum window size
                                max_window = np.round(max([nl, ns]) * 0.25)  # maximum size of half seach window is 1/4 of image size

                                #  find the possible largest window
                                gap_profiles = np.hstack((gap[j, :], gap[:, i]))
                                input_profile = np.hstack((mask2[j, :], mask2[:, i]))
                                spatial_position = np.abs(np.hstack((np.arange(ns) - i, np.arange(nl) - j)))
                                ind_outcloud = np.logical_and(gap_profiles == 0, input_profile == 0)
                                num_outcloud = np.sum(ind_outcloud)
                                if num_outcloud > 0:
                                    max_window_p = np.min(spatial_position[ind_outcloud])  # find the possibel maximum half window
                                    max_window_p = np.min([max_window, max_window_p + 50])
                                else:
                                    max_window_p = max_window

                                start_w = np.ceil(0.5 * (np.sqrt(min_similar) - 1))
                                # check large cloud patches to avoid searching similar pixels for it
                                if max_window_p == max_window:
                                    a1 = int(np.max([0, i - max_window]))
                                    a2 = int(np.min([ns - 1, i + max_window]))
                                    b1 = int(np.max([0, j - max_window]))
                                    b2 = int(np.max([nl - 1, j + max_window]))

                                    sub_gap = gap[b1:b2 + 1, a1:a2 + 1]
                                    sub_class = class2[b1:b2 + 1, a1:a2 + 1]
                                    sub_mask2 = mask2[b1:b2 + 1, a1:a2 + 1]
                                    ind_class = np.logical_and(np.logical_and(sub_gap == 0, sub_mask2 == 0), sub_class == classij)
                                    c_common = np.sum(ind_class)
                                    if c_common < 2.0 * min_similar:
                                        start_w = max_window_p

                                # use binary search to find the best window size
                                end_w = max_window_p
                                ind_success = 0
                                while ind_success == 0 and start_w <= end_w:
                                    mid = np.floor(start_w + (end_w - start_w) / 2.0)
                                    a1 = int(np.max([0, i - mid]))
                                    a2 = int(np.min([ns - 1, i + mid]))
                                    b1 = int(np.max([0, j - mid]))
                                    b2 = int(np.min([nl - 1, j + mid]))
                                    sub_gap = gap[b1:b2 + 1, a1:a2 + 1]
                                    sub_mask2 = mask2[b1:b2 + 1, a1:a2 + 1]
                                    sub_class = class2[b1:b2 + 1, a1:a2 + 1]
                                    ind_common = np.logical_and(np.logical_and(sub_gap == 0, sub_mask2 == 0), sub_class == classij)
                                    c_common = np.sum(ind_common)
                                    if c_common > 3.0 * min_similar:  # search left range
                                        end_w = mid - 1
                                    else:
                                        if c_common < 2.0 * min_similar:  # search right range
                                            start_w = mid + 1
                                        else:
                                            ind_success = 1

                                extent = mid
                                sub_off = fine1[:, b1:b2 + 1, a1:a2 + 1]
                                sub_on = fine2[:, b1:b2 + 1, a1:a2 + 1]

                                col_temp = np.arange((a2 - a1 + 1)).astype(int)
                                col_num = int(b2 - b1 + 1)
                                col_wind = np.tile(col_temp, (col_num, 1))
                                row_temp = np.arange(b2 - b1 + 1).astype(int)
                                row_num = int(a2 - a1 + 1)
                                row_wind = np.repeat(row_temp, row_num).reshape(-1, row_num)
                                it = i - a1
                                jt = j - b1

                                if c_common >= 2.0 * min_similar:  # if same-class pixels more than required sample size

                                    disi = np.sqrt((it - col_wind[ind_common]) ** 2 + (jt - row_wind[ind_common]) ** 2)
                                    sub_on_common = np.zeros((num_band, c_common)).astype(float)
                                    sub_off_common = np.zeros((num_band, c_common)).astype(float)
                                    for iband in range(0, num_band):
                                        sub_on_common[iband, :] = (sub_on[iband, :, :])[ind_common] - fine2[iband, j, i]
                                        sub_off_common[iband, :] = (sub_on[iband, :, :])[ind_common] - sub_off[iband, :, :][ind_common]

                                    rmsei = np.sqrt(np.mean(sub_on_common ** 2, axis=0)) + 0.0001
                                    rmse12 = np.sqrt(np.mean(sub_off_common ** 2, axis=0)) + 0.0001

                                    order_rmse = np.argsort(rmsei)
                                    ind_similar = order_rmse[0:min_similar]   # find similar pixels

                                    similar_rmse = rmsei[ind_similar]
                                    similar_rmse12 = rmse12[ind_similar]
                                    similar_dis = disi[ind_similar]
                                    C_D = np.multiply(similar_rmse, similar_dis)
                                    weight = (1.0 / C_D) / np.sum(1.0 / C_D)
                                    T_1 = np.mean(similar_rmse)
                                    T_2 = np.mean(similar_rmse12)
                                    W_T1 = T_2 / (T_1 + T_2)
                                    W_T2 = T_1 / (T_1 + T_2)

                                    for iband in range(0, num_band):
                                        similar_on = (sub_on[iband, :, :])[ind_common][ind_similar]
                                        similar_off = (sub_off[iband, :, :])[ind_common][ind_similar]
                                        predict_1 = np.sum(np.multiply(similar_off, weight))
                                        predict_2 = fine2[iband, j, i] + np.sum(np.multiply((similar_off - similar_on), weight))
                                        if DN_min < predict_2 < DN_max:
                                            # fine0[i_img * num_band + iband, j - location[isub, 2], i - location[
                                            #     isub, 0]] = W_T1 * predict_1 + W_T2 * predict_2
                                            fine0[i_img * num_band + iband, j, i] = W_T1 * predict_1 + W_T2 * predict_2
                                        else:
                                            # fine0[i_img * num_band + iband, j - location[isub, 2], i - location[
                                            #     isub, 0]] = predict_1
                                            fine0[i_img * num_band + iband, j, i] = predict_1
                                    note = 1
                                    # mark[i_img, j - location[isub, 2], i - location[isub, 0]] = 1 + 10 * i_input
                                    mark[i_img, j, i] = 1 + 10 * i_input
                                    gap[j, i] = -2

                                else:  # if no enough same-class pixels exist
                                    # IF reach the maximum window, use all the same-class pixels
                                    if c_common >= 3 and extent >= max_window:
                                        disi = np.sqrt((it - col_wind[ind_common]) ** 2 + (jt - row_wind[ind_common]) ** 2)
                                        sub_on_common = np.zeros((num_band, c_common)).astype(float)
                                        sub_off_common = np.zeros((num_band, c_common)).astype(float)
                                        for iband in range(0, num_band):
                                            sub_on_common[iband, :] = (sub_on[iband, :, :])[ind_common] - fine2[iband, j, i]
                                            sub_off_common[iband, :] = (sub_on[iband, :, :])[ind_common] - sub_off[iband, :, :][ind_common]

                                        rmsei = np.sqrt(np.mean(sub_on_common ** 2, axis=0)) + 0.0001
                                        rmse12 = np.sqrt(np.mean(sub_off_common ** 2, axis=0)) + 0.0001

                                        similar_rmse = rmsei
                                        similar_rmse12 = rmse12
                                        similar_dis = disi
                                        C_D = np.multiply(similar_rmse, similar_dis)
                                        weight = (1.0 / C_D) / np.sum(1.0 / C_D)
                                        T_1 = np.mean(similar_rmse)
                                        T_2 = np.mean(similar_rmse12)
                                        W_T1 = T_2 / (T_1 + T_2)
                                        W_T2 = T_1 / (T_1 + T_2)
                                        for iband in range(0, num_band):
                                            similar_on = (sub_on[iband, :, :])[ind_common]
                                            similar_off = (sub_off[iband, :, :])[ind_common]
                                            predict_1 = np.sum(np.multiply(similar_off, weight))
                                            predict_2 = fine2[iband, j, i] + np.sum(np.multiply((similar_off - similar_on), weight))
                                            if DN_min < predict_2 < DN_max:
                                                # fine0[i_img * num_band + iband, j - location[isub, 2], i - location[
                                                #     isub, 0]] = W_T1 * predict_1 + W_T2 * predict_2
                                                fine0[i_img * num_band + iband, j, i] = W_T1 * predict_1 + W_T2 * predict_2
                                            else:
                                                # fine0[i_img * num_band + iband, j - location[isub, 2], i - location[
                                                #     isub, 0]] = predict_1
                                                fine0[i_img * num_band + iband, j, i] = predict_1
                                        # mark[i_img, j - location[isub, 2], i - location[isub, 0]] = 2 + 10 * i_input
                                        mark[i_img, j, i] = 2 + 10 * i_input
                                        note = 1
                                        gap[j, i] = -2
                                    else:
                                        # if no same-class pixels exist when reaching the maximum window, directly replace the pixel by input image
                                        # only use temporal interpolation if it have observations before and after
                                        first_half = np.max([0, i_img - 1])
                                        second_half = np.min([i_img + 1, n_image - 1])
                                        ind_good_before = np.where(mask[0:first_half+1, j, i] == 0)
                                        num_good_before = int(int(np.size(ind_good_before)) / len(ind_good_before))
                                        ind_good_after = np.where(mask[second_half:n_image, j, i] == 0)
                                        num_good_after = int(int(np.size(ind_good_after)) / len(ind_good_after))
                                        if num_good_before > 0 and num_good_after > 0:
                                            i_before = ind_good_before[0][num_good_before - 1]
                                            i_after = ind_good_after[0][0] + i_img + 1
                                            T1 = DOY[i_img] - DOY[int(i_before)]
                                            T2 = DOY[int(i_after)] - DOY[i_img]
                                            for iband in range(0, num_band):
                                                before_band = fine1_all[i_before * num_band + iband, j, i]
                                                after_band = fine1_all[i_after * num_band + iband, j, i]
                                                # fine0[i_img * num_band + iband, j - location[isub, 2], i - location[isub, 0]] = (T2 * before_band + T1 * after_band) / (T2 + T1)
                                                fine0[i_img * num_band + iband, j, i] = (T2 * before_band + T1 * after_band) / (T2 + T1)
                                        else:
                                            # fine0[i_img * num_band:((i_img + 1) * num_band), j - location[isub, 2], i - location[isub, 0]] = fine2[:, j, i]
                                            fine0[i_img * num_band:((i_img + 1) * num_band), j, i] = fine2[:, j, i]

                                        # mark[i_img, j - location[isub, 2], i - location[isub, 0]] = 3 + 10 * i_input
                                        mark[i_img, j, i] = 3 + 10 * i_input
                                        note = 1
                                        gap[j, i] = -2

                # ind_unfilled = np.where(gap[location[isub, 2]:location[isub, 3] + 1, location[isub, 0]:location[isub, 1] + 1] > 0)
                ind_unfilled = np.where(gap > 0)
                c_unfilled = int(int(np.size(ind_unfilled)) / len(ind_unfilled))
                if c_unfilled == 0:
                    note_finish = 1
                else:
                    i_input = i_input + 1

            # print('finish', i_img + 1, "image in the time-series")
            progr_pred[split] = float(i_img + 1) / n_image * 100
            self.cons.progress(progr_pred, split)
            # from probgf.visualization import visualize_rgb
            # img = np.moveaxis(fine0[i_img * num_band:((i_img + 1) * num_band)], 0, 2)
            # visualize_rgb(img, 'test_{}_{}.png'.format(split, i_img), np.array([0, 1, 2]), np.array([0, 0, 0]), np.array([3000, 3000, 3000]))

        # reshape to probgf format
        fine0 = np.moveaxis(fine0, 0, 2).reshape(orig_shape)
        super().run_prediction(to_pred, obs, split, progr_pred) # for final console output

        return fine0

        # print("finished", isub + 1, "block")
        # tempoutname1 = temp_file + '\\temp_filled' + str(isub + 1) + suffix
        # tempoutname2 = temp_file + '\\temp_mark' + str(isub + 1) + suffix
        # fp = path1
        # writeimage(fine0, tempoutname1, fp)
        # writeimage(mark, tempoutname2, fp)

        # # *******************************************************************************************
        # # *******************************mosaic all the filled patch*********************************

        # datalist = []
        # minx_list = []
        # maxX_list = []
        # minY_list = []
        # maxY_list = []

        # for isub in range(0, n_ns * n_nl):
        #     out_name = temp_file + '\\temp_filled' + str(isub + 1) + suffix
        #     datalist.append(out_name)
        #     col1 = ind_patch1[isub, 0]
        #     col2 = ind_patch1[isub, 1]
        #     row1 = ind_patch1[isub, 2]
        #     row2 = ind_patch1[isub, 3]

        #     minx_list.append(col1)
        #     maxX_list.append(col2)
        #     minY_list.append(row1)
        #     maxY_list.append(row2)

        # minX = min(minx_list)
        # maxX = max(maxX_list)
        # minY = min(minY_list)
        # maxY = max(maxY_list)

        # xOffset_list = []
        # yOffset_list = []
        # i = 0
        # for data in datalist:
        #     ds = gdal.Open(data)
        #     transform = ds.GetGeoTransform()
        #     pixelWidth = transform[1]
        #     pixelHeight = transform[5]
        #     xOffset = int((minx_list[i] - minX) / pixelWidth)
        #     yOffset = int((minY_list[i] - minY) / pixelHeight)
        #     xOffset_list.append(xOffset)
        #     yOffset_list.append(yOffset)
        #     i += 1

        # in_ds = gdal.Open(path1)
        # path = os.path.splitext(path1)[0] + "_fill_NSPI_all" + suffix
        # if suffix == '.tif':
        #     driver = gdal.GetDriverByName("GTiff")
        # elif suffix == "":
        #     driver = gdal.GetDriverByName("ENVI")
        # dataset = driver.Create(path, orig_ns, orig_nl, nb, gdal.GDT_Float32)
        # i = 0
        # for data in datalist:
        #     nl, ns, datavalue = read_raster(data)
        #     for j in range(0, nb):
        #         dataset.GetRasterBand(j + 1).WriteArray(datavalue[j, :, :], xOffset_list[i], yOffset_list[i])
        #     i += 1

        # geoTransform = in_ds.GetGeoTransform()
        # dataset.SetGeoTransform(geoTransform)
        # proj = in_ds.GetProjection()
        # dataset.SetProjection(proj)

        # # mosiac all the mask patch
        # masklist = []
        # for isub in range(0, n_ns * n_nl):
        #     out_name = temp_file + '\\temp_mark' + str(isub + 1) + suffix
        #     masklist.append(out_name)

        # n_mask = int(n_image)
        # in_ds = gdal.Open(path1)
        # path = os.path.splitext(path1)[0] + "_fill_quality_flag_all" + suffix
        # if suffix == '.tif':
        #     driver = gdal.GetDriverByName("GTiff")
        # elif suffix == "":
        #     driver = gdal.GetDriverByName("ENVI")
        # maskset = driver.Create(path, orig_ns, orig_nl, n_mask, gdal.GDT_Float32)
        # i = 0
        # for mask in masklist:
        #     nl, ns, maskvalue = read_raster(mask)
        #     for j in range(0, n_mask):
        #         maskset.GetRasterBand(j + 1).WriteArray(maskvalue[j, :, :], xOffset_list[i], yOffset_list[i])
        #     i += 1

        # geoTransform = in_ds.GetGeoTransform()
        # maskset.SetGeoTransform(geoTransform)
        # proj = in_ds.GetProjection()
        # maskset.SetProjection(proj)

        # endtime = datetime.datetime.now()
        # print("Finish mosaic image patches!")
        # print('Time used: ', (endtime - starttime).seconds, 'seconds')
