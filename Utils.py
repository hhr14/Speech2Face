import math
import numpy as np
import scipy.signal
import random
import os
from sklearn import preprocessing
import argparse
import pickle
import gc
import keras
import math
import tensorflow as tf
from keras.models import load_model
from model import BRNN, WaveNet, TacotronDecoder, CNN
from module import ZoneoutLSTMCell
from keras.utils.generic_utils import CustomObjectScope


class CNNDataGenerator(keras.utils.Sequence):
    def __init__(self, hparams, ppg, fwh, step_per_epoch):
        self.ppg = ppg
        self.fwh = fwh
        self.batch_size = hparams.batch_size
        self.window_size = hparams.CNN_window_size
        self.continuous = hparams.CNN_continuous
        self.step_per_epoch = step_per_epoch
        self.hparams = hparams
        self.sample_list = []
        self.build_random_list()

    def __len__(self):
        return self.step_per_epoch

    def __getitem__(self, index):
        ppg_input = np.zeros((self.batch_size, self.window_size, self.ppg[0].shape[-1]))
        fwh_output = np.zeros((self.batch_size, self.window_size, self.fwh[0].shape[-1]))
        # unuse of context ??
        # myfwh_input = np.zeros((self.batch_size, self.input_time, self.fwh[0].shape[-1]))
        # myfwh_output = np.zeros((self.batch_size, self.fwh[0].shape[-1]))
        # condition_ppg = np.zeros((self.batch_size, self.input_time,
        #                           self.ppg[0].shape[-1] * self.context_size))

        if self.hparams.CNN_continuous is True:
            random_index = np.random.randint(0, len(self.sample_list))
            id = self.sample_list[random_index][0]
            begin = self.sample_list[random_index][1]
            for i in range(self.batch_size):
                ppg_input[i] = self.ppg[id][begin + i: begin + i + self.window_size]
                fwh_output[i] = self.fwh[id][begin + i: begin + i + self.window_size]
        else:
            for i in range(self.batch_size):
                random_index = np.random.randint(0, len(self.sample_list))
                id = self.sample_list[random_index][0]
                begin = self.sample_list[random_index][1]
                ppg_input[i] = self.ppg[id][begin: begin + self.window_size]
                fwh_output[i] = self.fwh[id][begin: begin + self.window_size]

        ppg_input = np.expand_dims(ppg_input, axis=2)
        fwh_output = np.expand_dims(fwh_output, axis=2)
        # change to image to use conv_transpose

        return ppg_input, fwh_output

    def build_random_list(self):
        if self.hparams.CNN_continuous is True:
            sample_size = self.window_size + self.batch_size
        else:
            sample_size = self.window_size

        for i in range(len(self.ppg)):
            if len(self.ppg[i]) > sample_size:
                for j in range(len(self.ppg[i]) - sample_size):
                    self.sample_list.append([i, j])
        print("sample_list length:", len(self.sample_list))


class wavenetDataGenerator(keras.utils.Sequence):
    def __init__(self, hparams, ppg, fwh):
        self.input_time = hparams.wavenet_input_time
        self.batch_size = hparams.batch_size
        self.ppg = ppg
        self.fwh = fwh
        self.context_size = hparams.wavenet_context_size
        self.is_neighbor = hparams.wavenet_context_neighbor
        self.step_per_epoch = hparams.wavenet_step_per_epoch
        self.sample_list = []
        self.build_random_list()

    def __len__(self):
        return self.step_per_epoch

    def __getitem__(self, index):
        myfwh_input = np.zeros((self.batch_size, self.input_time, self.fwh[0].shape[-1]))
        myfwh_output = np.zeros((self.batch_size, self.fwh[0].shape[-1]))
        condition_ppg = np.zeros((self.batch_size, self.input_time,
                                  self.ppg[0].shape[-1] * self.context_size))
        for i in range(self.batch_size):
            random_index = np.random.randint(0, len(self.sample_list))
            fwh_id = self.sample_list[random_index][0]
            fwh_begin = self.sample_list[random_index][1]
            myfwh_input[i] = self.fwh[fwh_id][fwh_begin: fwh_begin + self.input_time]
            myfwh_output[i] = self.fwh[fwh_id][fwh_begin + self.input_time]
            temp_ppg = modify_output_feature(self.ppg[fwh_id], mode='window',
                                             is_neighbor=self.is_neighbor,
                                             window_size=self.context_size)
            condition_ppg[i] = temp_ppg[fwh_begin + 1: fwh_begin + self.input_time + 1]
            #  condition_ppg is one step front of input fwh !!!!!!!!

        return [myfwh_input, condition_ppg], myfwh_output

    def build_random_list(self):
        for i in range(len(self.ppg)):
            if len(self.ppg[i]) > self.input_time:
                for j in range(len(self.ppg[i]) - self.input_time):
                    self.sample_list.append([i, j])
        print("sample_list length:", len(self.sample_list))


class dataGenerator(keras.utils.Sequence):
    def __init__(self, ppg, fwh, hparams):
        self.ppg = ppg
        self.fwh = fwh
        self.hparams = hparams
        self.batch_size = int(hparams.batch_size)

    def __len__(self):
        return len(self.ppg) // self.batch_size

    def __getitem__(self, index):
        pad_ppg = np.zeros((1, self.ppg[0].shape[-1]), dtype=float)
        pad_fwh = np.zeros((1, self.fwh[0].shape[-1]), dtype=float)
        max_time = -1
        self.data_length = np.zeros((self.batch_size, self.fwh[0].shape[-1]))
        for i in range(self.batch_size):
            cindex = index * self.batch_size + i
            max_time = max(max_time, len(self.ppg[cindex]))
            if len(self.ppg[cindex]) == 0:
                print(cindex)
            self.data_length[i] = 1 / len(self.ppg[cindex])
        #  get max_time.

        self.myppg = np.zeros((self.batch_size, max_time, self.ppg[0].shape[-1]))
        self.myfwh = np.zeros((self.batch_size, max_time, self.fwh[0].shape[-1]))
        self.mean_fwh = np.zeros((self.batch_size, 1, self.fwh[0].shape[-1]))
        self.data_mask = np.zeros((self.batch_size, max_time))
        self.one_mask = np.ones((self.batch_size, 1))
        for i in range(self.batch_size):
            cindex = index * self.batch_size + i
            pad_ppg_matrix = np.tile(pad_ppg, (max_time - len(self.ppg[cindex]), 1))
            pad_fwh_matrix = np.tile(pad_fwh, (max_time - len(self.fwh[cindex]), 1))
            self.myppg[i] = np.concatenate((self.ppg[cindex], pad_ppg_matrix), axis=0)
            self.myfwh[i] = np.concatenate((self.fwh[cindex], pad_fwh_matrix), axis=0)
            self.data_mask[i] = np.concatenate((np.ones(len(self.ppg[cindex])),
                                                np.zeros(max_time - len(self.ppg[cindex]))))
            if self.hparams.add_mean is True:
                self.mean_fwh[i][0] = np.mean(self.fwh[cindex], axis=0)
        if self.hparams.add_mean is True:
            output = [self.myfwh, self.mean_fwh]
            mask = [self.data_mask, self.one_mask]
            input_ = [self.myppg, self.data_length]
        else:
            output = self.myfwh
            mask = self.data_mask
            input_ = self.myppg
        return input_, output, mask

    def on_epoch_end(self):
        index = [i for i in range(len(self.ppg))]
        random.shuffle(index)
        self.ppg = self.ppg[index]
        self.fwh = self.fwh[index]


class TacotronGenerator(keras.utils.Sequence):
    def __init__(self, ppg, fwh, hparams):
        self.ppg = ppg
        self.fwh = fwh
        self.hparams = hparams

    def __len__(self):
        return len(self.ppg) // self.hparams.batch_size

    def __getitem__(self, index):
        pad_ppg = np.zeros((1, self.ppg[0].shape[-1] * self.hparams.Tacotron_context_size), dtype=float)
        pad_fwh = np.zeros((1, self.fwh[0].shape[-1]), dtype=float)

        # pad ppg and fwh with -1

        max_time = -1
        data_length = np.zeros((self.hparams.batch_size, 1))
        for i in range(self.hparams.batch_size):
            cindex = index * self.hparams.batch_size + i
            max_time = max(max_time, len(self.ppg[cindex]))
            data_length[i][0] = len(self.ppg[cindex])
        frame_r = int(math.ceil(max_time / self.hparams.frame_per_step))
        max_time = int(frame_r) * self.hparams.frame_per_step
        # to let max_time be a times of frame_per_step !!!!!!!!

        myppg = np.zeros((self.hparams.batch_size, max_time,
                          self.ppg[0].shape[-1] * self.hparams.Tacotron_context_size))
        myfwh = np.zeros((self.hparams.batch_size, max_time, self.fwh[0].shape[-1]))
        data_mask = np.zeros((self.hparams.batch_size, frame_r))
        # data_mask is for loss computation
        for i in range(self.hparams.batch_size):
            cindex = index * self.hparams.batch_size + i
            pad_ppg_matrix = np.tile(pad_ppg, (max_time - len(self.ppg[cindex]), 1))
            pad_fwh_matrix = np.tile(pad_fwh, (max_time - len(self.fwh[cindex]), 1))
            myppg[i] = np.concatenate((modify_output_feature(self.ppg[cindex], 'window', True,
                                                             self.hparams.Tacotron_context_size),
                                       pad_ppg_matrix), axis=0)
            myfwh[i] = np.concatenate((self.fwh[cindex], pad_fwh_matrix), axis=0)
            ones_length = int(math.ceil(len(self.ppg[cindex]) / self.hparams.frame_per_step))
            data_mask[i] = np.concatenate((np.ones(ones_length),
                                           np.zeros(frame_r - ones_length)))
            # 如果求的是每一时间步的loss，那么就应该把ones 改成 1/5。
        myfwh_r = np.reshape(myfwh, (self.hparams.batch_size, -1,
                                     self.fwh[0].shape[-1] * self.hparams.frame_per_step))
        myppg_r = np.reshape(myppg, (self.hparams.batch_size, -1,
                                     self.hparams.Tacotron_context_size * self.ppg[0].shape[-1] * self.hparams.frame_per_step))
        zeros_fwh = np.zeros((self.hparams.batch_size, 1, self.fwh[0].shape[-1] * self.hparams.frame_per_step)) + 1e-10
        input_myfwh_r = np.concatenate((zeros_fwh, myfwh_r), axis=1)

        initial_states = np.zeros((self.hparams.batch_size, self.hparams.Tacotron_decoder_output_size,
                                   self.hparams.Tacotron_decoder_output_size))

        # mask !!!!!!!!!!!!!!!!!!!!!!!!!!!
        return [input_myfwh_r[:, :-1, :], myppg_r], myfwh_r, data_mask

    def on_epoch_end(self):
        index = [i for i in range(len(self.ppg))]
        random.shuffle(index)
        self.ppg = self.ppg[index]
        self.fwh = self.fwh[index]


def modify_output_feature(fwh_data_now, mode, is_neighbor, window_size):
    if mode == 'MLPG':
        fwh_delta = np.zeros(fwh_data_now.shape)
        fwh_delta[1:-1] = (np.array(fwh_data_now[2:]) - np.array(fwh_data_now[:-2])) * 0.5
        fwh_delta[0] = (np.array(fwh_data_now[1]) - np.array(fwh_data_now[0])) * 0.5
        fwh_delta[-1] = (np.array(fwh_data_now[-1]) - np.array(fwh_data_now[-2])) * 0.5
        fwh_delta_2 = np.zeros(fwh_data_now.shape)
        fwh_delta_2[1:-1] = (np.array(fwh_data_now)[2:] - 2 * np.array(fwh_data_now[1:-1]) +
                             np.array(fwh_data_now[:-2]))
        fwh_delta_2[0] = np.array(fwh_data_now[1]) - np.array(fwh_data_now[0])
        fwh_delta_2[-1] = np.array(fwh_data_now[-2] - np.array(fwh_data_now[-1]))
        result = np.concatenate((fwh_data_now, fwh_delta, fwh_delta_2), axis=1)
    elif mode == 'window':
        #  is_neighbor True  represents [-3,-2,-1,0,1,2,3]
        #  is_neighbor False represents [-4,-2,-1,0,1,2,4]
        if bool(is_neighbor):
            window_front_list = np.array([(i + 1) for i in range(int((int(window_size) - 1) / 2))])
        else:
            window_front_list = np.array([2 ** i for i in range(int((int(window_size) - 1) / 2))])
        window_back_list = (-window_front_list[::-1])
        window_list = np.concatenate((window_back_list, np.zeros(1), window_front_list), axis=0)
        window_list = [int(a) for a in window_list]
        # print(window_list)
        result = np.array([[] for i in range(fwh_data_now.shape[0])])
        for i in range(len(window_list)):
            fwh_concat = np.zeros(fwh_data_now.shape)
            fwh_concat[max(-window_list[i], 0): min(fwh_data_now.shape[0]-window_list[i], fwh_data_now.shape[0])] =\
                fwh_data_now[max(window_list[i], 0): min(fwh_data_now.shape[0] + window_list[i], fwh_data_now.shape[0])]
            result = np.concatenate((result, fwh_concat), axis=1)
        assert result.shape == (fwh_data_now.shape[0], fwh_data_now.shape[1] * len(window_list))
    else:
        result = fwh_data_now
    return result


def dataset_preprocess(hparams):
    """
    read the dataset and pad it to max_time length,
    shuffle the dataset to make splited train, test keep a balance of 4 emotion,
    save the result to folder.
    :return: None
    """
    abspath = os.path.abspath('.')
    ppg_folder_abspath = os.path.join(abspath, hparams.ppg)
    fwh_folder_abspath = os.path.join(abspath, hparams.fwh)
    ppg_list = os.listdir(ppg_folder_abspath)
    fwh_list = os.listdir(fwh_folder_abspath)
    ppg_file_list = {}  # file_name: file_data
    fwh_file_list = []  # [file_name, file_path], ...
    ppg_data = []
    fwh_data = []
    data_length = []  # time length of each data
    data_mask = []
    del_list = []  # del the unpaired fwh_file
    max_time = -1
    for fwh_file in fwh_list:
        fwh_file_path = os.path.join(fwh_folder_abspath, fwh_file)
        fwh_file_list.append([fwh_file.split('_')[2], fwh_file_path])

    for ppg_file in ppg_list:
        ppg_file_path = os.path.join(ppg_folder_abspath, ppg_file)
        ppg_file_list[ppg_file.split('_')[2]] = np.load(ppg_file_path)

    neutral_count = 0
    for i in range(len(fwh_file_list)):
        if fwh_file_list[i][0] in ppg_file_list:
            _fwh_data = np.load(fwh_file_list[i][1])
            _ppg_data = ppg_file_list[fwh_file_list[i][0]][:len(_fwh_data)]
            if len(_ppg_data) == 0:
                print("ppg_length zero :", fwh_file_list[i][0])
            if hparams.get_sad_only is True and _ppg_data[0][-1] != 1:
                continue
            if hparams.balance is True and _ppg_data[0][-4] == 1:
                neutral_count += 1
                if neutral_count > 2000:
                    continue
            # 上面这句话的使用前提是ppg数据严格长于表情参数序列 ！！
            max_time = max(max_time, len(_ppg_data))
            data_length.append(len(_ppg_data))
            ppg_data.append(_ppg_data)
            fwh_data.append(_fwh_data)
        else:
            del_list.append(i)

    file_index = np.array(fwh_file_list)[:, 0]
    for i in range(len(del_list) - 1, -1, -1):
        del file_index[del_list[i]]

    print("before concat", len(ppg_data))
    #  先拼接上其他特征，如一阶与二阶动态特征
    for i in range(len(data_length)):
        fwh_data[i] = modify_output_feature(fwh_data[i], mode=hparams.mode, is_neighbor=hparams.is_neighbor,
                                            window_size=hparams.window_size)

    print("after concat", len(ppg_data))
    print("before standardlization")
    #  之后standardlization
    ppg_concate = []
    fwh_concate = []
    for i in range(len(ppg_data)):
        ppg_concate.extend(ppg_data[i])
        fwh_concate.extend(fwh_data[i])
    ppg_scaler = preprocessing.StandardScaler()
    fwh_scaler = preprocessing.StandardScaler()
    ppg_scaler.fit(ppg_concate)
    fwh_scaler.fit(fwh_concate)
    for i in range(ppg_data[0].shape[1]):
        if ppg_scaler.var_[i] < 1e-10:
            ppg_scaler.var_[i] = 1.0
    for i in range(fwh_data[0].shape[1]):
        if fwh_scaler.var_[i] < 1e-10:
            fwh_scaler.var_[i] = 1.0
    ppg_concate = ppg_scaler.transform(ppg_concate)
    fwh_concate = fwh_scaler.transform(fwh_concate)
    start_idx = 0
    for i in range(len(data_length)):
        end_idx = start_idx + data_length[i]
        ppg_data[i] = ppg_concate[start_idx: end_idx]
        fwh_data[i] = fwh_concate[start_idx: end_idx]
        start_idx = end_idx
    assert start_idx == len(ppg_concate)

    print("after standardlization")

    del ppg_file_list
    del fwh_file_list
    del ppg_concate
    del fwh_concate
    gc.collect()

    ppg_data = np.array(ppg_data)
    fwh_data = np.array(fwh_data)
    data_length = np.array(data_length)
    data_mask = np.array(data_mask)

    # shuffle
    print('before shuffle')
    dataset_size = ppg_data.shape[0]  # size of dataset
    index = [i for i in range(dataset_size)]
    random.shuffle(index)
    ppg_data = ppg_data[index]
    fwh_data = fwh_data[index]
    data_length = data_length[index]
    file_index = file_index[index]
    print('after shuffle')

    if hparams.mode == 'MLPG':
        data_path = 'data/mlpg/'
    elif hparams.mode == 'window':
        data_path = 'data/w' + str(hparams.window_size) + 'ne' + str(int(hparams.is_neighbor)) + '/'
    else:
        data_path = 'data/raw/'

    data_path = data_path[: -1] + '_np/'

    if hparams.get_sad_only is True:
        data_path = data_path[: -1] + '_sad_only/'

    if hparams.balance is True:
        data_path = data_path[: -1] + '_balance/'

    if hparams.data_path is not None:
        data_path = hparams.data_path
    # split train / validation / test
    np.save(data_path + 'data_mask_train', [])
    np.save(data_path + 'data_mask_validation', [])
    np.save(data_path + 'data_mask_evaluation', [])
    np.save(data_path + 'ppg_train', ppg_data[:int(0.8 * len(data_length))])
    np.save(data_path + 'ppg_validation', ppg_data[int(0.8 * len(data_length)):int(0.9 * len(data_length))])
    np.save(data_path + 'ppg_evaluation', ppg_data[int(0.9 * len(data_length)):])
    np.save(data_path + 'fwh_train', fwh_data[:int(0.8 * len(data_length))])
    np.save(data_path + 'fwh_validation', fwh_data[int(0.8 * len(data_length)):int(0.9 * len(data_length))])
    np.save(data_path + 'fwh_evaluation', fwh_data[int(0.9 * len(data_length)):])
    np.save(data_path + 'variance', fwh_scaler.var_)
    np.save(data_path + 'mean', fwh_scaler.mean_)
    shape_info = [max_time, ppg_data[0].shape[1], fwh_data[0].shape[1]]
    with open(data_path + 'scaler.pickle', 'wb') as handle:
        pickle.dump((ppg_scaler, fwh_scaler, shape_info), handle)

    f = open(data_path + 'file_index.txt', 'w')
    for i in range(len(file_index)):
        f.write(str(file_index[i]) + '\n')


def load_data(hparams):
    return np.load(os.path.join(hparams.dataset, 'ppg_train.npy')),\
           np.load(os.path.join(hparams.dataset, 'ppg_validation.npy')),\
           np.load(os.path.join(hparams.dataset, 'ppg_evaluation.npy')),\
           np.load(os.path.join(hparams.dataset, 'fwh_train.npy')),\
           np.load(os.path.join(hparams.dataset, 'fwh_validation.npy')),\
           np.load(os.path.join(hparams.dataset, 'fwh_evaluation.npy')),\
           np.load(os.path.join(hparams.dataset, 'data_mask_train.npy')),\
           np.load(os.path.join(hparams.dataset, 'data_mask_validation.npy')),\
           np.load(os.path.join(hparams.dataset, 'data_mask_evaluation.npy'))


def load_recent_model(path):
    """
    返回最近的权重文件
    :param path:
    :return:
    """
    abspath = os.path.abspath('.')
    model_path = os.path.join(abspath, path)
    model_list = os.listdir(model_path)
    recent_epoch = -1
    recent_file = None
    for model_file in model_list:
        epoch = int(((model_file.split('_')[-1]).split('-')[0])[1:])
        if epoch > recent_epoch:
            recent_epoch = epoch
            recent_file = model_file
    print('recent file', recent_file)
    if recent_file is None:
        return None
    else:
        return os.path.join(model_path, recent_file), recent_epoch


def load_best_model(path, hparams):
    """
    返回loss最小的权重文件
    :param path:
    :return:
    """
    abspath = os.path.abspath('.')
    model_path = os.path.join(abspath, path)
    model_list = os.listdir(model_path)
    best_loss = 1e10
    best_file = None
    for model_file in model_list:
        model_loss = float((model_file.split('-')[-1])[:-5])
        if hparams.load_epoch is not None:
            epoch = int(((model_file.split('_')[-1]).split('-')[0])[1:])
            if epoch != hparams.load_epoch:
                continue
        if model_loss < best_loss:
            best_loss = model_loss
            best_file = model_file
    print('best_file', best_file)
    return os.path.join(model_path, best_file)


def create_model(hparams, ppg_dim, fwh_dim, **kwargs):
    if load_recent_model(hparams.model_save_path) is not None:
        model_path, recent_epoch = load_recent_model(hparams.model_save_path)
        with CustomObjectScope({'ZoneoutLSTMCell': ZoneoutLSTMCell}):
            mymodel = load_model(model_path)
        return mymodel, recent_epoch
    else:
        if hparams.network == 'BLSTM':
            return BRNN(-1, ppg_dim, fwh_dim, hparams).get_model(), 0
        elif hparams.network == 'WaveNet':
            return WaveNet(hparams.wavenet_input_time, fwh_dim,
                           ppg_dim * hparams.wavenet_context_size, hparams).get_model(), 0
        elif hparams.network == 'Tacotron':
            Tacotron_model = TacotronDecoder(ppg_dim, fwh_dim, hparams, stateful=kwargs['stateful'],
                                             state_batch_size=kwargs['state_batch_size']).decode()
            if hparams.BLSTM_pretrain is True and hparams.Tacotron_encoder == 'BLSTM':
                BLSTM_model_path = load_best_model(hparams.BLSTM_pretrain_path, hparams)
                with CustomObjectScope({'ZoneoutLSTMCell': ZoneoutLSTMCell}):
                    BLSTM_model = load_model(BLSTM_model_path)
                print(BLSTM_model.layers)
                for i in range(3):
                    Tacotron_model.get_layer('Encoder_BLSTM_' + str(i + 1)).set_weights(
                        BLSTM_model.layers[len(BLSTM_model.layers) - 4 + i].get_weights())
                    if hparams.BLSTM_finetune is False:
                        Tacotron_model.get_layer('Encoder_BLSTM_' + str(i + 1)).trainable = False
            print(Tacotron_model.non_trainable_weights)
            return Tacotron_model, 0
        elif hparams.network == 'CNN':
            return CNN(ppg_dim, fwh_dim, hparams).get_model(), 0
        else:
            return None, 0


def get_power(wave, frame_rate, frame_length, frame_move, window_func=None):
    frame = frame_rate * frame_length
    overlap = frame_rate * frame_move
    step = frame - overlap
    frame_total = int(math.ceil(len(wave) / step))
    energy = np.zeros(frame_total)
    for i in range(frame_total):
        window = wave[np.arange(i * step, min(i * step + frame, len(wave)))]
        if window_func == "hanning":
            energy[i] = sum(np.multiply(window ** 2, scipy.signal.hanning(len(window))))
        else:
            energy[i] = sum(window ** 2)
    return energy


def get_predict_file_list(path):
    abspath = os.path.abspath('.')
    predict_path = os.path.join(abspath, path)
    if os.path.isfile(predict_path):
        return [predict_path]
    else:
        result = []
        predict_folder = os.listdir(predict_path)
        predict_folder.sort()
        for predict_file in predict_folder:
            result.append(os.path.join(predict_path, predict_file))
        return result


def pad_and_mask(ppg_data, max_time, ppg_dim):
    """
    pad and mask single ppg
    :return:
    """
    pad = np.zeros((1, ppg_dim))
    print(max_time, ppg_data.shape)
    pad_matrix = np.tile(pad, (max_time - ppg_data.shape[0], 1))
    ppg_predict = np.concatenate((ppg_data, pad_matrix), axis=0)
    return ppg_predict


def get_Tacotron_ppg(ppg_data, ppg_dim, hparams):
    pad = np.zeros((1, ppg_dim * hparams.Tacotron_context_size))
    frame_r = int(math.ceil(ppg_data.shape[0] / hparams.frame_per_step))
    max_time = int(frame_r) * hparams.frame_per_step
    pad_matrix = np.tile(pad, (max_time - ppg_data.shape[0], 1))
    ppg_matrix = np.concatenate((ppg_data, pad_matrix), axis=0)
    return ppg_matrix.reshape((frame_r, ppg_dim * hparams.frame_per_step * hparams.Tacotron_context_size))


def get_CNN_ppg(ppg_data, ppg_dim, hparams):
    pad = np.zeros((1, ppg_dim))
    frame_r = int(math.ceil(ppg_data.shape[0] / hparams.CNN_window_size))
    max_time = int(frame_r) * hparams.CNN_window_size
    pad_matrix = np.tile(pad, (max_time - ppg_data.shape[0], 1))
    ppg_matrix = np.concatenate((ppg_data, pad_matrix), axis=0)
    return np.expand_dims(ppg_matrix, axis=1)
    #  ppg_matrix [max_time(mod 35), 1, ppg_dim]


def get_post_mode(hparams):
    data_name = hparams.dataset.split('/')[-2]
    if data_name[:4] == 'mlpg':
        return 'MLPG', True, 5
    elif data_name[:3] == 'raw':
        return 'raw', True, 5
    else:
        return 'window', int(data_name[-4]), int(data_name[1: -6])


def get_Tacotron_fwh(ppg_file_name, fwh_scaler, hparams):
    ppg_file_name = ppg_file_name.split('/')[-1]
    fwh_name = ppg_file_name[: -6] + 'fwh100.npy'
    abspath = os.path.abspath('.')
    fwh_path = os.path.join(abspath, hparams.GTAPath + fwh_name)
    fwh_data = np.load(fwh_path)
    mode, ne, wsize = get_post_mode(hparams)
    fwh_data = modify_output_feature(fwh_data, mode, ne, wsize)
    fwh_data = fwh_scaler.transform(fwh_data)

    pad = np.zeros((1, fwh_data.shape[1]))
    frame_r = int(math.ceil(fwh_data.shape[0] / hparams.frame_per_step))
    max_time = int(frame_r) * hparams.frame_per_step
    pad_matrix = np.tile(pad, (max_time - fwh_data.shape[0], 1))
    fwh_matrix = np.concatenate((fwh_data, pad_matrix), axis=0)
    fwh_matrix = np.reshape(fwh_matrix, (frame_r, fwh_data.shape[1] * hparams.frame_per_step))
    zero_fwh = np.zeros((1, fwh_data.shape[1] * hparams.frame_per_step))
    fwh_input = np.concatenate((zero_fwh, fwh_matrix[:-1]), axis=0)
    return fwh_input, fwh_matrix


def _str_to_bool(s):
    """Convert string to bool (in argparse context)."""
    if s.lower() not in ['true', 'false']:
        raise ValueError('Argument needs to be a '
                         'boolean, got {}'.format(s))
    return {'true': True, 'false': False}[s.lower()]


def add_weight(fwh_predict, weight_path):
    abspath = os.path.abspath('.')
    weight_abspath = os.path.join(abspath, weight_path)
    weight_matrix = []
    f = open(weight_abspath, 'r')
    line = f.readline()
    while line != '':
        weight_matrix.append(float(line.strip()))
        line = f.readline()
    for i in range(len(weight_matrix)):
        for j in range(fwh_predict.shape[1] // 32):
            fwh_predict[:, i + 32 * j] *= weight_matrix[i]
    return fwh_predict


def test():
    input = np.array([[i] for i in range(10)])
    input = np.tile(input, (1, 3))
    print(input, '\n---------------')
    output = modify_output_feature(input, mode='window', is_neighbor=True, window_size=5)
    print(output)


if __name__ == "__main__":
    # mode = 'window' or 'MLPG'
    parser = argparse.ArgumentParser()
    parser.add_argument('--ppg', type=str)
    parser.add_argument('--fwh', type=str)
    parser.add_argument('--mode', type=str, default='raw')  # 'MLPG', 'window'
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--is_neighbor', type=_str_to_bool, default=True)
    parser.add_argument('--get_sad_only', type=_str_to_bool, default=False)
    parser.add_argument('--balance', type=_str_to_bool, default=False)
    parser.add_argument('--data_path', type=str, default=None)
    hparams = parser.parse_args()

    dataset_preprocess(hparams)

