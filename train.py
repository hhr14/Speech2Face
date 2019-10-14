import tensorflow as tf
import Utils
from hparams import get_hparams
from model import BRNN, WaveNet, TacotronDecoder
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import optimizers
from keras.models import load_model, save_model
import time
import numpy as np
import json
import os
import pickle
from keras.utils import multi_gpu_model
from keras import backend as K
from module import ZoneoutLSTMCell
from keras.utils.generic_utils import CustomObjectScope
import random


def train_Tacotron_epoch(generator, my_multi_gpu_model, hparams, training):
    epoch_loss = 0
    for b in range(generator.__len__()):
        data_input, data_gt, data_mask = generator.__getitem__(b)
        # generator parallel problem !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        fwh_r_last = np.zeros((hparams.batch_size, 1, data_gt.shape[-1])) + 1e-10
        fwh_input = np.zeros((hparams.batch_size, data_gt.shape[1], data_gt.shape[-1]))
        for s in range(data_gt.shape[1]):
            if random.random() < hparams.TFrate:
                fwh_r = (data_input[0][:, s, :]).reshape((hparams.batch_size, 1, -1))
            else:
                fwh_r = fwh_r_last
            fwh_input[:, s, :] = np.squeeze(fwh_r)
            fwh_r_last = my_multi_gpu_model. \
                predict_on_batch([fwh_r, (data_input[1][:, s, :]).reshape((hparams.batch_size, 1, -1))])
        # get all the input first !

        if training is True:
            batch_loss = my_multi_gpu_model. \
                train_on_batch([fwh_input, data_input[1]], data_gt, sample_weight=data_mask)
        else:
            batch_loss = my_multi_gpu_model. \
                test_on_batch([fwh_input, data_input[1]], data_gt, sample_weight=data_mask)

        if training is True:
            print('step:', b, 'train loss:', batch_loss)
        epoch_loss += batch_loss
        my_multi_gpu_model.reset_states()
    if training is True:
        print('training epoch loss:', epoch_loss / generator.__len__())
    else:
        print('validating epoch loss:', epoch_loss / generator.__len__())
    generator.on_epoch_end()
    return epoch_loss / generator.__len__()


def multi_gpu_train(hparams, ppg_train, fwh_train, ppg_validate, fwh_validate, mymodel,
                    my_multi_gpu_model, recent_epoch):
    log_name = hparams.model_save_path.split('/')[-2]
    logdir = 'logs/' + log_name
    tensorboard_callback = TensorBoard(log_dir=logdir)
    steps = hparams.epochs / hparams.check_point_distance
    model_name = hparams.model_save_path + 'b' + str(hparams.batch_size) +\
                 '_lr' + str(hparams.learning_rate) + '_' + hparams.loss + '_'
    print(mymodel.summary())
    if hparams.network == 'BLSTM':
        train_generator = Utils.dataGenerator(ppg_train, fwh_train, hparams)
        validate_generator = Utils.dataGenerator(ppg_validate, fwh_validate, hparams)
        validation_steps = validate_generator.__len__()
    elif hparams.network == 'WaveNet':
        train_generator = Utils.wavenetDataGenerator(hparams, ppg_train, fwh_train)
        validate_generator = Utils.wavenetDataGenerator(hparams, ppg_validate, fwh_validate)
        validation_steps = hparams.wavenet_validation_step
    elif hparams.network == 'Tacotron':
        train_generator = Utils.TacotronGenerator(ppg_train, fwh_train, hparams)
        validate_generator = Utils.TacotronGenerator(ppg_validate, fwh_validate, hparams)
        validation_steps = validate_generator.__len__()
    elif hparams.network == 'CNN':
        train_generator = Utils.CNNDataGenerator(hparams, ppg_train, fwh_train, hparams.CNN_train_step)
        validate_generator = Utils.CNNDataGenerator(hparams, ppg_validate, fwh_validate, hparams.CNN_validate_step)
        validation_steps = validate_generator.__len__()
    else:
        train_generator = None
        validate_generator = None
        validation_steps = 0
    for i in range(int(steps)):
        print("\nstep is:", i)
        if hparams.TF is True:
            history = my_multi_gpu_model.fit_generator(generator=train_generator,
                                                       validation_data=validate_generator,
                                                       epochs=hparams.check_point_distance,
                                                       callbacks=[tensorboard_callback],
                                                       verbose=1,
                                                       validation_steps=validation_steps,
                                                       use_multiprocessing=False)
            val_loss = history.history['val_loss']
        elif hparams.TF is False and hparams.network == 'Tacotron':
            val_loss = []
            for e in range(hparams.check_point_distance):
                print('\nepoch is:', e)
                _ = train_Tacotron_epoch(train_generator, my_multi_gpu_model, hparams, training=True)
                val_epoch_loss = train_Tacotron_epoch(validate_generator, my_multi_gpu_model, hparams, training=False)
                val_loss.append(val_epoch_loss)
        else:
            raise ValueError('Unlegal mode !')

        epoch = (i + 1) * hparams.check_point_distance + recent_epoch
        print('val_loss', val_loss)
        mymodel_name = model_name + 'e' + str(epoch) + '-' + format(val_loss[-1], '.4f') + '.hdf5'
        mymodel.save(mymodel_name)


def multi_gpu_evaluate(hparams, ppg_evaluate, fwh_evaluate):
    best_file_path = Utils.load_best_model(hparams.model_save_path, hparams)
    my_model = load_model(best_file_path)
    my_multi_gpu_model = get_multi_gpu_model(my_model, hparams.gpu)
    if hparams.network == 'BLSTM':
        evaluate_generator_ = Utils.dataGenerator(ppg_evaluate, fwh_evaluate, hparams)
    elif hparams.network == 'WaveNet':
        evaluate_generator_ = Utils.wavenetDataGenerator(ppg_evaluate, fwh_evaluate, hparams)
    elif hparams.network == 'Tacotron':
        evaluate_generator_ = Utils.TacotronGenerator(ppg_evaluate, fwh_evaluate, hparams)
    elif hparams.network == 'CNN':
        evaluate_generator_ = Utils.CNNDataGenerator(hparams, ppg_evaluate, fwh_evaluate, hparams.CNN_evaluate_step)
    else:
        evaluate_generator_ = None
    if hparams.TF is False:
        return train_Tacotron_epoch(evaluate_generator_, my_multi_gpu_model, hparams, training=False)
    evaluate_loss = my_multi_gpu_model.evaluate_generator(generator=evaluate_generator_)
    return evaluate_loss


def predict(hparams, predict_file_list, ppg_scaler, fwh_scaler, shape_info):
    best_file_path = Utils.load_best_model(hparams.model_save_path, hparams)
    with CustomObjectScope({'ZoneoutLSTMCell': ZoneoutLSTMCell}):
        mymodel = load_model(best_file_path)
    my_multi_gpu_model = get_multi_gpu_model(mymodel, hparams.gpu)
    print(predict_file_list, len(predict_file_list))
    for i in range(len(predict_file_list)):
        print('>>> predict : ', predict_file_list[i])
        ppg_input = np.load(predict_file_list[i])
        print('ppg_shape', ppg_input.shape)
        ppg_predict = ppg_scaler.transform(ppg_input)
        if hparams.batch_pad is False:
            ppg_predict = Utils.pad_and_mask(ppg_predict, shape_info[0], shape_info[1])
        if hparams.add_mean is True:
            data_length = np.ones((1, shape_info[2])) / ppg_input.shape[0]
            input_ = [ppg_predict[np.newaxis, :], data_length]
        else:
            input_ = ppg_predict[np.newaxis, :]
        fwh_predict = my_multi_gpu_model.predict_on_batch(input_)
        if hparams.add_mean is True:
            fwh_predict = fwh_predict[0]
        fwh_predict = fwh_predict.reshape((-1, shape_info[2]))
        fwh_predict = fwh_predict[:ppg_input.shape[0], :]
        if hparams.use_weight is True:
            print("use weight!")
            fwh_predict = Utils.add_weight(fwh_predict, hparams.weight_path)
        fwh_predict = fwh_scaler.inverse_transform(fwh_predict)
        ppg_file_name = (predict_file_list[i].split('/')[-1]).split('.')[0]
        np.save(os.path.join(hparams.output_folder, ppg_file_name + '_fwh32'), fwh_predict)


def Tacotron_predict(hparams, predict_file_list, ppg_scaler, fwh_scaler, shape_info):
    best_file_path = Utils.load_best_model(hparams.model_save_path, hparams)
    with CustomObjectScope({'ZoneoutLSTMCell': ZoneoutLSTMCell}):
        original_model = load_model(best_file_path)
    mymodel = TacotronDecoder(shape_info[1], shape_info[2], hparams, stateful=True).decode()
    for i in range(len(mymodel.layers)):
        print(i, mymodel.layers[i], original_model.layers[i])
        mymodel.layers[i].set_weights(original_model.layers[i].get_weights())
    if hparams.GTA is True:
        mymodel = original_model

    my_multi_gpu_model = get_multi_gpu_model(mymodel, hparams.gpu)
    print(mymodel.layers[-3].stateful)
    print(predict_file_list)
    for i in range(len(predict_file_list)):
        ppg_input = np.load(predict_file_list[i])
        ppg_scaled = ppg_scaler.transform(ppg_input)
        ppg_scaled = Utils.modify_output_feature(ppg_scaled, 'window', True, hparams.Tacotron_context_size)
        ppg_r = Utils.get_Tacotron_ppg(ppg_scaled, ppg_input.shape[1], hparams)
        print('\n---- predict :', predict_file_list[i], '----')
        print('ppg_r.shape', ppg_r.shape)
        fwh_r_output = []
        if hparams.GTA is False:
            fwh_r = np.zeros((1, shape_info[2] * hparams.frame_per_step)) + 1e-10
            for step in range(ppg_r.shape[0]):
                fwh_r = my_multi_gpu_model.predict_on_batch([fwh_r[np.newaxis, :], ppg_r[step].reshape((1, 1, -1))])
                fwh_r = fwh_r.reshape((1, shape_info[2] * hparams.frame_per_step))
                fwh_r_output.extend(fwh_r.reshape((-1, shape_info[2])))
            fwh_r_output = np.array(fwh_r_output)[:ppg_input.shape[0], :]
        else:
            fwh_r_input, fwh_r_gt = Utils.get_Tacotron_fwh(predict_file_list[i], fwh_scaler, hparams)
            print('fwh_gt.shape', fwh_r_input.shape, 'ppg_r.shape', ppg_r.shape)
            ppg_r = ppg_r[:fwh_r_input.shape[0], :]
            evaluate_loss = my_multi_gpu_model.evaluate([fwh_r_input[np.newaxis, :], ppg_r[np.newaxis, :]],
                                                        fwh_r_gt[np.newaxis, :], batch_size=1)
            print('evaluate loss :', evaluate_loss)
            fwh_r_output = my_multi_gpu_model.predict_on_batch([fwh_r_input[np.newaxis, :], ppg_r[np.newaxis, :]])
            fwh_r_output = fwh_r_output.reshape((-1, shape_info[2]))
            fwh_r_output = fwh_r_output[:ppg_input.shape[0], :]
        if hparams.use_weight is True:
            print('use weight!')
            fwh_r_output = Utils.add_weight(fwh_r_output, hparams.weight_path)
        fwh_predict = fwh_scaler.inverse_transform(fwh_r_output)
        ppg_file_name = (predict_file_list[i].split('/')[-1]).split('.')[0]
        np.save(os.path.join(hparams.output_folder, ppg_file_name + '_fwh32'), fwh_predict)


def wavenet_predict(hparams, predict_file_list, ppg_scaler, fwh_scaler, shape_info):
    best_file_path = Utils.load_best_model(hparams.model_save_path, hparams)
    mymodel = load_model(best_file_path)
    my_multi_gpu_model = get_multi_gpu_model(mymodel, hparams.gpu)
    print(predict_file_list)
    for i in range(len(predict_file_list)):
        condition = np.load(predict_file_list[i])
        condition_scaled = ppg_scaler.transform(condition)
        condition_context = Utils.modify_output_feature(condition_scaled, mode='window',
                                                        is_neighbor=hparams.wavenet_context_neighbor,
                                                        window_size=hparams.wavenet_context_size)
        #  condition_input shape: [condition_time, context_size * ppg_dim]
        fwh_input = np.zeros((hparams.wavenet_input_time, shape_info[2]))
        condition_input = np.zeros((hparams.wavenet_input_time, condition_context.shape[1]))
        fwh_predict = np.zeros((condition.shape[0], shape_info[2]))
        for current in range(len(condition)):
            condition_input = np.concatenate((condition_input[1:], condition_context[current].reshape((1, -1))), axis=0)
            #  condition_input should include information of predict frame.
            fwh_predict[current] = my_multi_gpu_model.predict_on_batch([fwh_input[np.newaxis, :],
                                                                        condition_input[np.newaxis, :]])
            #  fwh_predict shape is [1, fwh_dim]
            fwh_input = np.concatenate((fwh_input[1:], fwh_predict[current]), axis=0)
        fwh_predict = fwh_scaler.inverse_transform(fwh_predict)
        ppg_file_name = (predict_file_list[i].split('/')[-1]).split('.')[0]
        np.save(os.path.join(hparams.output_folder, ppg_file_name + '_fwh32'), fwh_predict)


def CNN_predict(hparams, predict_file_list, ppg_scaler, fwh_scaler, shape_info):
    best_file_path = Utils.load_best_model(hparams.model_save_path, hparams)
    mymodel = load_model(best_file_path)
    my_multi_gpu_model = get_multi_gpu_model(mymodel, hparams.gpu)
    print(predict_file_list)
    for i in range(len(predict_file_list)):
        ppg_data = np.load(predict_file_list[i])
        ppg_data = ppg_scaler.transform(ppg_data)
        print('ppg_data.shape', ppg_data.shape)
        ppg_input = Utils.get_CNN_ppg(ppg_data, ppg_data.shape[1], hparams)
        print('ppg_input.shape', ppg_input.shape)
        fwh_predict = []
        for s in range(ppg_input.shape[0] // hparams.CNN_window_size):
            ppg_input_cnn = ppg_input[hparams.CNN_window_size * s: hparams.CNN_window_size * (s + 1)]
            fwh_predict_cnn = my_multi_gpu_model.predict_on_batch(ppg_input_cnn[np.newaxis, :])
            fwh_predict.extend(np.squeeze(fwh_predict_cnn))

        fwh_predict = (np.array(fwh_predict))[:ppg_data.shape[0]]
        if hparams.use_weight is True:
            print('use weight!')
            fwh_predict = Utils.add_weight(fwh_predict, hparams.weight_path)
        fwh_predict = fwh_scaler.inverse_transform(fwh_predict)
        ppg_file_name = (predict_file_list[i].split('/')[-1]).split('.')[0]
        np.save(os.path.join(hparams.output_folder, ppg_file_name + '_fwh32'), fwh_predict)


def get_multi_gpu_model(model, gpus):
    gpu_numbers = len(gpus.split(','))
    if gpu_numbers <= 1:
        return model
    else:
        print("\nuse multi_gpu !\n")
        return multi_gpu_model(model, gpus=gpu_numbers)


def main():
    hparams = get_hparams()
    print(hparams)
    os.environ['CUDA_VISIBLE_DEVICES'] = hparams.gpu
    if hparams.network == "BLSTM":
        if hparams.mode == 'train':
            ppg_train, ppg_validate, ppg_evaluate, fwh_train, fwh_validate, fwh_evaluate, train_mask, validate_mask, evaluate_mask \
                = Utils.load_data(hparams)
            mymodel, recent_epoch = Utils.create_model(hparams, ppg_train[0].shape[1], fwh_train[0].shape[1])
            my_multi_gpu_model = get_multi_gpu_model(mymodel, hparams.gpu)
            opt = optimizers.Adam(lr=hparams.learning_rate)
            if hparams.add_mean is False:
                my_multi_gpu_model.compile(optimizer=opt, loss=hparams.loss, sample_weight_mode='temporal')
            else:
                my_multi_gpu_model.compile(optimizer=opt,
                                           loss={
                                               'output_fwh': hparams.loss,
                                               'output_mean': hparams.loss
                                           },
                                           loss_weights={
                                               'output_fwh': 1.0,
                                               'output_mean': hparams.mean_weight
                                           },
                                           sample_weight_mode='temporal')
            multi_gpu_train(hparams, ppg_train, fwh_train, ppg_validate, fwh_validate, mymodel, my_multi_gpu_model,
                            recent_epoch)
            evaluate_loss = multi_gpu_evaluate(hparams, ppg_evaluate, fwh_evaluate)
            print("evaluate_loss: {:.4f}".format(evaluate_loss))
        elif hparams.mode == 'predict':
            (ppg_scaler, fwh_scaler, shape_info) = pickle.load(open(os.path.join(hparams.dataset,
                                                                                 'scaler.pickle'), 'rb'))
            print("shape_info", shape_info)
            predict(hparams, Utils.get_predict_file_list(hparams.predict), ppg_scaler, fwh_scaler, shape_info)
    elif hparams.network == "WaveNet":
        if hparams.mode == "train":
            ppg_train, ppg_validate, ppg_evaluate, fwh_train, fwh_validate, fwh_evaluate, train_mask, validate_mask, evaluate_mask \
                = Utils.load_data(hparams)
            assert hparams.batch_pad is True
            mymodel, recent_epoch = Utils.create_model(hparams, ppg_train[0].shape[1], fwh_train[0].shape[1])
            my_multi_gpu_model = get_multi_gpu_model(mymodel, hparams.gpu)
            opt = optimizers.Adam(lr=hparams.learning_rate)
            my_multi_gpu_model.compile(optimizer=opt, loss=hparams.loss)
            multi_gpu_train(hparams, ppg_train, fwh_train, ppg_validate, fwh_validate, mymodel, my_multi_gpu_model,
                            recent_epoch)
            evaluate_loss = multi_gpu_evaluate(hparams, ppg_evaluate, fwh_evaluate)
            print("evaluate_loss: {:.4f}".format(evaluate_loss))
        elif hparams.mode == "predict":
            (ppg_scaler, fwh_scaler, shape_info) = pickle.load(open(os.path.join(hparams.dataset,
                                                                                 'scaler.pickle'), 'rb'))
            print("shape_info", shape_info)
            wavenet_predict(hparams, Utils.get_predict_file_list(hparams.predict), ppg_scaler, fwh_scaler, shape_info)
    elif hparams.network == "Tacotron":
        if hparams.mode == "train":
            ppg_train, ppg_validate, ppg_evaluate, fwh_train, fwh_validate, fwh_evaluate, train_mask, validate_mask, evaluate_mask \
                = Utils.load_data(hparams)
            assert hparams.batch_pad is True
            if hparams.TF is True:
                mymodel, recent_epoch = Utils.create_model(hparams, ppg_train[0].shape[1], fwh_train[0].shape[1])
            else:
                mymodel, recent_epoch = Utils.create_model(hparams, ppg_train[0].shape[1], fwh_train[0].shape[1],
                                                           stateful=True, state_batch_size=hparams.batch_size)
            print('\nstateful is', mymodel.layers[-3].stateful, '\n')
            my_multi_gpu_model = get_multi_gpu_model(mymodel, hparams.gpu)
            opt = optimizers.Adam(lr=hparams.learning_rate)
            my_multi_gpu_model.compile(optimizer=opt, loss=hparams.loss, sample_weight_mode='temporal')
            multi_gpu_train(hparams, ppg_train, fwh_train, ppg_validate, fwh_validate, mymodel, my_multi_gpu_model,
                            recent_epoch)
            evaluate_loss = multi_gpu_evaluate(hparams, ppg_evaluate, fwh_evaluate)
            print("evaluate_loss: {:.4f}".format(evaluate_loss))
        elif hparams.mode == 'predict':
            (ppg_scaler, fwh_scaler, shape_info) = pickle.load(open(os.path.join(hparams.dataset,
                                                                                 'scaler.pickle'), 'rb'))
            Tacotron_predict(hparams, Utils.get_predict_file_list(hparams.predict), ppg_scaler, fwh_scaler, shape_info)
    elif hparams.network == 'CNN':
        if hparams.mode == 'train':
            ppg_train, ppg_validate, ppg_evaluate, fwh_train, fwh_validate, fwh_evaluate, train_mask, validate_mask, evaluate_mask \
                = Utils.load_data(hparams)
            mymodel, recent_epoch = Utils.create_model(hparams, ppg_train[0].shape[1], fwh_train[0].shape[1])
            my_multi_gpu_model = get_multi_gpu_model(mymodel, hparams.gpu)
            opt = optimizers.Adam(lr=hparams.learning_rate)
            my_multi_gpu_model.compile(optimizer=opt, loss=hparams.loss)
            multi_gpu_train(hparams, ppg_train, fwh_train, ppg_validate, fwh_validate, mymodel, my_multi_gpu_model,
                            recent_epoch)
            evaluate_loss = multi_gpu_evaluate(hparams, ppg_evaluate, fwh_evaluate)
            print("evaluate_loss: {:.4f}".format(evaluate_loss))
        elif hparams.mode == 'predict':
            (ppg_scaler, fwh_scaler, shape_info) = pickle.load(open(os.path.join(hparams.dataset,
                                                                                 'scaler.pickle'), 'rb'))
            CNN_predict(hparams, Utils.get_predict_file_list(hparams.predict), ppg_scaler, fwh_scaler, shape_info)


if __name__ == "__main__":
    main()
