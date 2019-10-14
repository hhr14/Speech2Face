import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import *
from module import ZoneoutLSTMCell
from keras.utils.generic_utils import CustomObjectScope
import math
import random


class CNN(object):
    def __init__(self, ppg_dim, fwh_dim, hparams):
        self.ppg_dim = ppg_dim
        self.fwh_dim = fwh_dim
        self.hparams = hparams
        self.Encoder = [Conv2D(filters=self.hparams.CNN_encoder_channels[i],
                               kernel_size=(self.hparams.CNN_encoder_kernel_size[i], 1),
                               strides=(self.hparams.CNN_encoder_strides[i], 1),
                               activation='tanh')
                        for i in range(len(self.hparams.CNN_encoder_channels))]
        self.Encoder_dropout = [Dropout(self.hparams.dropout)
                                for i in range(len(self.hparams.CNN_encoder_channels))]
        self.Pooling = [MaxPool2D(pool_size=(self.hparams.CNN_encoder_pool_size[i], 1),
                                  strides=(self.hparams.CNN_encoder_pool_strides[i], 1))
                        for i in range(len(self.hparams.CNN_encoder_pool_size))]
        self.Dense = [Dense(self.hparams.CNN_dense_units[i])
                      for i in range(len(self.hparams.CNN_dense_units))]
        self.Decoder = [Conv2DTranspose(filters=self.hparams.CNN_decoder_channels[i],
                                        kernel_size=(self.hparams.CNN_decoder_kernel_size[i], 1),
                                        strides=(self.hparams.CNN_decoder_strides[i], 1))
                        for i in range(len(self.hparams.CNN_decoder_channels))]
        self.Decoder_dropout = [Dropout(self.hparams.dropout)
                                for i in range(len(self.hparams.CNN_decoder_channels))]

        self.PostDense = [Dense(self.hparams.CNN_post_dense_units[i], activation='tanh')
                          for i in range(len(self.hparams.CNN_post_dense_units))]


        self.P1Dense = [Dense(self.hparams.CNN_P1dense_units[i])
                        for i in range(len(self.hparams.CNN_P1dense_units))]

        self.Fulldense = [Dense(self.hparams.Dense_units[i], activation='tanh')
                          for i in range(len(self.hparams.Dense_units))]

    def get_model(self):
        input_ppg = Input(shape=(self.hparams.CNN_window_size, 1, self.ppg_dim))

        if self.hparams.CNN_mode == 'dense':
            dense_out = Flatten()(input_ppg)
            for i in range(len(self.Fulldense)):
                dense_out = self.Fulldense[i](dense_out)
            dense_out = Dense(self.hparams.finalunits)(dense_out)
            dense_out = Reshape(target_shape=(self.hparams.CNN_window_size, 1, self.fwh_dim))(dense_out)
            return Model(inputs=input_ppg, outputs=dense_out)

        encoder_out = input_ppg
        for i in range(len(self.Encoder)):
            encoder_out = self.Encoder[i](encoder_out)
            encoder_out = self.Pooling[i](encoder_out)
        dense_out = Flatten()(encoder_out)
        if self.hparams.CNN_mode == 'autoencoder':
            for i in range(len(self.Dense)):
                dense_out = self.Dense[i](dense_out)
            decoder_out = Reshape(self.hparams.CNN_decoder_shape)(dense_out)
            for i in range(len(self.Decoder)):
                decoder_out = self.Decoder[i](decoder_out)
            return Model(inputs=input_ppg, outputs=decoder_out)
        elif self.hparams.CNN_mode == 'postdense':
            if self.hparams.CNN_seq_out is True:
                for i in range(len(self.PostDense)):
                    dense_out = self.PostDense[i](dense_out)
                dense_out = Dense(self.hparams.finalunits)(dense_out)
                dense_out = Reshape(target_shape=(self.hparams.CNN_window_size, 1, self.fwh_dim))(dense_out)
                return Model(inputs=input_ppg, outputs=dense_out)
            else:
                for i in range(len(self.P1Dense)):
                    dense_out = self.P1Dense[i](dense_out)
                return Model(inpus=input_ppg, outputs=dense_out)


class BRNN(object):
    """
    convert PPGs to Blend shape parameters
    PPG: [batch_size, max_time, feature_dim(222=218+4)]
    fwh: [batch_size, max_time, feature_dim(32=25+7)]
    """
    def __init__(self, max_time, ppg_dim, fwh_dim, hparams):
        self.max_time = max_time
        self.ppg_dim = ppg_dim
        self.fwh_dim = fwh_dim
        self.hparams = hparams

    def get_model(self):
        input_ppg = Input(shape=(None, self.ppg_dim))
        output = Masking(mask_value=0)(input_ppg)
        for i in range(self.hparams.blstm_layers):
            if self.hparams.blstm_use_zoneout is False:
                output = Bidirectional(LSTM(units=self.hparams.blstm_hidden_size, return_sequences=True,
                                            dropout=self.hparams.dropout, recurrent_dropout=self.hparams.dropout),
                                       merge_mode='concat')(output)
            else:
                with CustomObjectScope({'ZoneoutLSTMCell': ZoneoutLSTMCell}):
                    output = Bidirectional(RNN(ZoneoutLSTMCell(units=self.hparams.blstm_hidden_size,
                                                               zoneout_factor_cell=self.hparams.zoneout,
                                                               zoneout_factor_output=self.hparams.zoneout),
                                               return_sequences=True),
                                           merge_mode='concat')(output)
            # output = Activation('tanh')(output)
        output_fwh = Dense(self.fwh_dim, name='output_fwh')(output)
        if self.hparams.add_mean is True:
            data_length = Input(shape=(self.fwh_dim,))
            output_sum = Lambda(lambda x: K.sum(x, axis=-2))(output_fwh)
            output_mean_ = Multiply()([output_sum, data_length])
            output_mean = Reshape((1, self.fwh_dim), name='output_mean')(output_mean_)
            model_output = [output_fwh, output_mean]
            model_input = [input_ppg, data_length]
        else:
            model_output = output_fwh
            model_input = input_ppg
        return Model(inputs=model_input, outputs=model_output)


class TacotronDecoder(object):
    def __init__(self, ppg_dim, fwh_dim, hparams, stateful=False, state_batch_size=1):
        self.ppg_dim = ppg_dim
        self.fwh_dim = fwh_dim
        self.hparams = hparams
        # share layer
        self.PreNet_Dense_list = [Dense(self.hparams.PreNet_hidden_size, activation='relu',
                                        name='{}_dense_{}'.format('PreNet', i + 1))
                                  for i in range(self.hparams.PreNet_layers)]
        self.PreNet_dropout_list = [Dropout(self.hparams.Tacotron_Conv_dropout,
                                            name='{}_dropout_{}'.format('PreNet', i+1))
                                    for i in range(self.hparams.PreNet_layers)]
        self.Encoder_Dense_list = [Dense(self.hparams.Tacotron_encoder_hidden_size, activation='relu',
                                         name='{}_dense_{}'.format('Encoder', i + 1))
                                   for i in range(self.hparams.Tacotron_encoder_layers)]
        self.Encoder_Conv1D = [Conv1D(filters=self.hparams.Tacotron_encoder_hidden_size,
                                      kernel_size=self.hparams.Tacotron_encoder_kernel_size,
                                      strides=self.hparams.Tacotron_encoder_conv_strides,
                                      padding='same', activation='relu',
                                      name='{}_Conv1D_{}'.format('Encoder', i + 1))
                               for i in range(self.hparams.Tacotron_encoder_layers)]
        self.Encoder_BatchNorm = [BatchNormalization(name='{}_BatchNorm_{}'.format('Encoder', i + 1))
                                  for i in range(self.hparams.Tacotron_encoder_layers)]
        if hparams.Tacotron_use_zoneout is False:
            self.Encoder_BLSTM = [Bidirectional(LSTM(units=self.hparams.Tacotron_encoder_hidden_size,
                                                     return_sequences=True,
                                                     dropout=self.hparams.dropout,
                                                     recurrent_dropout=self.hparams.dropout),
                                                merge_mode='concat',
                                                name='{}_BLSTM_{}'.format('Encoder', i + 1))
                                  for i in range(self.hparams.Tacotron_encoder_layers)]
        else:
            with CustomObjectScope({'ZoneoutLSTMCell': ZoneoutLSTMCell}):
                self.Encoder_BLSTM = [Bidirectional(RNN(ZoneoutLSTMCell(units=self.hparams.Tacotron_encoder_hidden_size,
                                                                        zoneout_factor_cell=self.hparams.zoneout,
                                                                        zoneout_factor_output=self.hparams.zoneout),
                                                        return_sequences=True),
                                                    merge_mode='concat',
                                                    name='{}_BLSTM_{}'.format('Encoder', i + 1))
                                      for i in range(self.hparams.Tacotron_encoder_layers)]
        self.Encoder_dropout_list = \
            [Dropout(self.hparams.Tacotron_Conv_dropout, name='{}_dropout_{}'.format('Encoder', i + 1))
             for i in range(self.hparams.Tacotron_encoder_layers)]
        #  each Decoder layer don't share the same state, state spreads across the one layer
        self.states = [None for i in range(self.hparams.Tacotron_decoder_layers)]
        if hparams.Tacotron_use_zoneout is False:
            self.Decoder_LSTM = [LSTM(units=self.hparams.Tacotron_decoder_output_size,
                                      return_sequences=True,
                                      return_state=False,
                                      dropout=self.hparams.dropout,
                                      recurrent_dropout=self.hparams.dropout,
                                      name='{}_LSTM_{}'.format('Decoder', i + 1),
                                      stateful=stateful)
                                 for i in range(self.hparams.Tacotron_decoder_layers)]
        else:
            with CustomObjectScope({'ZoneoutLSTMCell': ZoneoutLSTMCell}):
                self.Decoder_LSTM = [RNN(ZoneoutLSTMCell(units=self.hparams.Tacotron_decoder_output_size,
                                                         zoneout_factor_output=self.hparams.zoneout,
                                                         zoneout_factor_cell=self.hparams.zoneout),
                                         return_sequences=True,
                                         return_state=False,
                                         name='{}_LSTM_{}'.format('Decoder', i + 1),
                                         stateful=stateful)
                                     for i in range(self.hparams.Tacotron_decoder_layers)]
        self.Linear_projection = Dense(self.fwh_dim * self.hparams.frame_per_step)
        self.PostNet_Conv1D = [Conv1D(filters=self.hparams.PostNet_hidden_size,
                               kernel_size=self.hparams.PostNet_kernel_size,
                               strides=1, padding='same', activation='tanh',
                                      name='{}_Conv1D_{}'.format('PostNet', i + 1))
                               for i in range(self.hparams.PostNet_layers - 1)] + [
            Conv1D(filters=self.fwh_dim * self.hparams.frame_per_step,
                   kernel_size=self.hparams.PostNet_kernel_size,
                   strides=1, padding='same', activation='tanh',
                   name='{}_Conv1D_{}'.format('PostNet', self.hparams.PostNet_layers))]
        self.PostNet_dropout_list = [Dropout(self.hparams.Tacotron_Conv_dropout,
                                             name='{}_dropout_{}'.format('PostNet', i + 1))
                                     for i in range(self.hparams.PostNet_layers)]
        self.PostNet_BatchNorm = [BatchNormalization(name='{}_BatchNorm_{}'.format('PostNet', i + 1))
                                  for i in range(self.hparams.PostNet_layers)]
        if stateful is True:
            self.input_decoder_shape = None
            self.input_ppg_shape = None
            self.input_batch_decoder_shape = (state_batch_size, None, self.fwh_dim * self.hparams.frame_per_step)
            self.input_batch_ppg_shape = (state_batch_size, None,
                                          self.ppg_dim * self.hparams.frame_per_step * hparams.Tacotron_context_size)
        else:
            self.input_decoder_shape = (None, self.fwh_dim * self.hparams.frame_per_step)
            self.input_ppg_shape = (None, self.ppg_dim * self.hparams.frame_per_step * hparams.Tacotron_context_size)
            self.input_batch_decoder_shape = None
            self.input_batch_ppg_shape = None

    def PreNet(self, input):
        # input : [batch_size, self.fwh_dim * self.hparams.frame_per_step]
        output = input
        for i in range(self.hparams.PreNet_layers):
            output = self.PreNet_Dense_list[i](output)
            output = self.PreNet_dropout_list[i](output)
        return output

    def Encoder(self, input, mask=None):
        output = input
        for i in range(self.hparams.Tacotron_encoder_layers):
            if self.hparams.Tacotron_encoder == 'Dense':
                output = self.Encoder_Dense_list[i](output)
                output = self.Encoder_dropout_list[i](output)
            elif self.hparams.Tacotron_encoder == 'Conv':
                output = self.Encoder_Conv1D[i](output)
                output = self.Encoder_BatchNorm[i](output)
                output = self.Encoder_dropout_list[i](output)
            elif self.hparams.Tacotron_encoder == 'BLSTM':
                output = self.Encoder_BLSTM[i](output, mask=mask)
            elif self.hparams.Tacotron_encoder == 'FlattenDense':
                output = Flatten()(output)
            else:
                raise ValueError('encoder should be specified!')
        return output

    def decode(self):
        """
        input is one batch_size, so after one batch_size, so after one batch, each state should equal to zeros
        :return:
        """
        decoder_input = Input(shape=self.input_decoder_shape, batch_shape=self.input_batch_decoder_shape)
        ppg_input = Input(shape=self.input_ppg_shape, batch_shape=self.input_batch_ppg_shape)

        if self.hparams.Masking is True:
            mask_decoder_input = Masking(mask_value=0)(decoder_input)
            mask_ppg_input = Masking(mask_value=0)(ppg_input)
            prenet_output = self.PreNet(mask_decoder_input)
            encoder_input = self.Encoder(mask_ppg_input)
            decoder_mask = None
        else:
            decoder_mask = Masking(mask_value=0).compute_mask(ppg_input)
            prenet_output = self.PreNet(decoder_input)
            encoder_input = self.Encoder(ppg_input, decoder_mask)

        rnn_output = Concatenate(axis=-1)([prenet_output, encoder_input])
        # mask = Input(shape=(self.hparams.PreNet_hidden_size + self.hparams.Tacotron_encoder_hidden_size))
        # diff_mask = Input(shape=(self.hparams.PreNet_hidden_size + self.hparams.Tacotron_encoder_hidden_size))
        for i in range(self.hparams.Tacotron_decoder_layers):
            rnn_output = self.Decoder_LSTM[i](rnn_output, mask=decoder_mask)

            # feed by self.states is unhelpful in training, since we don't stop rnn during epochs
            # but it is important in generating since each fit states will be set to zeros.!!!!!!
            rnn_output = Concatenate(axis=-1)([rnn_output, encoder_input])
        decoder_output = self.Linear_projection(rnn_output)
        if self.hparams.Tacotron_postnet is True:
            residual_output = decoder_output
            for i in range(self.hparams.PostNet_layers):
                residual_output = self.PostNet_Conv1D[i](residual_output)
                residual_output = self.PostNet_BatchNorm[i](residual_output)
                residual_output = self.PostNet_dropout_list[i](residual_output)
            decoder_output = Add()([decoder_output, residual_output])
        return Model(inputs=[decoder_input, ppg_input], outputs=decoder_output)

    def initialize_decoder_states(self):
        self.states = [K.zeros((self.hparams.Tacotron_decoder_output_size,)),
                       K.zeros((self.hparams.Tacotron_decoder_output_size,))]


class WaveNet(object):
    def __init__(self, max_time, fwh_dim, ppg_dim, hparams):
        self.max_time = max_time
        self.ppg_dim = ppg_dim
        self.fwh_dim = fwh_dim
        self.hparams = hparams

    def gated_activation(self, inputs):
        def lambda_split(x):
            l_input, r_input = tf.split(x, 2, axis=2)
            return l_input, r_input

        # l_input, r_input = Lambda(lambda_split)(inputs)
        split_point = self.fwh_dim
        l_input = Lambda(lambda x: x[:, :, :split_point])(inputs)
        r_input = Lambda(lambda x: x[:, :, split_point:])(inputs)
        tanh_output = Activation('tanh')(l_input)
        sigmoid_output = Activation('sigmoid')(r_input)
        gated_output = Multiply()([tanh_output, sigmoid_output])
        return gated_output

    def get_model(self):
        input_fwh = Input(shape=(self.max_time, self.fwh_dim))
        condition_input_ppg = Input(shape=(self.max_time, self.ppg_dim))
        dilation_list = [2 ** i for i in range(self.hparams.wavenet_layers)]
        causal_out = Conv1D(filters=self.fwh_dim, kernel_size=2, padding='causal',
                            dilation_rate=1)(input_fwh)
        skip_out_list = []
        for i in range(self.hparams.wavenet_blocks):
            for j in range(len(dilation_list)):
                causal_out = Conv1D(filters=self.fwh_dim * 2, kernel_size=2, padding='causal',
                                    dilation_rate=dilation_list[j])(causal_out)
                condition_conv_out = Conv1D(filters=self.fwh_dim * 2, kernel_size=1,
                                            padding='same')(condition_input_ppg)
                causal_out = Add()([causal_out, condition_conv_out])
                causal_out = self.gated_activation(causal_out)
                residual_out = Conv1D(filters=self.fwh_dim, kernel_size=1, padding='same')(causal_out)
                skip_out = Conv1D(filters=self.fwh_dim, kernel_size=1, padding='same')(causal_out)
                causal_out = Add()([causal_out, residual_out])
                skip_out_list.append(skip_out)
        post_out = Add()(skip_out_list)
        post_out = ReLU()(post_out)
        post_out = Conv1D(filters=self.fwh_dim, kernel_size=1, padding='same')(post_out)
        post_out = ReLU()(post_out)
        post_out = Conv1D(filters=self.fwh_dim, kernel_size=1, padding='same')(post_out)
        post_out = Lambda(lambda x: x[:, -1, :])(post_out)
        #  [batch_size, input_time, fwh_dim]
        return Model(inputs=[input_fwh, condition_input_ppg], outputs=post_out)

