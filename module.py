import keras
import keras.backend as K
from keras.layers import *


class ZoneoutLSTMCell(LSTMCell):
    def __init__(self, units,
                 zoneout_factor_cell=0.,
                 zoneout_factor_output=0.,
                 **kwargs):
        super(ZoneoutLSTMCell, self).__init__(units, **kwargs)
        zm = min(zoneout_factor_output, zoneout_factor_cell)
        zs = max(zoneout_factor_output, zoneout_factor_cell)

        if zm < 0. or zs > 1.:
            raise ValueError('One / both provided zoneoutfacotrs are not in [0, 1]')

        self._zoneout_cell = zoneout_factor_cell
        self._zoneout_outputs = zoneout_factor_output

    def get_config(self):
        config = super(ZoneoutLSTMCell, self).get_config()
        config['zoneout_factor_cell'] = self._zoneout_cell
        config['zoneout_factor_output'] = self._zoneout_outputs
        return config

    def call(self, inputs, state, **kwargs):
        output, new_state = super(ZoneoutLSTMCell, self).call(inputs, state)
        (prev_c, prev_h) = state
        (new_c, new_h) = new_state

        drop_c = K.dropout(new_c - prev_c, self._zoneout_cell)
        drop_h = K.dropout(new_h - prev_h, self._zoneout_outputs)
        c = (1. - self._zoneout_cell) * drop_c + prev_c
        h = (1. - self._zoneout_outputs) * drop_h + prev_h

        return output, [c, h]
