import tensorflow as tf
import argparse


def get_hparams():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]
    error = {'mse': 'mean_squared_error', 'abs': 'mean_absolute_error'}
    parser = argparse.ArgumentParser(description="Speech2Face hparams")
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--network', type=str, default='BLSTM')
    parser.add_argument('--model_save_path', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--zoneout', type=float, default=0.1)
    parser.add_argument('--load_epoch', type=int, default=None)

    parser.add_argument('--add_mean', type=_str_to_bool, default=False)
    parser.add_argument('--mean_weight', type=float, default=1.0)

    parser.add_argument('--predict', type=str, default='sample/sample_ppg/',
                        help='if mode is predict, this parameter represents the folder or file of ppg to be predicted')
    parser.add_argument('--output_folder', type=str, default='predict_result/fwh_npy')
    parser.add_argument('--batch_pad', type=_str_to_bool, default=True)

    parser.add_argument('--use_weight', type=_str_to_bool, default=True)
    parser.add_argument('--weight_path', type=str, default='data/weight.txt')

    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--loss', type=str, default=error['mse'])
    parser.add_argument('--validation_freq', type=int, default=5)
    parser.add_argument('--check_point_distance', type=int, default=5)
    parser.add_argument('--early_stop_patience', type=int, default=50)

    # CNN
    parser.add_argument('--CNN_mode', type=str, default='autoencoder')
    parser.add_argument('--CNN_seq_out', type=_str_to_bool, default=True)
    parser.add_argument('--CNN_reg', type=_str_to_bool, default=True)

    parser.add_argument('--CNN_predict_step', type=int, default=35)
    parser.add_argument('--CNN_continuous', type=_str_to_bool, default=False)
    parser.add_argument('--CNN_train_step', type=int, default=150)
    parser.add_argument('--CNN_validate_step', type=int, default=25)
    parser.add_argument('--CNN_evaluate_step', type=int, default=1000)
    parser.add_argument('--CNN_P1dense_units', type=list, default=[200])

    parser.add_argument('--CNN_window_size', type=int, default=35)
    parser.add_argument('--Dense_units', type=list, default=[2048, 1024])
    parser.add_argument('--finalunits', type=int, default=3360)
    # window: 35                 20                 10                5
    # dense : [4096, 4096, 3360] [4096, 4096, 1920] [4096, 2048, 960] [4096, 2048, 1024, 480]

    parser.add_argument('--CNN_post_dense_units', type=list, default=[4096, 4096])
    # window:  35            20           10            5           raw
    # dense :  [4096, 3360]  [4096, 1920] [2048, 960]  [1024, 480]  [4096, 2048, 1120]

    parser.add_argument('--CNN_encoder_channels', type=list, default=[256, 512])
    parser.add_argument('--CNN_encoder_kernel_size', type=list, default=[6, 6])
    parser.add_argument('--CNN_encoder_strides', type=list, default=[1, 1])
    parser.add_argument('--CNN_encoder_pool_size', type=list, default=[2, 2])
    parser.add_argument('--CNN_encoder_pool_strides', type=list, default=[2, 2])
    parser.add_argument('--CNN_dense_units', type=list, default=[1024, 1024, 512])
    parser.add_argument('--CNN_decoder_shape', type=tuple, default=(1, 1, 512))
    parser.add_argument('--CNN_decoder_channels', type=list, default=[256, 128, 96])
    parser.add_argument('--CNN_decoder_kernel_size', type=list, default=[10, 10, 8])
    parser.add_argument('--CNN_decoder_strides', type=list, default=[2, 2, 1])



    # BLSTM
    parser.add_argument('--blstm_use_zoneout', type=_str_to_bool, default=False)
    parser.add_argument('--blstm_hidden_size', type=int, default=256)
    parser.add_argument('--blstm_layers', type=int, default=3)

    # WaveNet
    parser.add_argument('--wavenet_input_time', type=int, default=128)
    #  = receptive field
    parser.add_argument('--wavenet_context_size', type=int, default=11)
    parser.add_argument('--wavenet_context_neighbor', type=_str_to_bool, default=True)
    parser.add_argument('--wavenet_blocks', type=int, default=4)
    parser.add_argument('--wavenet_layers', type=int, default=7)
    # set to x that limited to 2 ** x = input_time
    parser.add_argument('--wavenet_step_per_epoch', type=int, default=1000)
    parser.add_argument('--wavenet_validation_step', type=int, default=50)
    # max_step = wavenet_step_per_epoch * epochs.

    parser.add_argument('--arr', type=list, default=[1, 2])

    # Tacotron
    parser.add_argument('--TF', type=_str_to_bool, default=True)
    # teacher forcing during training
    parser.add_argument('--GTA', type=_str_to_bool, default=False)
    # ground truth alignment during synthesis
    parser.add_argument('--Tacotron_encoder', type=str, default='Dense')
    parser.add_argument('--Tacotron_postnet', type=_str_to_bool, default=False)
    parser.add_argument('--frame_per_step', type=int, default=1)
    parser.add_argument('--Tacotron_use_zoneout', type=_str_to_bool, default=True)
    parser.add_argument('--TFrate', type=float, default=0)
    parser.add_argument('--Masking', type=_str_to_bool, default=False)
    parser.add_argument('--Tacotron_context_size', type=int, default=1)

    parser.add_argument('--BLSTM_pretrain', type=_str_to_bool, default=False)
    parser.add_argument('--BLSTM_pretrain_path', type=str, default='weight/BLSTM_mlpg_np_3layer/')
    parser.add_argument('--BLSTM_finetune', type=_str_to_bool, default=True)

    parser.add_argument('--Tacotron_Conv_dropout', type=float, default=0.5)
    parser.add_argument('--PreNet_hidden_size', type=int, default=256)
    parser.add_argument('--PreNet_layers', type=int, default=2)
    parser.add_argument('--Tacotron_encoder_hidden_size', type=int, default=256)
    parser.add_argument('--Tacotron_encoder_layers', type=int, default=3)
    parser.add_argument('--Tacotron_encoder_kernel_size', type=int, default=5)
    parser.add_argument('--Tacotron_encoder_conv_strides', type=int, default=1)
    parser.add_argument('--Tacotron_decoder_layers', type=int, default=2)
    parser.add_argument('--Tacotron_decoder_output_size', type=int, default=256)
    parser.add_argument('--PostNet_hidden_size', type=int, default=512)
    parser.add_argument('--PostNet_kernel_size', type=int, default=5)
    parser.add_argument('--PostNet_layers', type=int, default=5)
    parser.add_argument('--GTAPath', type=str, default='sample/sample_fwh/')
    return parser.parse_args()

