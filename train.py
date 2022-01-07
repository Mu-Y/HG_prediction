import pickle
import os,datetime
import sys
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
import joblib
import numpy as np
import argparse
from model import TimeSeriesModel
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import json
import time
import pdb

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"



#################################### MAIN SECTION ############################################
def main(args):
    maindir = os.getcwd()+'/'+args.outstr
    if not os.path.exists(maindir):
        os.makedirs(maindir)

    # save args
    with open(os.path.join(maindir, 'args_{}.json'.format(args.outstr)), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4, separators=(',', ': '))

    model_dir=maindir+'/model0'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print('model_dir:',model_dir)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)

    batch_size = args.BATCHSIZE

    forecast_length = args.horizon
    backcast_length = args.history

    X_train, X_dev, X_test, \
    y_num_train, y_num_dev, y_num_test, \
    y_cat_train, y_cat_dev, y_cat_test = makedata(args.datadir, backcast_length, forecast_length,\
                                           sub_id=None, policy=args.data_policy, extra_vars=args.AVD)
    traingen  = data_generater(batch_size, X_train, y_num_train, y_cat_train, args.shuffle, args.SEED)
    valgen  = data_generater(batch_size, X_dev, y_num_dev, y_cat_dev, False, args.SEED)
    testgen  = data_generater(batch_size, X_test, y_num_test, y_cat_test, False, args.SEED)

    model = TimeSeriesModel(args)

    model.train_and_evaluate(model_dir, forecast_length, backcast_length, traingen, valgen, testgen)






####################################  DATA LOADER SECTION  ###################################
def get_time_encoding(x):
    '''
    input x: a 1-d array with time stamps in seconds, e.g. [0, 300, 600, ...]
    return the sin encoding of x over 24 hour periods
    '''
    t=np.sin( x*2*np.pi/86400)  # 86400 = 24 * 3600
    return t


def makedata(datadir, blen, flen, sub_id=None, policy='random', extra_vars=True):
    '''
    blen: backcast length
    flen: forecast length
    sub_id: subject id. If sub_id is None, then all subjects will be used
    extra_vars: whether to use extra variables. For now the extra variabels are just time encoding
    policy: among [random, ordered_by_pat]:
            random: randomly split all_patient data into 80/10/10 partition
            ordered_by_pat: for each patient, based on time-order, split into 80/10/10 partition
    if sub_id is not None, then policy has to be "ordered_by_pat"
    '''
    if (sub_id is None) and (policy=='random'):
        all_X = pickle.load(open(os.path.join(datadir, 'blen{}_flen{}_w-time'\
                                                        .format(blen, flen), 'All_X.pkl'), 'rb'))
        all_y_num = pickle.load(open(os.path.join(datadir, 'blen{}_flen{}_w-time'\
                                                             .format(blen, flen), 'All_y_num.pkl'), 'rb'))
        all_y_cat = pickle.load(open(os.path.join(datadir, 'blen{}_flen{}_w-time'\
                                                             .format(blen, flen), 'All_y_cat.pkl'), 'rb'))
        if extra_vars:
            X = []
            for i in range(all_X['Time'].shape[0]):
                time_encoding = get_time_encoding(all_X['Time'][i])
                X.append(np.column_stack((all_X['CGM'][i], time_encoding)))
            all_X = np.concatenate([x[np.newaxis, :, :] for x in X], axis=0)
        else:
            all_X = all_X['CGM']
        assert len(all_X) == len(all_y_num)
        assert len(all_X) == len(all_y_cat)

        sss = StratifiedShuffleSplit(test_size=0.3, random_state=0)
        X_train, X_test, y_num_train, y_num_test, y_cat_train, y_cat_test = \
                train_test_split(all_X, all_y_num, all_y_cat, \
                                 test_size=0.2, stratify=all_y_cat, random_state=0)

        X_dev, X_test, y_num_dev, y_num_test, y_cat_dev, y_cat_test = \
                train_test_split(X_test, y_num_test, y_cat_test, \
                                 test_size=0.5, stratify=y_cat_test, random_state=0)
    elif (sub_id is None) and (policy=='ordered_by_pat'):
        X_train, y_num_train, y_cat_train = [], [], []
        X_dev, y_num_dev, y_cat_dev = [], [], []
        X_test, y_num_test, y_cat_test = [], [], []
        flist = [x for x in os.listdir(os.path.join(datadir, 'blen{}_flen{}_w-time'.format(blen, flen))) \
                 if x.startswith('F-') and x.endswith('X.pkl')]
        for f in flist:
            pat_id = f.split('_')[0]
            pat_X = pickle.load(open(os.path.join(datadir, \
                     'blen{}_flen{}_w-time/{}_X.pkl'.format(blen, flen, pat_id)), 'rb'))
            pat_y_num = pickle.load(open(os.path.join(datadir, \
                     'blen{}_flen{}_w-time/{}_y_num.pkl'.format(blen, flen, pat_id)), 'rb'))
            pat_y_cat = pickle.load(open(os.path.join(datadir, \
                     'blen{}_flen{}_w-time/{}_y_cat.pkl'.format(blen, flen, pat_id)), 'rb'))

            if extra_vars:
                X = []
                for i in range(pat_X['Time'].shape[0]):
                    time_encoding = get_time_encoding(pat_X['Time'][i])
                    X.append(np.column_stack((pat_X['CGM'][i], time_encoding)))
                pat_X = np.concatenate([x[np.newaxis, :, :] for x in X], axis=0)
            else:
                pat_X = pat_X['CGM']

            X_train.append(pat_X[:int(0.8*len(pat_X))])
            y_num_train.append(pat_y_num[:int(0.8*len(pat_X))])
            y_cat_train.append(pat_y_cat[:int(0.8*len(pat_X))])

            X_dev.append(pat_X[int(0.8*len(pat_X)):int(0.9*len(pat_X))])
            y_num_dev.append(pat_y_num[int(0.8*len(pat_X)):int(0.9*len(pat_X))])
            y_cat_dev.append(pat_y_cat[int(0.8*len(pat_X)):int(0.9*len(pat_X))])

            X_test.append(pat_X[int(0.9*len(pat_X)):])
            y_num_test.append(pat_y_num[int(0.9*len(pat_X)):])
            y_cat_test.append(pat_y_cat[int(0.9*len(pat_X)):])
        X_train = np.concatenate(X_train, axis=0)
        y_num_train = np.concatenate(y_num_train, axis=0)
        y_cat_train = np.concatenate(y_cat_train, axis=0)
        X_dev = np.concatenate(X_dev, axis=0)
        y_num_dev = np.concatenate(y_num_dev, axis=0)
        y_cat_dev = np.concatenate(y_cat_dev, axis=0)
        X_test = np.concatenate(X_test, axis=0)
        y_num_test = np.concatenate(y_num_test, axis=0)
        y_cat_test = np.concatenate(y_cat_test, axis=0)
    elif sub_id is not None:
        all_X = pickle.load(open(os.path.join(datadir, 'blen{}_flen{}_w-time'.format(blen, flen), \
                                  '{}_X.pkl'.format(sub_id)), 'rb'))
        all_y_num = pickle.load(open(os.path.join(datadir, 'blen{}_flen{}_w-time'.format(blen, flen), \
                                  '{}_y_num.pkl'.format(sub_id)), 'rb'))
        all_y_cat = pickle.load(open(os.path.join(datadir, 'blen{}_flen{}_w-time'.format(blen, flen), \
                                  '{}_y_cat.pkl'.format(sub_id)), 'rb'))
        if extra_vars:
            X = []
            for i in range(all_X['Time'].shape[0]):
                time_encoding = get_time_encoding(all_X['Time'][i])
                X.append(np.column_stack((all_X['CGM'][i], time_encoding)))
            all_X = np.concatenate([x[np.newaxis, :, :] for x in X], axis=0)
        else:
            all_X = all_X['CGM']
        assert len(all_X) == len(all_y_num)
        assert len(all_X) == len(all_y_cat)

        if policy=='random':
            X_train, X_test, y_num_train, y_num_test, y_cat_train, y_cat_test = \
                    train_test_split(all_X, all_y_num, all_y_cat, \
                                     test_size=0.2, stratify=all_y_cat, random_state=0)

            X_dev, X_test, y_num_dev, y_num_test, y_cat_dev, y_cat_test = \
                    train_test_split(X_test, y_num_test, y_cat_test, \
                                     test_size=0.5, stratify=y_cat_test, random_state=0)
        elif policy=='ordered_by_pat':
            X_train = all_X[:int(0.8*len(all_X))]
            y_num_train = all_y_num[:int(0.8*len(all_y_num))]
            y_cat_train = all_y_cat[:int(0.8*len(all_y_cat))]

            X_dev = all_X[int(0.8*len(all_X)):int(0.9*len(all_X))]
            y_num_dev = all_y_num[int(0.8*len(all_y_num)):int(0.9*len(all_y_num))]
            y_cat_dev = all_y_cat[int(0.8*len(all_y_cat)):int(0.9*len(all_y_cat))]

            X_test = all_X[int(0.9*len(all_X)):]
            y_num_test = all_y_num[int(0.9*len(all_y_num)):]
            y_cat_test = all_y_cat[int(0.9*len(all_y_cat)):]



    print('Training samples {}, Hypoglycemia Ratio {}'\
            .format(len(y_cat_train), 1.0*sum(y_cat_train)/len(y_cat_train)))
    print('Dev samples {}, Hypoglycemia Ratio {}'\
            .format(len(y_cat_dev), 1.0*sum(y_cat_dev)/len(y_cat_dev)))
    print('Test samples {}, Hypoglycemia Ratio {}'\
            .format(len(y_cat_test), 1.0*sum(y_cat_test)/len(y_cat_test)))
    return X_train, X_dev, X_test, y_num_train, y_num_dev, y_num_test, y_cat_train, y_cat_dev, y_cat_test

def data_generater(batch_size, X, y_num, y_cat, shuffle=True, seed=0):
    done = False
    i = 0
    while (True):
        if i+batch_size < X.shape[0]:
            if shuffle:
                X_shuffle, y_num_shuffle, y_cat_shuffle = sklearn.utils.shuffle(X[i:i+batch_size],
                                                                                y_num[i:i+batch_size],
                                                                                y_cat[i:i+batch_size],
                                                                                random_state=seed)
                yield X_shuffle, y_num_shuffle, y_cat_shuffle, False
            else:
                yield X[i:i+batch_size], y_num[i:i+batch_size], y_cat[i:i+batch_size], False
            i += batch_size

        else:
            if shuffle:
                X_shuffle, y_num_shuffle, y_cat_shuffle = sklearn.utils.shuffle(X[i:i+batch_size],
                                                                                y_num[i:i+batch_size],
                                                                                y_cat[i:i+batch_size],
                                                                                random_state=seed)
                yield X_shuffle, y_num_shuffle, y_cat_shuffle, True
            else:
                yield X[i:i+batch_size], y_num[i:i+batch_size], y_cat[i:i+batch_size], True
            i = 0
###########################################################




def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    p = argparse.ArgumentParser()

    #use rnn?
    p.add_argument('-rnn', type=str2bool, default=True)

    #general hyperparameters
    p.add_argument('-LR', type=float, default=0.0002)
    p.add_argument('-BATCHSIZE', type=int, default=512)  # was 512
    p.add_argument('-NUMBLOCKS', type=int, default=7)
    p.add_argument('-HIDDEN', type=int, default=300)
    p.add_argument('-SEED', type=int, default=0)
    p.add_argument('-patience', type=int, default=10)
    p.add_argument('-hidden_layer_units', type=int, default=512)
    p.add_argument('-nb_blocks_per_stack', type=int, default=1)
    p.add_argument('-lstm_aggre', type=str, default='last', choices=['last', 'max_pool', 'avg_pool'])
    p.add_argument('-dropout', type=float, default=0.3)
    p.add_argument('-epochs', type=int, default=100)
    p.add_argument('-shuffle', type=str2bool, default=True)
    p.add_argument('-norm_mse', type=str2bool, default=True)

    #Add extra variables? (sine time encoding)
    p.add_argument('-AVD', type=str2bool, default=True)

    #Auxillary losses defined in the Umich paper
    p.add_argument('-IL', type=str2bool, default=True, help='regr loss, if True will do tsf task')
    p.add_argument('-FIL', type=str2bool, default=True, help='regr loss, if True will do tsf task')
    p.add_argument('-SL', type=str2bool, default=True, help='regr loss, if True will do tsf task')

    #hyperparams for additional losses
    p.add_argument('-prop', type=float, default=.1*100000)
    p.add_argument('-poww', type=float, default=1.0)
    p.add_argument('-fpoww', type=int, default=3)
    p.add_argument('-proportion', type=float, default=.3)
    # categorical classification loss
    p.add_argument('-cat_loss_weight', type=float, default=100.0, help='if > 0, will do classification task')
    p.add_argument('-cat_loss_mthd', type=str, default='sum_stack', choices=['per_stack', 'sum_stack'])
    p.add_argument('-pos_weight', type=float, default=1.0, help='pos class weight (for hypoglycemia) of cross entropy, need to be >=1.0')


    #prediction horizon and history lengths
    p.add_argument('-horizon', type=int, default=6)#
    p.add_argument('-history', type=int, default=48)

    p.add_argument('-datadir', type=str, default='./RWE_for_DNN')
    p.add_argument('-data_policy', type=str, default='ordered_by_pat', choices=['random', 'ordered_by_pat'])
    #where to save the outputs (a name for a new folder)
    p.add_argument('-outstr', type=str, default='TEST')



    args = p.parse_args()


    #NUMBER OF INPUT VARIABLES
    # in our case, whether to use the sine encoding of time
    args.nv=2 if args.AVD else 1


    main(args)




