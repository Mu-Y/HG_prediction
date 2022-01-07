import numpy as np
import pickle
import os
import pdb
import sys

def judge_label(input_val):
    '''
    if input_val < 70 --> hypoglycemia(1)
    if input_val >= 70 --> not hypoglycemia(0)
    '''
    if input_val < 70:
        return 1
    else:
        return 0


def pkl_to_X_y(pkl_data, out_dir, blen, flen, ignore_NA=True):
    '''
    blen: backcast length, unit in sample (not seconds), e.g. 12
    flen, forecast length, unit in sample (not seconds), e.g 6
    if ignore_NA is True, will only use data sample where no NA appears in its feature vector and target y
    '''
    outdir = './{}/blen{}_flen{}_w-time/'.format(out_dir, blen, flen)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    X_all, time_stamps_all, y_all_num, y_all_cat = [], [], [], []
    for pat_id, feats_dict in pkl_data.items():
        X_pat, time_stamps_pat, y_pat_num, y_pat_cat = [], [], [], []
        for sec, feat in feats_dict.items():
            b_time_steps = [sec-(i*300) for i in reversed(range(0, blen))]
            f_time_steps = [sec+(j*300) for j in range(1, flen+1)]
            if ignore_NA is True:
                if any([t not in feats_dict for t in b_time_steps]):
                    # some backward time steps not available
                    continue
                if any([t not in feats_dict for t in f_time_steps]):
                    # some forward time steps not available
                    continue
                if any([feats_dict[t]['glucose'] == 'NA' for t in b_time_steps]):
                    # some backward glucose values not available
                    continue
                if any([feats_dict[t]['glucose'] == 'NA' for t in f_time_steps]):
                    # some forward glucose values not available
                    continue
                if any([x=='NA' for x in list(feat.values())]):
                    # to be compatible with the feature-based approach
                    continue
                X_window = [feats_dict[t]['glucose'] for t in b_time_steps]
                y_num_window = [feats_dict[t]['glucose'] for t in f_time_steps]
                time_stamps_window = [feats_dict[t]['Time'] for t in b_time_steps]
                if any([judge_label(y) for y in y_num_window]):
                    y_cat_window = 1
                else:
                    y_cat_window = 0

                X_pat.append(X_window)
                time_stamps_pat.append(time_stamps_window)
                y_pat_num.append(y_num_window)
                y_pat_cat.append(y_cat_window)
            else:
                raise NotImplementedError("Haven't implemented the ignore_NA=False case")
        X_pat = np.array(X_pat)
        time_stamps_pat = np.array(time_stamps_pat)
        assert X_pat.shape[0] == time_stamps_pat.shape[0]  # same time length
        y_pat_num = np.array(y_pat_num)
        y_pat_cat = np.array(y_pat_cat)
        # per-subject data
        with open(os.path.join(outdir, '{}_X.pkl'.format(pat_id)), 'wb') as f:
            pickle.dump({'CGM': X_pat, 'Time': time_stamps_pat}, f)
        with open(os.path.join(outdir, '{}_y_num.pkl'.format(pat_id)), 'wb') as f:
            pickle.dump(y_pat_num, f)
        with open(os.path.join(outdir, '{}_y_cat.pkl'.format(pat_id)), 'wb') as f:
            pickle.dump(y_pat_cat, f)
        X_all.append(X_pat)
        time_stamps_all.append(time_stamps_pat)
        y_all_num.append(y_pat_num)
        y_all_cat.append(y_pat_cat)
        print('processed subject {}, samples {}, hypoglycemia ratio {}'.format(pat_id, \
                                                   y_pat_cat.shape[0], \
                                                   1.0*np.sum(y_pat_cat)/y_pat_cat.shape[0]))
    # all subject data
    X_all = np.concatenate(X_all, axis=0)
    time_stamps_all = np.concatenate(time_stamps_all, axis=0)
    assert X_all.shape[0] == time_stamps_all.shape[0]  # same time length
    y_all_num = np.concatenate(y_all_num, axis=0)
    y_all_cat = np.concatenate(y_all_cat, axis=0)
    with open(os.path.join(outdir, 'All_X.pkl'), 'wb') as f:
        pickle.dump({'CGM': X_all, 'Time': time_stamps_all}, f)
    with open(os.path.join(outdir, 'All_y_num.pkl'), 'wb') as f:
        pickle.dump(y_all_num, f)
    with open(os.path.join(outdir, 'All_y_cat.pkl'), 'wb') as f:
        pickle.dump(y_all_cat, f)
    print('Finished all subjects, samples {}, hypoglycemia ratio {}'.format(\
                                                   y_all_cat.shape[0], \
                                                   1.0*np.sum(y_all_cat)/y_all_cat.shape[0]))






if __name__ == '__main__':
    # feats.pkl stored data directly converted from the given csv files in RWE
    # The only preprocessing in feats.pkl is: time stamps are contiguous, plus features
    # That is, for the missing values for some missing time stamps in the original RWE data, they
    # are filled in as 'NA'
    feats_pkl = sys.argv[1]
    out_dir = sys.argv[2]
    RWE_data = pickle.load(open(feats_pkl, 'rb'))
    pkl_to_X_y(RWE_data, out_dir, blen=48, flen=6, ignore_NA=True)
