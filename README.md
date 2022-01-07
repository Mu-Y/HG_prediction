# glucose-prediction-umich

Code for our ICASSP 2022 paper "Joint hypoglycemia prediction and glucose forecasting via deep multi-task learning", by Mu Yang, Darpit Dave, Madhav Erraguntla, Gerard L. Cote and Ricardo Gutierrez-Osuna. 

Right now this repo contains codes for model and training scripts. Due to our data collection protocol, the dataset used in our paper cannot be made public yet. We'll update this repo when the dataset is ready to publish. Alternatively, reach out to mu.yang@utdallas.edu if you have any questions.


## Install and dependencies

The code is tested on python 3.7.0, pytorch 1.7.1 (with CUDA 11.0). Other dependencies include:
```
pip install joblib sklearn datetime
```


## Run code

1. Preprocess data. TODO


2. Train and evaluate models. The multitask learning model (MT-NB-L*, MT-NB) and single-task model (NB-tsf, LSTM-cls) was controled by loss weights arguments (see `train.py`):
  - `-cat_loss_weight`: control the classification task. If > 0, will do classification task.
  - `-IL`, `-FIL`, `-SL`: control the time series forecastng task. When all three are True, will do the time series forecasting task described in [N-BEATS](http://ceur-ws.org/Vol-2675/paper18.pdf).
  - `norm_mse`: Bool, control whether use the Normarlized MSE loss.
  - You may need to change `-outstr` for output model path, and `-datadir` for data path.
  
  some example commands:
  ```
  # MT-NB-L*-tsf and MT-NB-L*-cls
  python train.py -outstr=test_mt_nb_l  -cat_loss_weight=100.0 -norm_mse=True -IL=True -FIL=True -SL=True
  
  # MT-NB-tsf and MT-NB-cls
  python train.py -outstr=test_mt_nb  -cat_loss_weight=100.0 -norm_mse=False -IL=True -FIL=True -SL=True
  
  # NB-tsf
  python train.py -outstr=test_nb_tsf  -cat_loss_weight=0.0 -norm_mse=False -IL=True -FIL=True -SL=True
  
  # LSTM-cls
  python train.py -outstr=test_lstm  -cat_loss_weight=1.0 -norm_mse=False -IL=False -FIL=False -SL=False
  ```


## Acknowledgement

Codes are adapted from https://gitlab.eecs.umich.edu/mld3/deep-residual-time-series-forecasting, which is the official implementation of [N-BEATS](http://ceur-ws.org/Vol-2675/paper18.pdf) for glucose forecasting.
 
