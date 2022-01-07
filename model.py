
import os,datetime
import sys
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
import joblib
import numpy as np
import argparse
from network import network
import pdb

#############  TRAINING, AND EVALUATION SECTION ###################

class TimeSeriesModel(nn.Module):
    def __init__(self, args):
        super(TimeSeriesModel, self).__init__()
        self.args = args
        # self.net = network(device, backcast_length, forecast_length, args.NUMBLOCKS)

    def train_and_evaluate(self, model_dir, forecast_length, backcast_length, \
                                 traingen, valgen, testgen):

        pin_memory=True
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


        net = network(self.device, backcast_length, forecast_length, self.args.NUMBLOCKS, self.args)
        optimiser = optim.Adam(net.parameters(),lr=self.args.LR)

        self.fit(net, optimiser, traingen, valgen, model_dir)

        self.evaluate(net, optimiser, testgen, model_dir, partition='test')



    def fit(self, net, optimiser, traingen, valgen, model_dir):
        losss=self.mse_one
        losss2=self.mse  # for computing Lf loss in the Umich paper
        losss3=self.msedoubs  # for computing Lr loss in the Umich paper
        # classification loss
        loss_cat = nn.CrossEntropyLoss(weight=torch.tensor([1.0, self.args.pos_weight]).to(self.device))

        lossbonsum=1
        lossbonsumf=1
        magbonsum=1
        for i in range(self.args.NUMBLOCKS):
            lossbonsum = lossbonsum+(i+1)**self.args.poww
            if self.args.fpoww>0:
                lossbonsumf = lossbonsumf+(i+1)**self.args.fpoww
            magbonsum=magbonsum+1/(i+1)


        self.load(net, optimiser, model_dir, warm_start=False)
        net.to(self.device)
        net.train()

        train_loss=[]
        patience=self.args.patience
        unimproved=0
        for grad_step in range(self.args.epochs):
            batch_loss=[]
            total=0
            net.train()
            while(True):
                loss = torch.tensor(0.0).to(self.device)
                optimiser.zero_grad()
                x,target,target_cat, done=next(traingen)
                target_cat = torch.tensor(target_cat, dtype=torch.long).to(self.device)
                total=total+x.shape[0]
                # final_forecast: y_hat(fn') n is number of stacks
                # fores: [f1', f2', ...fn']
                # backs: [b1, b2, ...bn]
                # backsum: b1+b2+...bn
                # back_targets: inputs to all stack(x_{i-1}-b_{i-1}), backcast targets for all stacks
                final_forecast,fores,backs,backsum,back_targets,cat_logits= \
                        net(torch.tensor(x, dtype=torch.float).to(self.device))
                if self.args.FIL:
                    # Lf loss computed by all fi'
                    Lf = 1/lossbonsumf*losss2(fores, torch.tensor(target, dtype=torch.float)\
                                                                  .to(self.device), self.args.norm_mse)
                    loss += Lf
                if self.args.AVD:
                    # extract CGM only, for loss computing
                    x=x[:,:,0]
                if self.args.IL:
                    # Lr loss
                    Lr = 1.0/lossbonsum*losss3(backs,back_targets, self.args.norm_mse)
                    loss += self.args.proportion*Lr
                if self.args.SL:
                    # Lm loss
                    Lm = 1.0/magbonsum*self.calcsizeloss(backs)
                    loss += self.args.prop*Lm
                if self.args.cat_loss_weight > 0:
                    if self.args.cat_loss_mthd=='per_stack':
                        categorical_loss=loss_cat(torch.cat(cat_logits, dim=0),
                                              torch.cat([target_cat]*self.args.NUMBLOCKS, dim=0))
                    elif self.args.cat_loss_mthd=='sum_stack':
                        categorical_loss=loss_cat(
                            torch.sum(torch.cat([x.unsqueeze(2) for x in cat_logits], dim=2), dim=2),
                            target_cat)
                    loss += self.args.cat_loss_weight*categorical_loss

                loss.backward()
                optimiser.step()
                batch_loss.append(loss.item())
                if done:
                    break
            train_loss.append(np.mean(batch_loss))
            print('grad_step = '+str(grad_step)+' loss = '+str(train_loss[-1]))


            ##### Validation at the end of epoch #####
            val_loss = []
            val_batch_loss=[]
            targets, preds= [],[]
            y_trues, y_preds = [], []
            total=0
            best_metric = 0.
            net.eval()
            while(True):
                with torch.no_grad():
                    loss = torch.tensor(0.0).to(self.device)
                    x, target, target_cat, done=next(valgen)
                    target_cat = torch.tensor(target_cat, dtype=torch.long).to(self.device)
                    total=total+x.shape[0]
                    final_forecast,fores,backs,backsum,back_targets, cat_logits= \
                            net(torch.tensor(x, dtype=torch.float).to(self.device))
                    if any([self.args.IL, self.args.FIL, self.args.SL]):
                        loss = losss(final_forecast, torch.tensor(target, dtype=torch.float).to(self.device), False)
                    if self.args.cat_loss_weight > 0:
                        if self.args.cat_loss_mthd=='per_stack':
                            categorical_loss=loss_cat(torch.cat(cat_logits, dim=0),
                                                  torch.cat([target_cat]*self.args.NUMBLOCKS, dim=0))
                        elif self.args.cat_loss_mthd=='sum_stack':
                            categorical_loss=loss_cat(
                                torch.sum(torch.cat([x.unsqueeze(2) for x in cat_logits], dim=2),
                                                      dim=2), target_cat)
                        loss += self.args.cat_loss_weight*categorical_loss
                    val_batch_loss.append(loss.item())

                    # regression prediction
                    targets.append(target)
                    preds.append(final_forecast.cpu().numpy())

                    # categorical classification logits
                    if self.args.cat_loss_weight > 0:
                        y_trues.append(target_cat.cpu().numpy())
                        pred_logits = F.softmax(torch.sum(
                            torch.cat([x.unsqueeze(2) for x in cat_logits], dim=2), dim=2),
                            dim=1).cpu().numpy()
                        y_preds.append(pred_logits)

                    if done:
                        break
            val_loss.append(np.mean(val_batch_loss))
            # compute EER, decoded from the regression and categorical classification respectively
            targets = np.concatenate(targets, axis=0)
            preds = np.concatenate(preds, axis=0)
            reg_eer, reg_sens, reg_spec = compute_reg_eer_sens_spec(targets, preds,'any')
            if self.args.cat_loss_weight > 0:
                y_trues = np.concatenate(y_trues, axis=0)
                y_preds = np.concatenate(y_preds, axis=0)
                cls_eer, cls_sens, cls_spec = compute_cls_eer_sens_spec(y_trues, y_preds,'any')
            else:
                cls_sens=0.0
                cls_spec=0.0


            print('val loss: {:.4f}, reg_sens: {:.4f}, reg_spec: {:.4f}, cat_sens: {:.4f}, cat_spec: {:.4f}'.format(val_loss[-1], reg_sens, reg_spec, cls_sens, cls_spec))

            # select model by HG classification results
            if not any([self.args.IL, self.args.FIL, self.args.SL]):
                # lstm-cls case
                metric = (cls_sens+cls_spec)/2
            elif any([self.args.IL, self.args.FIL, self.args.SL]) and self.args.cat_loss_weight > 0:
                # MT-NB case
                metric = (reg_sens+reg_spec+cls_sens+cls_spec)/4
            else:
                # NB-regr case
                metric = (reg_sens+reg_spec)/2
            if metric > best_metric:
                print('eval metric improved')
                best_metric = metric
                unimproved=0
                self.save(net, optimiser, grad_step, model_dir)
            else:
                unimproved +=1
                print('eval metric did not improve for '+str(unimproved)+'th time')
            if unimproved>patience:
                print('Finished.')
                break
        del net



    def evaluate(self, net, optimiser, testgen, model_dir, partition='test'):
        print('Evaluating')
        net.to(self.device)
        net.eval()
        self.load(net, optimiser, model_dir, warm_start=False)
        net.to(self.device)
        net.eval()

        with torch.no_grad():
            totalpoints=0
            rmselosses=[]
            targets, preds = [], []  # exact values
            y_trues, y_preds = [], []  # categorical labels
            while(True):
                x,target,target_cat, done=next(testgen)
                totalpoints = totalpoints+x.shape[0]
                final_forecast,_,backs,_,_, cat_logits = net(torch.tensor(x, dtype=torch.float)\
                                                           .to(self.device))
                rmselosses.append(self.mse_lastpointonly(final_forecast, \
                                    torch.tensor(target, dtype=torch.float).to(self.device))\
                                    .item()*x.shape[0])
                # regression prediction
                targets.append(target)
                preds.append(final_forecast.cpu().numpy())

                # categorical classification logits
                if self.args.cat_loss_weight > 0:
                    y_trues.append(target_cat)
                    pred_logits = F.softmax(torch.sum(
                        torch.cat([x.unsqueeze(2) for x in cat_logits], dim=2), dim=2),
                        dim=1).cpu().numpy()
                    y_preds.append(pred_logits)
                if done:
                    break
            # compute EER, decoded from the regression results and categorical classification respectively
            targets = np.concatenate(targets, axis=0)
            preds = np.concatenate(preds, axis=0)
            reg_eer, reg_sens, reg_spec = compute_reg_eer_sens_spec(targets, preds,'any')
            if self.args.cat_loss_weight > 0:
                y_trues = np.concatenate(y_trues, axis=0)
                y_preds = np.concatenate(y_preds, axis=0)
                cls_eer, cls_sens, cls_spec = compute_cls_eer_sens_spec(y_trues, y_preds,'any')
            else:
                cls_sens=0.0
                cls_spec=0.0


            rmse = np.sqrt(np.sum(np.asarray(rmselosses))/totalpoints)
            print('RMSE: {:.4f}, reg_sens: {:.4f}, reg_spec: {:.4f}, cat_sens: {:.4f}, cat_spec: {:.4f}'.format(rmse, reg_sens, reg_spec, cls_sens, cls_spec))


            #write final loss
            t=open(model_dir+"/"+\
                    str(rmse)+\
                    ".{}Rmseout".format(partition),"w")
            t=open(model_dir+"/"+"{}_reg-sens{:.4f}_reg-spec{:.4f}_cls-sens{:.4f}_cls-spec{:.4f}".format(partition, reg_sens, reg_spec, cls_sens, cls_spec), "w")
            #dump out predictions to be used in ensembling
            joblib.dump(targets,model_dir+'/{}_reg_targets.pkl'.format(partition))
            joblib.dump(preds,model_dir+'/{}_reg_preds.pkl'.format(partition))
            if self.args.cat_loss_weight > 0:
                joblib.dump(y_trues,model_dir+'/{}_cls_targets.pkl'.format(partition))
                joblib.dump(y_preds,model_dir+'/{}_cls_preds.pkl'.format(partition))


    #SAVE AND LOAD FUNCTIONS
    def save(self, model, optimiser, grad_step, mdir):
        torch.save({
            'grad_step': grad_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimiser.state_dict(),
        }, mdir+'/model_out.th')
        print('saved model to {}'.format(mdir+'/model_out.th'))


    def load(self, model, optimiser, mdir, warm_start=False):
        if os.path.exists(mdir+'/model_out.th'):
            checkpoint = torch.load(mdir+'/'+'model_out.th')
            model.load_state_dict(checkpoint['model_state_dict'])
            if warm_start:
                optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
                grad_step = checkpoint['grad_step']
            print('loaded model from {}'.format(mdir+'/'+'model_out.th'))



    #LOSS FUNCTIONS
    def mse_one(self, output, target, normalize=False):
        if normalize:
            return torch.mean(((output-target)/target)**2)
        else:
            return torch.mean((output - target)**2)

    def mse(self, output, target, normalize=False):
        '''
        For computing Lf loss - see UMich paper for more details
        '''
        loss=self.mse_one(output[0],target, normalize)
        for r in np.arange(1,len(output)):
            mul=(r+1)**self.args.fpoww
            loss=loss+mul*self.mse_one(output[r],target, normalize)
        return loss

    def msedoubs(self, output, target, normalize=False):
        '''
        For computing Lr loss - see UMich paper for more details
        '''
        loss=self.mse_one(output[0],target[0], normalize)
        for r in np.arange(1,len(output)):
            mul=(r+1)**self.args.poww
            loss=loss+mul*self.mse_one(output[r],target[r], normalize)
        return loss


    def calcsizeloss(self, output):
        '''
        For computing Lm loss - see UMich paper for more details
        '''
        for r in np.arange(0,len(output)):
            if r==0:
                loss=1/torch.sum(torch.abs(output[r]))/(r+1)
            else:
                loss=loss+1/torch.sum(torch.abs(output[r]))/(1+r)
        return loss


    def mse_lastpointonly(self, output, target):
        '''
        Used during evaluation. MSE was computed using only the last point in each predicted horizon
        '''
        output=output[:,-1]
        target=target[:,-1]
        loss = torch.mean((output - target)**2)
        return loss


def compute_reg_eer_sens_spec(targets, preds, condition='any'):
    '''
    decode from regression branch
    both targets and preds is an array of size(batch_size, pred_range), e.g. (512, 6)
    '''
    fprs, tprs = [], []
    best_diff = 100.0
    best_fpr, best_fnr = 0.0, 0.0
    y_true = binarize(targets, 70, condition)  # always keep the thres for target values as 70
    for thres in np.arange(60, 99, 1):
        y_pred = binarize(preds, thres, condition)
        p, n, tp, tn, fp, fn = compute_stats(y_true, y_pred)
        sens = 0. if p==0 else 1.0*tp/p
        spec = 0. if n==0 else 1.0*tn/n
        # fpr = 0. if n==0 else 1.0*fp/n
        # fnr = 0. if p==0 else 1.0*fn/p
        # if abs(fpr - fnr) < best_diff:
        if abs(sens - spec) < best_diff:
            # best_diff = abs(fpr - fnr)
            best_diff = abs(sens - spec)
            # best_fpr = fpr
            # best_fnr = fnr
            best_cutoff = thres
            # best_sens = 1 - best_fnr
            # best_spec = 1 - best_fpr
            best_sens = sens
            best_spec = spec
    return best_cutoff, best_sens, best_spec
def compute_cls_eer_sens_spec(targets, preds, condition='any'):
    '''
    decode from classification branch
    preds is an array of size(batch_size, 2), e.g. (512, 2)
    targets is an array of size (batch_size, ), i.e. binary label
    '''
    fprs, tprs = [], []
    best_diff = 100.0
    best_fpr, best_fnr = 0.0, 0.0
    y_true = targets
    for thres in np.arange(0.02, 0.99, 0.01):
        y_pred = []
        for pred in preds:
            if pred[1] > thres:
                y_pred.append(1)
            elif pred[1] <= thres:
                y_pred.append(0)
        p, n, tp, tn, fp, fn = compute_stats(y_true, y_pred)
        sens = 0. if p==0 else 1.0*tp/p
        spec = 0. if n==0 else 1.0*tn/n
        # fpr = 0. if n==0 else 1.0*fp/n
        # fnr = 0. if p==0 else 1.0*fn/p
        # if abs(fpr - fnr) < best_diff:
        if abs(sens - spec) < best_diff:
            # best_diff = abs(fpr - fnr)
            best_diff = abs(sens - spec)
            # best_fpr = fpr
            # best_fnr = fnr
            best_cutoff = thres
            # best_sens = 1 - best_fnr
            # best_spec = 1 - best_fpr
            best_sens = sens
            best_spec = spec

    return best_cutoff, best_sens, best_spec
def binarize(y, thres=70, condition='any'):
    '''
    condition: "any" or "last". when condition="any", as long as anyone of the values in the range < 70
    the range will be assigned positive, when condition="last", only consider last value in the range
    y is a array of size(batch_size, pred_range), e.g. (512, 6)
    '''
    out = []
    for i in range(len(y)):
        if condition == 'any':
            if any([x<thres for x in y[i]]):
                out.append(1)
            else:
                out.append(0)
        elif condition == 'last':
            if y[i][-1]<thres:
                out.append(1)
            else:
                out.append(0)

    return out

def compute_stats(y_true, y_pred):
    '''
    both y_true and y_pred are binarized list
    '''
    p = len([x for x in y_true if x==1])
    n = len([x for x in y_true if x==0])
    tp, tn, fp, fn = 0,0,0,0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        if y_true[i] == 0 and y_pred[i] == 0:
            tn += 1
        if y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
        if y_true[i] == 1 and y_pred[i] == 0:
            fn += 1
    return p, n, tp, tn, fp, fn
