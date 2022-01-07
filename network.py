
import os,datetime
import sys
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
import joblib
import numpy as np
import pdb



class Block(nn.Module):

    def __init__(self, units, device, backcast_length, forecast_length, args):
        super(Block, self).__init__()
        self.args = args
        self.backlen=backcast_length
        self.forecast_length=forecast_length
        self.input=1
        self.outer=1
        if self.args.AVD:
            self.input=self.args.nv
        self.device = device
        if not self.args.rnn:
            self.lin1 = nn.Linear(backcast_length*self.args.nv, units)
            self.lin2 = nn.Linear(units, units)
            self.lin3 = nn.Linear(units, units)
            self.lin4 = nn.Linear(units, units)
            self.backcast_layer = nn.Linear(units, units)
            self.forecast_layer = nn.Linear(units, units)
            self.backcast_out = nn.Linear(units, backcast_length)
            self.forecast_out = nn.Linear(units, forecast_length)
        if self.args.rnn:
            self.units=args.HIDDEN
            self.bs=args.BATCHSIZE
            self.lstm=nn.LSTM(self.input,self.units, num_layers=2,batch_first=True,bidirectional=True)
            self.lin=nn.Linear(self.units *2, (self.backlen+self.forecast_length)*self.outer)
            # self.h_0=(torch.zeros(4,self.bs,self.units)).to(self.device)#,
            # self.c_0 = (torch.zeros(4,self.bs,self.units)).to(self.device)#,
        if self.args.cat_loss_weight > 0:
            # linear layers for the categorical classification task
            self.lin_cat1 = nn.Linear(self.units *2, self.units)
            self.lin_cat2 = nn.Linear(self.units, 2)
            self.dropout = nn.Dropout(self.args.dropout)


    def forward(self, x):
        if not self.args.rnn:
            if self.args.AVD:
                x = F.relu(self.lin1(x.flatten(1,-1).to(self.device)))
            else:
                x = F.relu(self.lin1(x.to(self.device)))
            x = F.relu(self.lin2(x))
            x = F.relu(self.lin3(x))
            x = F.relu(self.lin4(x))
            theta_b = F.relu(self.backcast_layer(x))
            theta_f = F.relu(self.forecast_layer(x))
            out = self.backcast_out(theta_b)
            forecast = self.forecast_out(theta_f)
            return out,forecast
        if self.args.rnn:
            # origbs=x.size()[0]
            # if origbs<self.bs:
            #     if self.args.AVD:
            #         x=F.pad(input=x, pad=( 0,0,0,0,0,self.bs-origbs), mode='constant', value=0)
            #     else:
            #         x=F.pad(input=x, pad=( 0,0,0,self.bs-origbs), mode='constant', value=0)
            # self.h_0=self.h_0.data
            # self.c_0=self.c_0.data
            if not self.args.AVD:
                x=x.unsqueeze(2)
            # lstm_out, (self.h_0,self.c_0) = self.lstm(x, (self.h_0,self.c_0))
            lstm_out, _ = self.lstm(x)
            #print(lstm_out[10,-1,10])
            if self.args.lstm_aggre == 'last':
                out=self.lin(lstm_out[:,-1,:])
            elif self.args.lstm_aggre == 'max_pool':
                out, _ = torch.max(lstm_out, dim=1)
                out=self.lin(out)
            elif self.args.lstm_aggre == 'avg_pool':
                out = torch.mean(lstm_out, dim=1)
                out=self.lin(out)
            # categorical classification task
            if self.args.cat_loss_weight > 0:
                cat_logits = self.lin_cat1(self.dropout(lstm_out[:,-1,:]))  # last
                cat_logits = self.lin_cat2(F.relu(self.dropout((cat_logits))))
            else:
                cat_logits=None
            return out[:,:self.backlen*self.outer],out[:,self.backlen*self.outer:], cat_logits



class Stack(nn.Module):

    def __init__(self, nb_blocks_per_stack, hidden_layer_units, device, \
                   forecast_length, backcast_length, args):
        super(Stack, self).__init__()
        self.args = args
        self.device=device
        self.backcast_length=backcast_length
        self.fl=forecast_length
        self.blocks = nn.ModuleList([Block(hidden_layer_units, device, \
                                        backcast_length, forecast_length, self.args)])
        for block_id in range(1,nb_blocks_per_stack):
            block = Block(hidden_layer_units, device, backcast_length, forecast_length, self.args)
            self.blocks.append(block)



    def forward(self, x, backcast,backsum,device):
        '''
        params:
            x: f1+f2+...f_{i-1}
            backcast: input to this stack (x_{i-1}-b_{i-1})
        returns:
            x: f1+f2+...fi(fi')
            backcast： input backcast - bi(output of this stack), this is the input to next stack
            backsum: b1+b2+...bi
            fores: fi'   #[f1, f2, ...fi]
            backs: bi   #[b1, b2, ...bi]
            backtargs: input to this stack(x_{i-1}-b_{i-1}), backcast target for a particular stack
        '''
        backs=[]
        fores=[]
        backtargs=[]
        cat_logits=[]
        for block_id in range(len(self.blocks)):
            b, f, cat_logit = self.blocks[block_id](backcast)
            #backcast = torch.cat( [(backcast[:,:,0] - b).view([-1,self.backcast_length,1]),backcast[:,:,1:] ],dim=2)
            if self.args.AVD:
                backtargs.append(backcast.clone()[:,:,0])
                backcast2=backcast.clone()
                backcast2[:,:,0]=backcast2[:,:,0]-b
                backcast=backcast2
            else:
                backtargs.append(backcast.clone())
                backcast=backcast-b
            backsum=backsum+b
            x= x + f
            #for loss calculation
            fores.append(x.clone())
            backs.append(b.clone())
            if cat_logit is not None:
                cat_logits.append(cat_logit.clone())
        return x,backcast,backsum,fores,backs,backtargs, cat_logits





class network(nn.Module):
    def __init__(self,device,backcast_length,forecast_length,nb_stacks,args):
        super(network, self).__init__()
        self.args = args
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.hidden_layer_units = self.args.hidden_layer_units
        self.nb_blocks_per_stack = self.args.nb_blocks_per_stack
        self.nb_stacks=nb_stacks
        self.device = device
        self.stacks = nn.ModuleList([Stack( self.nb_blocks_per_stack, self.hidden_layer_units, \
                                            self.device, forecast_length, backcast_length, \
                                            self.args)])

        for stack_id in range(1,self.nb_stacks):
            self.stacks.append(Stack( self.nb_blocks_per_stack, self.hidden_layer_units, \
                                      self.device, forecast_length, backcast_length, \
                                      self.args))
        self.to(self.device)





    def forward(self, backcast):
        '''
        params:
            backcast: input data
        returns:
            forecast: y_hat(fn') n is number of stacks
            fores: [f1', f2', ...fn']
            backs: [b1, b2, ...bn]
            backsum: b1+b2+...bn
            backtargs: inputs to all stack(x_{i-1}-b_{i-1}), backcast targets for all stacks
        '''
        forecast = torch.zeros(size=(backcast.size()[0], self.forecast_length,)).to(self.device)
        backsum = torch.zeros(size=(backcast.size()[0], self.backcast_length,)).to(self.device)
        fores=[]
        backs=[]
        backtargs=[]
        cat_logits_all=[]
        for stack_id in range(len(self.stacks)):
            forecast,backcast,backsum,curfores,curbacks,cbacktargs, cat_logits =self.stacks[stack_id](forecast,backcast,backsum,self.device)
            for ff in curfores:
                fores.append(ff)
            for ff in curbacks:
                backs.append(ff)
            for ff in cbacktargs:
                backtargs.append(ff)
            if len(cat_logits) > 0:
                for ff in cat_logits:
                    cat_logits_all.append(ff)
        return forecast,fores,backs,backsum,backtargs,cat_logits_all
