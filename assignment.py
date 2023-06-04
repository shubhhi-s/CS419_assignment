#!/usr/bin/env python
# coding: utf-8

import pickle
import torch
import matplotlib.pyplot as plt
import random
import pandas as pd

def set_seed(seed:int=42):
    """Sets the seed for torch, numpy and random
    Args:
        seed (int): [description]
    """
    random.seed(seed)
    torch.manual_seed(seed)

class LinearRegression:
    def __init__(self, xdim, args) -> None:
        """This is the model class for Linear Regression
        
        Args:
            xdim - Dimension of input
            args - dictionary containing hyper-parameters used in the model 
        """
        self.xdim = xdim
        self.args = args 

        self.weights, self.bias = self.init_weights(xdim)
        assert self.weights.shape == (xdim, )


    def init_weights(self, xdim):
        """Initializes the weights of the logistic regression model. 
        We have weights of shape (xdim,) and bias a scalar

        Returns:
            w, b
        """
        w, b = None, None

        ## TODO

        ## END TODO
        
        return w, b

    def forward(self, batch_x):
        """ Runs forward on a batch of features x and returns the predictions in yhat variable
        The code must be vectorized.

        Returns:
            the predictions
        """
        assert (len(batch_x.size())==2)

        yhat = None
        
        ## TODO
        
        ## End TODO
        
        assert yhat.shape == (batch_x.shape[0], )
        return yhat

    def backward(self, batch_x, batch_y, batch_yhat):
        """Computes the gradients and updates the weights with the self.args.lr
            
            Returns the updated weights
        """
        assert (len(batch_x.size())==2)
        assert (len(batch_y.size())==1)
        assert (len(batch_yhat.size())==1)
        
        weights_new, bias_new = None, None
        
        ## TODO
        
        ## End TODO
        
        self.weights = weights_new
        self.bias = bias_new
        return weights_new, bias_new

    def loss(self, y, y_hat):
        """Computes the loss and returns a scalar
        """
        assert (len(y.size())==1)
        assert (len(y_hat.size())==1)
        loss = 0

        ## TODO

        ## End TODO

        assert len(loss.shape) == 0, "loss should be a torch scalar"
        return loss
    
    def score(self, yhat, y):
        """Computes the negative opf mean squared error as score.

        Args:
            yhat (_type_): predictions
            y (_type_): _targets

        Returns:
            negative of mse
        """
        assert (len(y.size())==1)
        assert (len(yhat.size())==1)
        return - torch.mean(torch.square(y-yhat))
    


class EarlyStoppingModule(object):
  """
    Module to keep track of validation score across epochs
    Stop training if score not imroving exceeds patience
  """  
  def __init__(self, args):
    """
      input : args 
      patience: number of epochs to wait for improvement in validation score
      delta: minimum difference between two validation scores that can be considered as an improvement 
      best_score: keeps track of best validation score observed till now (while training)
      num_bad_epochs: keeps track of number of training epochs in which no improvement has been observed
      should_stop_now: boolean flag deciding whether training should early stop at this epoch
    """
    self.args = args
    self.patience = args.patience 
    self.delta = args.delta
    self.best_score = None
    self.num_bad_epochs = 0 
    self.should_stop_now = False

  def save_best_model(self, model, epoch): 
    fname =f"./{self.args.model_type}_bestValModel.pkl"
    pickle.dump(model, open(fname,"wb"))
    print(f"INFO:: Saving best validation model at epoch {epoch}")
    

  def load_best_model(self):
    fname =f"./{self.args.model_type}_bestValModel.pkl"
    try: 
      model = pickle.load(open(fname,"rb"))
      print(f"INFO:: Loading best validation model from {fname}")
    except Exception as e:
      print(f"INFO:: Cannot load best model due to exception: {e}") 

    return model

  def check(self, curr_score, model, epoch) :
    """Checks whether the current model has the best validation accuracy and decides to stop or proceed.
    If the current score on validation dataset is the best so far, it saves the model weights and bias.

    Args:
        curr_score (_type_): Score of the current model
        model (_type_): Trained Logistic/Linear model
        epoch (_type_): current epoch

    Returns:
        self.stop_now: Whether or not to stop

    Task1: Check for stoppage as per the early stopping criteria 
    Task2: Save best model as required
    """    

    ## TODO
    
    ## END TODO
    return self.should_stop_now 




def minibatch(trn_X, trn_y, batch_size):
    """
    Function that yields the next minibatch for your model training
    IMPORTANT: DO NOT RETURN
    """
    #TODO
    
    #TODO 
    pass

def split_data(X:torch.Tensor, y:torch.Tensor, split_per=0.6):
    """
    Splits the dataset into train and test.
    """
    # TODO

    # END TODO

    """
    return X_train, y_train, X_test, y_test
    """



def train(args, X_tr, y_tr, X_val, y_val, model):
    es = EarlyStoppingModule(args)
    losses = []
    val_acc = []
    epoch_num = 0
    while (epoch_num<=args.max_epochs): 
        for idx, (batch_x, batch_y) in enumerate(minibatch(X_tr, y_tr, args.batch_size)):
            if idx == 0:
                assert batch_x.shape[0] == args.batch_size
            batch_yhat = model.forward(batch_x)
            losses.append(model.loss(batch_y, batch_yhat).item())
            updated_wts = model.backward(batch_x, batch_y, batch_yhat)
        
        val_score = model.score(model.forward(torch.Tensor(X_val)), torch.Tensor(y_val))
        print(f"INFO:: Validation score at epoch {epoch_num}: {val_score}")
        val_acc.append(val_score)
        if es.check(val_score,model,epoch_num):
          break
        epoch_num +=1  
    return losses, val_acc


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


if __name__ == '__main__':

    set_seed(0)

    # """
    # Linear Regression
    # """


    print("MODEL:: LINEAR REGRESSION")
    args = {'batch_size': 128,
        'max_epochs': 500, 
        'lr': 1e-3,
        'patience' : 10,
        'delta' : 1e-4,
        'model_type': 'linear'} 

    args = objectview(args)
    
    data = pd.read_csv('dataset4.csv', index_col=0)
    X = (data.iloc[:,:-1].to_numpy())
    y = (data.iloc[:,-1].to_numpy())
    X, y = torch.Tensor(X), torch.Tensor(y)

    X_train, y_train, X_val, y_val = split_data(X, y,)
    LinR = LinearRegression(X_train.shape[1], args)
    losses, val_acc = train(args, X_train, y_train,X_val, y_val,  model=LinR)
    
    es = EarlyStoppingModule(args)
    best_model = es.load_best_model()
    
    print(LinR.score(best_model.forward(torch.Tensor(X_val)), torch.Tensor(y_val)))
