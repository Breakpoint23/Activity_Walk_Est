import torch
import numpy as np
import matplotlib.pyplot as plt
import pylsl
import sys,os
import pickle

from importlib import resources

from ActWalkEst.SpeedEst.model import CNN
from ActWalkEst.utils.lsl_imu import DataInlet,SetupStreams

class Prediction():
    def __init__(self,walking_index=1):
        self.buffer=[]
        self.filtered_buffer=[]
        self.walking_index=walking_index
        with resources.path('ActWalkEst.resources','walking_model1.h5') as p:
            self.MODEL_PATH=str(p)
        self.model=self.load_model()
        with resources.path('ActWalkEst.resources','walking_norm.pickle') as p:
            self.NORMALIZER_PATH=str(p)
        
        self.normalizer=self.load_normalizer()
        self.acquisition=SetupStreams()
        return None
    


    def load_model(self):
        model = CNN(input_features=1,input_length=200,num_classes=1)
        model.load_state_dict(torch.load(self.MODEL_PATH))
        model.eval()
        return model
    
    def load_normalizer(self):
        normalizer=pickle.load(open(self.NORMALIZER_PATH,'rb'))
        return normalizer
    
    def get_data(self,acquisition,normalizer,data_length=200):
        data=acquisition.get_data(data_length)
        if data.shape[0]!=data_length:
            return None
        data=data[:,[0,1]]
        data=normalizer.transform(data.reshape(1,-1))
        data=data.reshape(1,2,200)
        return data
    
    def predict(self,model,data):
        with torch.no_grad():
            pred=model.forward_run(data)
        return pred.item()
    
    def round_nearest(self,x,a=0.01):
        return np.round(np.round(x/a)*a,2)
    
    def kalman(self,state,measurement,process_var=0.02**2,measurement_var=0.05**2):
        estimate=[[],[]]
        state[0],state[1]=state[0]+0,state[1]+process_var
        estimate[0],estimate[1]=(state[1]*measurement+measurement_var*state[0])/(state[1]+measurement_var),(state[1]*measurement_var)/(state[1]+measurement_var)
        state=estimate

        return state
    
    def reset_buffer(self):
        self.filtered_buffer=[]
        return None

    def output(self,model,normalizer,acquisition,Activity_pred):
        data=self.get_data(acquisition,normalizer)
        if data is None:
            return 0
        pred=self.predict(model,data)
        self.buffer.append(pred)
        if Activity_pred==self.walking_index:
            #return pred
            if len(self.filtered_buffer)==0:
                self.filtered_buffer.append(pred)
                self.state=[pred,0.1**2]
                return self.round_nearest(pred)
            else:
                self.state=self.kalman(self.state,pred)
                self.filtered_buffer.append(self.round_nearest(self.state[0]))
                return self.round_nearest(self.state[0])
        else:
            self.reset_buffer()
            return 0
              
        
        



