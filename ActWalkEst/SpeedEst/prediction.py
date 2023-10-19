import torch
import numpy as np
import matplotlib.pyplot as plt
import pylsl
import sys,os
import pickle

from importlib import resources

from ActWalkEst.SpeedEst.model import CNN as SA1
from ActWalkEst.SpeedEst.model2 import CNN as CNN1
from ActWalkEst.SpeedEst.model3 import CNN as SA2
from ActWalkEst.SpeedEst.model4 import CNN as CNN2
from ActWalkEst.SpeedEst.model5 import CNN as CNN3

from ActWalkEst.utils.lsl_imu import DataInlet,SetupStreams

class Prediction():
    def __init__(self,walking_index=1):
        self.buffer=[]
        self.filtered_buffer=[]
        self.walking_index=walking_index
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.MODEL_STRING=""" Select the Walking Speed Estimation model\n
        press, \n
        1 for Self Attention with vertical and lateral acc (1 s)\n
        2 for CNN with only vertical acc (1s) \n
        3 for Self Attention with only vertical acc (1s)\n
        4 for CNN with vertical and lateral acc (1s) \n
        5 for CNN with vertical and lateral acc (0.5 s)

        
        """
        #self.model_map={1:SA1,2:CNN1,3:SA2,4:CNN2,5:CNN3}
        self.acc_ind={"1":0,"2":[0,1]}
        self.acc_len=200

        self.desired_model=int(input(self.MODEL_STRING))

        if self.desired_model not in [1,2,3,4,5]:
            print("Could not understand the desired model \n switching to default Self Attention model")
            self.desired_model=int(1)

        if self.desired_model in [1,4,5]:
            self.aix=self.acc_ind["2"]
            self.ch_ln=2
        else:
            self.aix=0
            self.ch_ln=1

        if self.desired_model==5:
            self.acc_len=100
        else:
            self.acc_len=200
        
        """
        this is for fixed model running

        with resources.path('ActWalkEst.resources','walking_model1.h5') as p:
            self.MODEL_PATH=str(p)
        self.model=self.load_model()
        with resources.path('ActWalkEst.resources','walking_norm.pickle') as p:
            self.NORMALIZER_PATH=str(p)
        """
        self.load_model(self.desired_model)
        self.normalizer=self.load_normalizer(self.desired_model)
        self.acquisition=SetupStreams()
        return None
    


    def load_model(self,model_num):
        
        if model_num==1:
            model=SA1()

        elif model_num==3:
            model=SA2()
            
        elif model_num==2:
            model=CNN1()

        elif model_num==4:
            model=CNN2()

        elif model_num==5:
            model=CNN3()

        #model = CNN(input_features=1,input_length=200,num_classes=1)
        with resources.path('ActWalkEst.resources',f'walking_model{model_num}.h5') as p:
            self.MODEL_PATH=str(p)
        model.load_state_dict(torch.load(self.MODEL_PATH))
        model.eval()
        return model
    
    def load_normalizer(self,model_num):

        if model_num in [1,4]:
            norm_num=1

        elif model_num in [2,3]:
            norm_num=2

        else:
            norm_num=3
        
        with resources.path('ActWalkEst.resources',f'walking_norm{norm_num}.pickle') as p:
            self.NORMALIZER_PATH=str(p)

        normalizer=pickle.load(open(self.NORMALIZER_PATH,'rb'))
        return normalizer
    
    def get_data(self,acquisition,normalizer,data_length=200):
        data=acquisition.get_data(self.acc_len)
        if data.shape[0]!=self.acc_len:
            return None
        data=data[:,self.aix]
        data=normalizer.transform(data.reshape(1,-1))
        data=data.reshape(1,self.ch_ln,self.acc_len)
        return data
    
    def predict(self,model,data):
        data=data.from_numpy(data).float()
        data=data.to(self.device)
        with torch.no_grad():
            pred=model.forward(data)
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
              
        
        



