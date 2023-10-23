import torch
import numpy as np
import matplotlib.pyplot as plt
import pylsl
import sys,os
import pickle

from importlib import resources

# Importing different models for walking speed estimation
from ActWalkEst.SpeedEst.model import CNN as SA1
from ActWalkEst.SpeedEst.model2 import CNN as CNN1
from ActWalkEst.SpeedEst.model3 import CNN as SA2
from ActWalkEst.SpeedEst.model4 import CNN as CNN2
from ActWalkEst.SpeedEst.model5 import CNN as CNN3

# Importing objects to get data from polar stream
from ActWalkEst.utils.lsl_imu import DataInlet,SetupStreams

class Prediction():
    def __init__(self,walking_index=1):

    
        self.buffer=[]  # buffer for raw output of the model
        self.filtered_buffer=[] # buffer after passing the raw output through kalman filter
        self.walking_index=walking_index # for activity recognition 

        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.MODEL_STRING=""" Select the Walking Speed Estimation model\n
        press, \n
        1 for Self Attention with vertical and lateral acc (1 s)\n
        2 for CNN with only vertical acc (1s) \n
        3 for Self Attention with only vertical acc (1s)\n
        4 for CNN with vertical and lateral acc (1s) \n
        5 for CNN with vertical and lateral acc (0.5 s)

        
        """


        self.acc_ind={"1":0,"2":[0,1]} # different acceleration axis used by different models

        self.acc_len=200 # initializing window lenght

        self.desired_model=int(input(self.MODEL_STRING)) # getting user input for which model they want to use

        if self.desired_model not in [1,2,3,4,5]:
            print("Could not understand the desired model \n switching to default Self Attention model")
            self.desired_model=int(1)

        # Based on model fixing the acceleration used (vertical or (vertical and lateral)) 

        if self.desired_model in [1,4,5]:
            self.aix=self.acc_ind["2"]
            self.ch_ln=2
        else:
            self.aix=0
            self.ch_ln=1

        # fixing window size
            
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
        # model and normalizer loading

        self.model=self.load_model(self.desired_model)
        self.normalizer=self.load_normalizer(self.desired_model)

        # Initiating instance for object responsible for getting data from polar stream
        self.acquisition=SetupStreams()

        return None
    


    def load_model(self,model_num):
        
        """
        This function returns a walking speed estimation model with loaded weights based on user input
        
        """

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

        
        with resources.path('ActWalkEst.resources',f'walking_model{model_num}.h5') as p:
            self.MODEL_PATH=str(p)

        model.load_state_dict(torch.load(self.MODEL_PATH))

        model.eval() # putting the model in evaluation mode

        return model
    
    def load_normalizer(self,model_num):

        """
        This function return normalizer from sklearn library prefitted on the training data according to the model type 
        
        """

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

        """
        This function pulls data from the polar stream and uses the normalizer to scale the input data
        """

        data=acquisition.get_data(self.acc_len) # get data from polar

        if data.shape[0]!=self.acc_len:
            return None
        
        data=data[:,self.aix] # take only relavant axis data (vertical or vertical and lateral)
        data=normalizer.transform(data.reshape(1,-1)) # normalize the data
        data=data.reshape(1,self.ch_ln,self.acc_len) # reshape the data in to (batch_size,channels,window_length)

        return data
    
    def predict(self,model,data):
        data=torch.from_numpy(data).float()
        with torch.no_grad():
            pred=model.forward(data)
            pred=model.forward(data)
        return pred.item()
    
    def round_nearest(self,x,a=0.01):
        """
        returns x in multiple of a 
        """
        return np.round(np.round(x/a)*a,2)
    
    def kalman(self,state,measurement,process_var=0.02**2,measurement_var=0.05**2):

        """
        Implementation of one dimensional kalman filter 
        """
        
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
              
        
        



