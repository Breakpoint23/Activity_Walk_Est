import torch
import numpy as np
import matplotlib.pyplot as plt
import pylsl
import sys,os
import pickle

from pylsl import StreamInfo,StreamOutlet

from ActWalkEst.SpeedEst.prediction import Prediction as Walking_Prediction
from ActWalkEst.ActivityRec.prediction import prediction as Activity_Prediction

from ActWalkEst.UI.ui1 import Ui_MainWindow

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog

class App():
    def __init__(self):
        self.app=QApplication(sys.argv)
        self.ui=Ui_MainWindow()
        self.MainWindow=QMainWindow()
        self.ui.setupUi(self.MainWindow)
        self.standing=self.ui.label
        self.squatting=self.ui.label_3
        self.walking=self.ui.label_5
        self.running=self.ui.label_7
        self.label_list=[self.running,self.walking,self.standing,self.squatting]
        self.act_info=StreamInfo('Activity','Marker',1,0,'float32')
        self.act_outlet=StreamOutlet(self.act_info)
        self.speed_info=StreamInfo('Speed Estimation','Marker',1,0,'float32')
        self.speed_out=StreamOutlet(self.speed_info)
        for label in self.label_list:
            label.setStyleSheet("background-color: red")

        self.act_pred=Activity_Prediction(self.ui.horizontalSlider)
        self.walking_pred=Walking_Prediction()
        self.act_buffer=[2]
        self.walk_buffer=[]
        self.first_flag=True
        self.timer=QtCore.QTimer()
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.main_pred)
        self.timer.start()

    def act_out(self):

        self.act_pred.mean_data(self.ui.pushButton.isChecked())
        pred_max,pred_prob,data=self.act_pred.predict()
        index=np.where(pred_prob>0)[0]
        if index.size==0:
            index=self.act_buffer[-1]
            self.act_buffer.append(index)
            return index        
        else:
            if index[0]==2:
                index[0]=1
            elif index[0]==3:
                index[0]=2
            elif index[0]==4:
                index[0]=3
                pass

        for i in range(4):
            if i==index[0]:
                self.label_list[i].setStyleSheet("background-color: green")
            else:
                self.label_list[i].setStyleSheet("background-color: red")

        if index.size==0:
            self.act_buffer.append(np.NAN)
        else:
            self.act_buffer.append(index[0])

        return index[0]
    
    def main_pred(self):
        if self.first_flag:
            data=self.act_pred.acquisition.get_data(200)
            if data.shape[0]<200:
                return None
            self.first_flag=False
        act_index=self.act_out()
        self.act_outlet.push_sample([act_index])
        speed=self.walking_pred.output(self.walking_pred.model,self.walking_pred.normalizer,self.walking_pred.acquisition,act_index)
        self.walk_buffer.append(speed)
        self.speed_out.push_sample([speed])
        self.ui.figure1.clear()
        ax=self.ui.figure1.add_subplot(111)
        ax.plot(self.walk_buffer[-15:])
        ax.set_ylim(0,2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Walking Speed (m/s)')
        self.ui.canvas1.draw()

def start():
    UI=App()
    UI.MainWindow.show()
    UI.app.exec_()
if __name__ == '__main__':
    start()
        
