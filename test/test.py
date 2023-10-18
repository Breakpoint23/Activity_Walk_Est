from ActWalkEst.UI import ui1
import sys
from PyQt5 import QtCore,QtGui,QtWidgets
from PyQt5.QtWidgets import QApplication,QMainWindow

app=QApplication(sys.argv)
main_window=QMainWindow()
ui_1=ui1.Ui_MainWindow()
ui_1.setupUi(main_window)
main_window.show()
sys.exit(app.exec_())
