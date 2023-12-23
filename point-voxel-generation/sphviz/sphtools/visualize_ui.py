# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'visualize.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1386, 794)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setObjectName("widget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.widget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.framedelay = QtWidgets.QSlider(self.widget)
        self.framedelay.setMaximum(300)
        self.framedelay.setOrientation(QtCore.Qt.Horizontal)
        self.framedelay.setObjectName("framedelay")
        self.gridLayout_2.addWidget(self.framedelay, 0, 7, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setObjectName("label_3")
        self.gridLayout_2.addWidget(self.label_3, 0, 8, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 0, 6, 1, 1)
        self.playpause = QtWidgets.QPushButton(self.widget)
        self.playpause.setShortcut("")
        self.playpause.setObjectName("playpause")
        self.gridLayout_2.addWidget(self.playpause, 0, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 1, 1, 1)
        self.pointsize = QtWidgets.QSlider(self.widget)
        self.pointsize.setMinimum(1)
        self.pointsize.setMaximum(40)
        self.pointsize.setOrientation(QtCore.Qt.Horizontal)
        self.pointsize.setObjectName("pointsize")
        self.gridLayout_2.addWidget(self.pointsize, 0, 9, 1, 1)
        self.framenr = QtWidgets.QSlider(self.widget)
        self.framenr.setOrientation(QtCore.Qt.Horizontal)
        self.framenr.setObjectName("framenr")
        self.gridLayout_2.addWidget(self.framenr, 0, 2, 1, 1)
        self.verticalLayout.addWidget(self.widget)
        self.widget_2 = QtWidgets.QWidget(self.centralwidget)
        self.widget_2.setObjectName("widget_2")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.widget_2)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.leftBox = QtWidgets.QGroupBox(self.widget_2)
        self.leftBox.setObjectName("leftBox")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.leftBox)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.left_scalar = QtWidgets.QComboBox(self.leftBox)
        self.left_scalar.setObjectName("left_scalar")
        self.horizontalLayout_3.addWidget(self.left_scalar)
        self.left_cmap = QtWidgets.QComboBox(self.leftBox)
        self.left_cmap.setObjectName("left_cmap")
        self.horizontalLayout_3.addWidget(self.left_cmap)
        self.left_opacity = QtWidgets.QComboBox(self.leftBox)
        self.left_opacity.setObjectName("left_opacity")
        self.horizontalLayout_3.addWidget(self.left_opacity)
        self.horizontalLayout_5.addWidget(self.leftBox)
        self.rightBox = QtWidgets.QGroupBox(self.widget_2)
        self.rightBox.setObjectName("rightBox")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.rightBox)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.right_scalar = QtWidgets.QComboBox(self.rightBox)
        self.right_scalar.setObjectName("right_scalar")
        self.horizontalLayout_4.addWidget(self.right_scalar)
        self.right_cmap = QtWidgets.QComboBox(self.rightBox)
        self.right_cmap.setObjectName("right_cmap")
        self.horizontalLayout_4.addWidget(self.right_cmap)
        self.right_opacity = QtWidgets.QComboBox(self.rightBox)
        self.right_opacity.setObjectName("right_opacity")
        self.horizontalLayout_4.addWidget(self.right_opacity)
        self.horizontalLayout_5.addWidget(self.rightBox)
        self.verticalLayout.addWidget(self.widget_2)
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout.addWidget(self.frame)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1386, 25))
        self.menubar.setObjectName("menubar")
        self.menuTest = QtWidgets.QMenu(self.menubar)
        self.menuTest.setObjectName("menuTest")
        self.menuClip = QtWidgets.QMenu(self.menubar)
        self.menuClip.setObjectName("menuClip")
        self.menuUtils = QtWidgets.QMenu(self.menubar)
        self.menuUtils.setObjectName("menuUtils")
        self.menuGlpyths = QtWidgets.QMenu(self.menubar)
        self.menuGlpyths.setObjectName("menuGlpyths")
        self.menuOrbit = QtWidgets.QMenu(self.menubar)
        self.menuOrbit.setObjectName("menuOrbit")
        MainWindow.setMenuBar(self.menubar)
        self.actionLoad_Directory = QtWidgets.QAction(MainWindow)
        self.actionLoad_Directory.setObjectName("actionLoad_Directory")
        self.actionCreate_Movie = QtWidgets.QAction(MainWindow)
        self.actionCreate_Movie.setObjectName("actionCreate_Movie")
        self.actionToggle_PNG_Screenshots = QtWidgets.QAction(MainWindow)
        self.actionToggle_PNG_Screenshots.setObjectName("actionToggle_PNG_Screenshots")
        self.actionPrint_Camera_Position = QtWidgets.QAction(MainWindow)
        self.actionPrint_Camera_Position.setObjectName("actionPrint_Camera_Position")
        self.actionToggle_Glyphs = QtWidgets.QAction(MainWindow)
        self.actionToggle_Glyphs.setObjectName("actionToggle_Glyphs")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionLoad_Second_Directory = QtWidgets.QAction(MainWindow)
        self.actionLoad_Second_Directory.setObjectName("actionLoad_Second_Directory")
        self.actionToggle_Scale_Bar = QtWidgets.QAction(MainWindow)
        self.actionToggle_Scale_Bar.setObjectName("actionToggle_Scale_Bar")
        self.actionToggle_Stimuli_Points = QtWidgets.QAction(MainWindow)
        self.actionToggle_Stimuli_Points.setObjectName("actionToggle_Stimuli_Points")
        self.actionToggle_Plane = QtWidgets.QAction(MainWindow)
        self.actionToggle_Plane.setObjectName("actionToggle_Plane")
        self.actionToggle_Clipping = QtWidgets.QAction(MainWindow)
        self.actionToggle_Clipping.setObjectName("actionToggle_Clipping")
        self.actionGlyphMagnitude = QtWidgets.QAction(MainWindow)
        self.actionGlyphMagnitude.setObjectName("actionGlyphMagnitude")
        self.actionGlyphXZ = QtWidgets.QAction(MainWindow)
        self.actionGlyphXZ.setObjectName("actionGlyphXZ")
        self.actionGlyphXY = QtWidgets.QAction(MainWindow)
        self.actionGlyphXY.setObjectName("actionGlyphXY")
        self.actionGlyphYZ = QtWidgets.QAction(MainWindow)
        self.actionGlyphYZ.setObjectName("actionGlyphYZ")
        self.actionOrbit = QtWidgets.QAction(MainWindow)
        self.actionOrbit.setObjectName("actionOrbit")
        self.actionOrbit_Video = QtWidgets.QAction(MainWindow)
        self.actionOrbit_Video.setObjectName("actionOrbit_Video")
        self.menuTest.addAction(self.actionLoad_Directory)
        self.menuTest.addAction(self.actionLoad_Second_Directory)
        self.menuTest.addSeparator()
        self.menuTest.addAction(self.actionPrint_Camera_Position)
        self.menuTest.addSeparator()
        self.menuTest.addAction(self.actionCreate_Movie)
        self.menuTest.addAction(self.actionToggle_PNG_Screenshots)
        self.menuTest.addSeparator()
        self.menuTest.addAction(self.actionExit)
        self.menuClip.addAction(self.actionToggle_Clipping)
        self.menuClip.addAction(self.actionToggle_Plane)
        self.menuUtils.addAction(self.actionToggle_Scale_Bar)
        self.menuUtils.addAction(self.actionToggle_Stimuli_Points)
        self.menuGlpyths.addAction(self.actionToggle_Glyphs)
        self.menuGlpyths.addSeparator()
        self.menuGlpyths.addAction(self.actionGlyphMagnitude)
        self.menuGlpyths.addAction(self.actionGlyphXZ)
        self.menuGlpyths.addAction(self.actionGlyphXY)
        self.menuGlpyths.addAction(self.actionGlyphYZ)
        self.menuOrbit.addAction(self.actionOrbit)
        self.menuOrbit.addAction(self.actionOrbit_Video)
        self.menubar.addAction(self.menuTest.menuAction())
        self.menubar.addAction(self.menuUtils.menuAction())
        self.menubar.addAction(self.menuClip.menuAction())
        self.menubar.addAction(self.menuGlpyths.menuAction())
        self.menubar.addAction(self.menuOrbit.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "SPH Viewer"))
        self.label_3.setText(_translate("MainWindow", "Point Size"))
        self.label_2.setText(_translate("MainWindow", "Speed"))
        self.playpause.setText(_translate("MainWindow", "Pause"))
        self.label.setText(_translate("MainWindow", "Frame"))
        self.leftBox.setTitle(_translate("MainWindow", "Left"))
        self.rightBox.setTitle(_translate("MainWindow", "Right"))
        self.menuTest.setTitle(_translate("MainWindow", "File"))
        self.menuClip.setTitle(_translate("MainWindow", "Clip"))
        self.menuUtils.setTitle(_translate("MainWindow", "Utils"))
        self.menuGlpyths.setTitle(_translate("MainWindow", "Glpyths"))
        self.menuOrbit.setTitle(_translate("MainWindow", "Orbit"))
        self.actionLoad_Directory.setText(_translate("MainWindow", "Load Directory"))
        self.actionLoad_Directory.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.actionCreate_Movie.setText(_translate("MainWindow", "Create Movie"))
        self.actionCreate_Movie.setShortcut(_translate("MainWindow", "Ctrl+M"))
        self.actionToggle_PNG_Screenshots.setText(_translate("MainWindow", "Toggle PNG Screenshots"))
        self.actionToggle_PNG_Screenshots.setShortcut(_translate("MainWindow", "Ctrl+P"))
        self.actionPrint_Camera_Position.setText(_translate("MainWindow", "Print Camera Position"))
        self.actionPrint_Camera_Position.setShortcut(_translate("MainWindow", "Ctrl+J"))
        self.actionToggle_Glyphs.setText(_translate("MainWindow", "Toggle Glyphs"))
        self.actionToggle_Glyphs.setShortcut(_translate("MainWindow", "Ctrl+G"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionExit.setShortcut(_translate("MainWindow", "Ctrl+Q"))
        self.actionLoad_Second_Directory.setText(_translate("MainWindow", "Load Second Directory"))
        self.actionLoad_Second_Directory.setShortcut(_translate("MainWindow", "Ctrl+L"))
        self.actionToggle_Scale_Bar.setText(_translate("MainWindow", "Toggle Scale Bar"))
        self.actionToggle_Scale_Bar.setShortcut(_translate("MainWindow", "Ctrl+B"))
        self.actionToggle_Stimuli_Points.setText(_translate("MainWindow", "Toggle Stimuli Points"))
        self.actionToggle_Stimuli_Points.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionToggle_Plane.setText(_translate("MainWindow", "Toggle Plane"))
        self.actionToggle_Plane.setShortcut(_translate("MainWindow", "Ctrl+K"))
        self.actionToggle_Clipping.setText(_translate("MainWindow", "Toggle Clipping"))
        self.actionToggle_Clipping.setShortcut(_translate("MainWindow", "Ctrl+C"))
        self.actionGlyphMagnitude.setText(_translate("MainWindow", "Color = Magnitude"))
        self.actionGlyphXZ.setText(_translate("MainWindow", "Color = XZ angle"))
        self.actionGlyphXY.setText(_translate("MainWindow", "Color = XY angle"))
        self.actionGlyphYZ.setText(_translate("MainWindow", "Color = YZ angle"))
        self.actionOrbit.setText(_translate("MainWindow", "Orbit"))
        self.actionOrbit_Video.setText(_translate("MainWindow", "Orbit Video"))
