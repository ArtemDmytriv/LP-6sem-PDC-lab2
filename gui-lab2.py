import sys

from PyQt5.QtCore import QLine, Qt
from PyQt5.QtWidgets import *
from window import *

from sympy import *

class PyMatrix(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.initList()
        self.initTable()

        self.ui.spinBox.valueChanged.connect(lambda: self.onSizeChanged(self.ui.spinBox.value()))

        self.show()

    def initList(self):
        self.ui.matrixList.addItem("matrix A")
        self.ui.matrixList.addItem("matrix A1")
        self.ui.matrixList.addItem("vector b1")
        self.ui.matrixList.addItem("vector c1")
        self.ui.matrixList.addItem("matrix A2")
        self.ui.matrixList.addItem("matrix B2")
        self.ui.matrixList.setCurrentItem(self.ui.matrixList.itemAt(0, 0))

        self.ui.matrixList.currentItemChanged.connect(lambda: self.onChangedMatrix(self.ui.matrixList.currentRow()))

    def initTable(self):
        self.ui.matrixTable.setColumnCount(self.ui.spinBox.value())
        self.ui.matrixTable.setRowCount(self.ui.spinBox.value())
        self.ui.matrixTable.resizeColumnsToContents()

    def onChangedMatrix(self, index):
        if (index == 2 or index == 3):
            self.ui.matrixTable.setColumnCount(1)
        else:
            self.ui.matrixTable.setColumnCount(self.ui.spinBox.value())
        self.ui.matrixTable.setRowCount(self.ui.spinBox.value())
        self.ui.matrixTable.resizeColumnsToContents()
        print("mat no=", index)

    def onSizeChanged(self, newsize):
        self.onChangedMatrix(self.ui.matrixList.currentRow())
        print("size=",newsize)


def main():
    app = QApplication(sys.argv)
    view = PyMatrix()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

