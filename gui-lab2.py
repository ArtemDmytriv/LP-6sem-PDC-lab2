import sys
import subprocess

from PyQt5.QtCore import QLine, Qt
from PyQt5.QtWidgets import *

from window import *
from check_result import checkResult

from collections import OrderedDict
from sympy import *

default_N = 4

class SymMatrices:
    def __init__(self, n = default_N):
        self.N = n
        self.matrices = OrderedDict([ ("A" , zeros(self.N, self.N)),
                          ("A1", zeros(self.N, self.N)),
                          ("b1", zeros(self.N, 1)),
                          ("c1", zeros(self.N, 1)),
                          ("A2", zeros(self.N, self.N)),
                          ("B2", zeros(self.N, self.N))])

    def readToFile(self):
        f = open("matrices.txt", 'w')
        f.write(str(self.N) + '\n')
        for key, item in self.matrices.items():
            for i in range(item.rows):
                for j in range(item.cols):
                    f.write(str(item[i, j]) + " ")
                f.write('\n')
        f.close()

    def set_N(self, n):
        while (self.N != n):
            if (self.N < n):
                #print('inc N')
                for key in self.matrices:
                    #pprint(self.matrices[key])
                    self.matrices[key] = self.matrices[key].row_insert(self.N, zeros(1, self.matrices[key].shape[1]))
                    if (key != "b1" and key != 'c1'):
                        self.matrices[key] = self.matrices[key].col_insert(self.N, zeros(self.matrices[key].shape[0], 1))
                    #pprint(self.matrices[key])
                self.N += 1
            else:
                for key in self.matrices:
                    self.matrices[key].row_del(self.N - 1)
                    if (key != "b1" and key != 'c1'):
                        self.matrices[key].col_del(self.N - 1)
                self.N -= 1

class PyMatrix(QMainWindow):
    def __init__(self):
        super().__init__()
        self.mat = SymMatrices()
        self.makePopulate = False

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.initList()
        self.initTable()

        self.ui.spinBox.valueChanged.connect(lambda: self.onChangedMatrix(self.ui.matrixList.currentRow()))
        self.ui.btnStartAlgo.clicked.connect(lambda: self.onClickedStart())
        self.ui.btnGenerateMatrix.clicked.connect(lambda: self.onRandomStart())
        self.ui.btnAllGenerate.clicked.connect(lambda: self.onAllRandomStart())
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
        self.populateTable()
        self.ui.matrixTable.resizeColumnsToContents()
        self.ui.matrixTable.cellChanged.connect(lambda: self.onUpdatedCell())

    def populateTable(self):
        self.makePopulate = True
        key = self.ui.matrixList.currentItem().text().split()[1]
        for row in range(self.ui.matrixTable.rowCount()):
            for col in range(self.ui.matrixTable.columnCount()):
                self.ui.matrixTable.setItem(row, col,
                    QTableWidgetItem(str(self.mat.matrices[key][row, col])))
        self.makePopulate = False

    def onUpdatedCell(self):
        if (self.makePopulate == False):
            print('was updated cell')
            key = self.ui.matrixList.currentItem().text().split()[1]
            for row in range(self.ui.matrixTable.rowCount()):
                for col in range(self.ui.matrixTable.columnCount()):
                    self.mat.matrices[key][row, col] = self.ui.matrixTable.item(row, col).text()

    def onChangedMatrix(self, index):
        if (index == 2 or index == 3):
            self.ui.matrixTable.setColumnCount(1)
        else:
            self.ui.matrixTable.setColumnCount(self.ui.spinBox.value())
        self.ui.matrixTable.setRowCount(self.ui.spinBox.value())
        self.mat.set_N(self.ui.spinBox.value())
        self.populateTable()
        self.ui.matrixTable.resizeColumnsToContents()

    def onRandomStart(self):
        key = self.ui.matrixList.currentItem().text().split()[1]
        for row in range(self.ui.matrixTable.rowCount()):
            for col in range(self.ui.matrixTable.columnCount()):
                self.mat.matrices[key] = randMatrix(self.ui.matrixTable.rowCount(), self.ui.matrixTable.columnCount(), 0.0, 100.0)
        self.populateTable()

    def onAllRandomStart(self):
        for key, val in self.mat.matrices.items():
            if (key == 'b1' or key == 'c1'):
                self.mat.matrices[key] = randMatrix(self.ui.spinBox.value(), 1, 0.0, 100.0)
            else:
                self.mat.matrices[key] = randMatrix(self.ui.spinBox.value(), self.ui.spinBox.value(), 0.0, 100.0)
        self.populateTable()
        return

    def onClickedStart(self):
        self.mat.readToFile()
        inp = open('matrices.txt', 'r')
        out = subprocess.check_output(['nice', '-n', '-20','mpirun', '-n', '3', './lab2', '-f'], stdin=inp)
        out = out.splitlines()

        res = []
        time_mpi = ""
        time_one_thread = ""
        for i in range(len(out)):
            if out[i].find(b"Time taken by program") >= 0:
                time_mpi = out[i].decode("utf-8").split(":")[1].split(" ")[1]
            if out[i].find(b"Result") >= 0:
                for k in range(1, self.mat.N + 1):
                    res.append(list(filter(lambda x: x != '', out[i+k].decode("utf-8").split(' '))))
            if out[i].find(b"Time taken by one thread program") >= 0:
                time_one_thread = out[i].decode("utf-8").split(":")[1].split(" ")[1]

        print("MPI time: " + time_mpi)
        print("OneThread time: " + time_one_thread)
        #pprint(Matrix(res))
        #print("Python recheck result=")
        recheck = checkResult()
        #pprint(recheck)

        dlg = QDialog(self)
        dlg.setGeometry(100, 100, 500, 500)

        mpi_res = QTableWidget()
        mpi_res.setRowCount(len(res))
        mpi_res.setColumnCount(len(res[0]))

        for row in range(mpi_res.rowCount()):
            for col in range(mpi_res.columnCount()):
                mpi_res.setItem(row, col, QTableWidgetItem(res[row][col]))
        mpi_res.resizeColumnsToContents()

        py_res = QTableWidget()
        py_res.setRowCount(recheck.shape[0])
        py_res.setColumnCount(recheck.shape[1])

        for row in range(py_res.rowCount()):
            for col in range(py_res.columnCount()):
                py_res.setItem(row, col, QTableWidgetItem(str(recheck[row, col])))

        py_res.resizeColumnsToContents()

        layout = QGridLayout(dlg)
        layout.addWidget(QLabel("MPI res = " + time_mpi + " us :: 1-thread res = " + time_one_thread + " us"), 0, 0)
        layout.addWidget(QLabel("Python Result"), 0, 1)
        layout.addWidget(mpi_res, 1, 0)
        layout.addWidget(py_res, 1, 1)

        dlg.setWindowTitle("Results")
        dlg.exec_()



def main():
    app = QApplication(sys.argv)
    view = PyMatrix()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

