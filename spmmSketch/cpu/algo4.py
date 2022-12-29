import numpy as np
from typing import Optional, Tuple
from spmmSketch.util import *
from algo2 import Sketch

dataSet = "gaussianLarge2"
outputFile = "result4.txt"

def prepareData():
    a = readMatrix(f"data/{dataSet}A.txt")
    b = readMatrix(f"data/{dataSet}B.txt")
    c = readMatrix(f"data/{dataSet}C.txt")

    aCSR = COO2CSR(a)
    aCSC = COO2CSC(a)
    bCSR = COO2CSR(b)
    bCSC = COO2CSC(b)
    return (aCSR, aCSC, bCSR, bCSC, c)

@timer
def makeSketch(aCSC:CSC, bCSR:CSR):
    sketch = Sketch(39000, 300000)
    lastA = 0
    lastB = 0
    for cPos, rPos in zip(aCSC.colPos, bCSR.rowPos):
        for i in range(lastA, cPos):
            for j in range(lastB, rPos):
                x = aCSC.rowId[i]
                y = bCSR.colId[j]
                c = aCSC.data[i] * bCSR.data[j]
                sketch.insert(x,y,c)
        lastA = cPos
        lastB = rPos
    return sketch

@timer
def collectResult(sketch:Sketch):
    result = []
    for j in range(0, sketch.hashTable.M):
        for i in range(0, sketch.hashTable.N):
            if sketch.hashTable.idTab[j][i][0] != -1:
                result.append((sketch.hashTable.idTab[j][i][0], sketch.hashTable.idTab[j][i][1], sketch.hashTable.cTab[j][i]))
    return result

if __name__ == "__main__":
    aCSR, aCSC, bCSR, bCSC, c = prepareData()
    t, sketch = makeSketch(aCSC, bCSR)
    print(t)
    t, result = collectResult(sketch)
    print(t)
    print(COORMSE(result, c), np.max([abs(z) for _,_,z in c]))
    printMatrix(f"out/{outputFile}", result)
