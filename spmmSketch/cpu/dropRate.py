import numpy as np
from typing import Optional, Tuple
from spmmSketch.util import *
import matplotlib.pyplot as plt

dataSet = "gaussianLarge2"

thresholds = [1e-2, 1e-1, 1, 5, 10, 50, 100]

def prepareData():
    a = readMatrix(f"data/{dataSet}A.txt")
    b = readMatrix(f"data/{dataSet}B.txt")
    c = readMatrix(f"data/{dataSet}C.txt")

    aCSR = COO2CSR(a)
    aCSC = COO2CSC(a)
    bCSR = COO2CSR(b)
    bCSC = COO2CSC(b)
    return (aCSR, aCSC, bCSR, bCSC, c)

class HashTable:
    def __init__(self, N, M)->None:
        self.N = N
        self.M = M

        self.idTab = np.zeros(dtype=np.int32, shape=(M,N,2))
        self.cTab = np.zeros(dtype=np.float32, shape=(M,N))
        self.idTab.fill(-1)

        self.cnt = 0
        self.dropCnt = [0 for i in range(len(thresholds))]
        self.dropRate = [[] for i in range(len(thresholds))]

    def idHash(self, x, y, i) -> int:
        return hash((x,y,i))%self.N
    
    def insert(self, x, y, c):
        m = 1e30
        sel = -1
        selH = -1
        for i in range(self.M):
            h = self.idHash(x,y,i)
            if self.idTab[i][h][0] == -1:
                self.idTab[i][h][0] = x
                self.idTab[i][h][1] = y
                self.cTab[i][h] = c
                self.inserted()
                return
            if self.idTab[i][h][0] == x and self.idTab[i][h][1] == y :
                self.cTab[i][h] += c
                self.inserted()
                return
            if sel==-1 or abs(self.cTab[i][h]) < abs(m):
                m = self.cTab[i][h]
                sel = i
                selH = h
        self.idTab[sel][selH][0] = x
        self.idTab[sel][selH][1] = y
        self.cTab[sel][selH] = c
        self.dropped(c)
    
    def query(self, x, y) -> Optional[float]:
        for i in range(self.M):
            h = self.idHash(x,y,i)
            if self.idTab[i][h][0] == x and self.idTab[i][h][1] == y :
                return self.cTab[i][h]
        return None
    
    def inserted(self):
        self.incTime()

    def dropped(self, c):
        for i, t in enumerate(thresholds):
            if abs(c) >= t:
                self.dropCnt[i] +=1
        self.incTime()
    
    def incTime(self):
        self.cnt += 1
        if self.cnt == 1000:
            for i in range(len(thresholds)):
                self.dropRate[i].append(self.dropCnt[i]/1000)
                self.dropCnt[i] = 0
            self.cnt = 0
            
class Sketch:
    def __init__(self, hashN) -> None:
        self.hashN = hashN

        self.hashTable = HashTable(hashN, 4)

    def insert(self, x, y, c) -> None:
        ret = self.hashTable.insert(x, y, c)
    
    def query(self, x, y):
        hashRet = self.hashTable.query(x,y)
        if hashRet is None:
            return 0.0
        else:
            return hashRet
    
    def getDropRate(self):
        return self.hashTable.dropRate
    
def makeSketch(aCSC:CSC, bCSR:CSR):
    sketch = Sketch(16000)
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


if __name__ == "__main__":
    aCSR, aCSC, bCSR, bCSC, c = prepareData()
    sketch = makeSketch(aCSC, bCSR)
    dropRate = sketch.getDropRate()
    for i, t in enumerate(thresholds):
        plt.plot(list(range(1,len(dropRate[i])+1)), dropRate[i], label=f">{t}")
    plt.legend()
    plt.savefig("dropRate.png")
