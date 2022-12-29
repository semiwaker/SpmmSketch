import numpy as np
from typing import Optional, Tuple
from spmmSketch.util import *

dataSet = "gaussianLarge2"
outputFile = "result7.txt"

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

        self.threshold = 0
        self.dropCnt = 0
        self.inCnt = 0

        self.next = None

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
                self.inc()
                return
            if self.idTab[i][h][0] == x and self.idTab[i][h][1] == y :
                self.cTab[i][h] += c
                self.inc()
                return
            if sel==-1 or abs(self.cTab[i][h]) < abs(m):
                m = self.cTab[i][h]
                sel = i
                selH = h
        ret = (self.idTab[sel][selH][0], self.idTab[sel][selH][1], self.cTab[sel][selH])
        self.idTab[sel][selH][0] = x
        self.idTab[sel][selH][1] = y
        self.cTab[sel][selH] = c
        if abs(ret[2]) > self.threshold:
            self.dropCnt += 1
            self.inc()
            if self.next is not None:
                return self.next.insert(*ret)
            else:
                return ret
        else:
            self.inc()
    
    def inc(self):
        self.inCnt += 1
        if self.inCnt == 100:
            self.inCnt = 0
            if self.dropCnt > 70:
                self.threshold += 1e-3
            elif self.dropCnt < 70 and self.threshold > 0:
                self.threshold -= 1e-3
            self.dropCnt = 0
    
    def collectAll(self, result):
        for j in range(0, self.M):
            for i in range(0, self.N):
                if self.idTab[j][i][0] != -1:
                    result.append((self.idTab[j][i][0], self.idTab[j][i][1], self.cTab[j][i]))

    def dump(self, result):
        if self.next is None:
            return
        for j in range(0, self.M):
            for i in range(0, self.N):
                if self.idTab[j][i][0] != -1:
                    ret = self.next.insert(self.idTab[j][i][0], self.idTab[j][i][1], self.cTab[j][i])
                    if ret is not None:
                        result.append(ret)
        self.threshold = 0
            
class Sketch:
    def __init__(self, hashN) -> None:
        self.hashN = hashN

        self.hashTable1 = HashTable(hashN, 4)
        self.hashTable2 = HashTable(hashN, 4)
        self.hashTable3 = HashTable(hashN, 4)
        self.hashTable4 = HashTable(hashN, 4)

        self.hashTable1.next = self.hashTable2
        self.hashTable2.next = self.hashTable3
        self.hashTable3.next = self.hashTable4

        self.cnt = 0 
        self.result = []

    def insert(self, x, y, c) -> None:
        self.hashTable1.insert(x, y, c)
        
        self.cnt += 1
        if self.cnt == 100000:
            self.cnt = 0
            # self.hashTable1.dump(self.result)
            # self.hashTable2.dump(self.result)
            # self.hashTable3.dump(self.result)
    
    def query(self, x, y):
        hashRet = self.hashTable1.query(x,y)
        if hashRet is None:
            hashRet = self.hashTable2.query(x,y)
            return 0.0 if hashRet is None else hashRet
        else:
            return hashRet
    
    
@timer
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
    
@timer
def collectResult(sketch:Sketch):
    result = sketch.result
    sketch.hashTable1.collectAll(result)
    sketch.hashTable2.collectAll(result)
    sketch.hashTable3.collectAll(result)
    sketch.hashTable4.collectAll(result)
    print(len(result))

    return result

if __name__ == "__main__":
    aCSR, aCSC, bCSR, bCSC, c = prepareData()
    t, sketch = makeSketch(aCSC, bCSR)
    print(t)
    t, result = collectResult(sketch)
    print(t)
    print(COORMSE(result, c), np.max([abs(z) for _,_,z in c]))
    printMatrix(f"out/{outputFile}", result)
