import numpy as np
from typing import Optional, Tuple
from spmmSketch.util import *

dataSet = "gaussianLarge2"
outputFile = "result1.txt"

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

    def idHash(self, x, y, i) -> int:
        return hash((x,y,i))%self.N
    
    def insert(self, x, y, c) -> Optional[Tuple[int, int, float]]:
        m = 1e30
        sel = -1
        selH = -1
        for i in range(self.M):
            h = self.idHash(x,y,i)
            if self.idTab[i][h][0] == -1:
                self.idTab[i][h][0] = x
                self.idTab[i][h][1] = y
                self.cTab[i][h] = c
                return None
            if self.idTab[i][h][0] == x and self.idTab[i][h][1] == y :
                self.cTab[i][h] += c
                return None
            if sel==-1 or abs(self.cTab[i][h]) < abs(m):
                m = self.cTab[i][h]
                sel = i
                selH = h
        ret = (self.idTab[sel][selH][0], self.idTab[sel][selH][1], self.cTab[sel][selH])
        self.idTab[sel][selH][0] = x
        self.idTab[sel][selH][1] = y
        self.cTab[sel][selH] = c
        return ret
    
    def query(self, x, y) -> Optional[float]:
        for i in range(self.M):
            h = self.idHash(x,y,i)
            if self.idTab[i][h][0] == x and self.idTab[i][h][1] == y :
                return self.cTab[i][h]
        return None
            
class SketchTable:
    def __init__(self, N, M) -> None:
        self.N = N
        self.M = M
        self.table = np.zeros(N, np.float32)

    def idHash(self, x, y, i) -> int:
        return hash((x,y,i))%self.N
    
    def sign(self,x,y,i) -> int:
        return 1 if hash((i,x,y))%2==1 else -1
    
    def insert(self, x, y, c) -> None:
        for i in range(self.M):
            h = self.idHash(x,y,i)
            self.table[h] += c * self.sign(x,y,i)
    
    def query(self, x, y):
        return np.mean([self.table[self.idHash(x,y,i)]*self.sign(x,y,i) for i in range(self.M)])

class Sketch:
    def __init__(self, hashN, sketchN) -> None:
        self.hashN = hashN
        self.sketchN = sketchN

        self.hashTable = HashTable(hashN, 4)
        self.sketchTable = SketchTable(sketchN, 2)

    def insert(self, x, y, c) -> None:
        ret = self.hashTable.insert(x, y, c)
        if ret is not None:
            x1, y1, c1 = ret
            self.sketchTable.insert(x1, y1, c1)
    
    def query(self, x, y):
        hashRet = self.hashTable.query(x,y)
        sketchRet = self.sketchTable.query(x,y)
        if hashRet is None:
            return sketchRet
        else:
            return hashRet
    
@timer
def makeSketch(aCSC:CSC, bCSR:CSR):
    sketch = Sketch(8000, 100000)
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
def collectResult(sketch:Sketch, aCSC:CSC, bCSR:CSR):
    result = {}
    lastA = 0
    lastB = 0
    for cPos, rPos in zip(aCSC.colPos, bCSR.rowPos):
        for i in range(lastA, cPos):
            for j in range(lastB, rPos):
                x = aCSC.rowId[i]
                y = bCSR.colId[j]
                c = sketch.query(x,y)
                if (x,y) in result:
                    result[(x,y)] += c
                else:
                    result[(x,y)] = c
        lastA = cPos
        lastB = rPos
    return [(x,y,c) for (x,y),c in result.items()]

# @timer
# def collectResult(sketch:Sketch, aCSC:CSC, bCSR:CSR):
#     for aRow in range(len(aCSR.rowPos)):
#         for bCol in range(len(bCSC.colPos)):
#             c = 0.0
#             aRange = (0 if aRow == 0 else aCSR.rowPos[aRow-1], aCSR.rowPos[aRow])
#             bRange = (0 if bCol == 0 else bCSC.colPos[bCol-1], bCSC.colPos[bCol])
#             i=aRange[0]
#             j=bRange[0]
#             while i<aRange[1] and j<bRange[1]:
#                 if aCSR.colId[i] > bCSC.rowId[j]:
#                     j+=1
#                 elif aCSR.colId[i] < bCSC.rowId[j]:
#                     i+=1
#                 else:
#                     c+=sketch.query(i,j)
#                     i+=1
#                     j+=1
#             if abs(c) >1e-6:
#                 result.append((aRow, bCol, c))
#     return result


if __name__ == "__main__":
    aCSR, aCSC, bCSR, bCSC, c = prepareData()
    t, sketch = makeSketch(aCSC, bCSR)
    print(t)
    t, result = collectResult(sketch, aCSC, bCSR)
    print(t)
    print(COORMSE(result, c), np.max([abs(z) for _,_,z in c]))
    printMatrix(f"out/{outputFile}", result)
