import numpy as np
from typing import Optional, Tuple
from spmmSketch.util import *

dataSet = "gaussian"
outputFile = "nonzero.txt"

def prepareData():
    a = readMatrix(f"data/{dataSet}A.txt")
    b = readMatrix(f"data/{dataSet}B.txt")
    c = readMatrix(f"data/{dataSet}C.txt")

    aCSR = COO2CSR(a)
    # aCSC = COO2CSC(a)
    # bCSR = COO2CSR(b)
    bCSC = COO2CSC(b)
    return (aCSR, bCSC, c)

class SketchTable:
    def __init__(self, N, M) -> None:
        self.N = N
        self.M = M
        self.table = np.zeros(N, np.int8)

    def idHash(self, x, i):
        return hash((x,i))%self.N
    
    def sign(self,x,i) -> int:
        return 1 if hash((i,x))%2==1 else -1

    def insert(self, x) -> None:
        for i in range(self.M):
            self.table[self.idHash(x, i)] += self.sign(x,i)
    
    # def query(self, x):
        # return np.min([self.table[self.idHash(x,i)] for i in range(self.M)])

def buildTable(pos, id, N, M) -> List[Tuple[int, SketchTable]]:
    last = 0
    result = []
    for i, p in enumerate(pos):
        if last == p:
            continue
        table = SketchTable(N, M)
        for x in id[last: p]:
            table.insert(x)
        result.append((i, table)) 
        last = p
    return result

@timer
def getNonZero(aCSR:CSR, bCSC:CSC):
    M = 1
    a = buildTable(aCSR.rowPos, aCSR.colId, 10, M)
    b = buildTable(bCSC.colPos, bCSC.rowId, 10, M)

    print(len(a), len(b))

    blockN = 10
    aMat = np.array([x[1].table for x in a])
    bMat = np.array([x[1].table for x in b]).transpose()

    result = []
    for i in range(0, len(a), blockN):
        for j in range(0, len(b), 1):
            print(i,j)
            c = np.matmul(aMat[i:i+blockN], bMat[:,j:j+blockN])
            print(c)

            v = np.argwhere(c>=M) + np.array((i,j))
            result.extend(v)
            print(len(result))
            exit(1)
            
    result = [(a[x][0], b[y][0]) for x, y in result]

    return result

# @timer
# def getNonZero(aCSR:CSR, bCSC:CSC):
#     M = 4
#     b = buildTable(bCSC.colPos, bCSC.rowId, 100, M)
#     last = 0
#     result = []
#     for x, p in enumerate(aCSR.rowPos):
#         print(x, len(result))
#         if last == p:
#             continue
#         for y, tab in b:
#             for i in aCSR.colId[last: p]:
#                 if tab.query(i) == M:
#                     result.append((x,y))
#                     break
#         last = p
#     return result


def nonZeroEvaluate(result, c):
    a = set(result)
    b = {(x,y) for x,y,_ in c}
    z = a.intersection(b)
    accuracy = len(z) / len(a)
    recall = len(z) / len(b)
    f1 = 2 * accuracy * recall / (accuracy + recall)
    return accuracy, recall, f1

if __name__ == "__main__":
    aCSR, bCSC, c = prepareData()
    print("Start")
    t, result = getNonZero(aCSR, bCSC)
    print(t, nonZeroEvaluate(result, c))
    with open(f"out/{outputFile}", "w") as fout:
        fout.writelines("\n".join(result.map(lambda x: f"{x[0]} {x[1]}")))
