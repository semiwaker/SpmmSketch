import numpy as np
from typing import List, Tuple
from dataclasses import dataclass, field
import time
import math

@dataclass(init=True)
class CSR:
    rowPos: List[int]
    colId: List[int]
    data: List[float]

@dataclass(init=True)
class CSC:
    colPos: List[int]
    rowId: List[int]
    data: List[float]

def COO2CSR(mat:List[Tuple[int, int, float]]):
    @dataclass(init=True, order=True)
    class Unit:
        x:int
        y:int
        c:float=field(compare=False)
    m = [Unit(x=x,y=y,c=c) for x,y,c in mat]
    m.sort()
    rowPos = []
    colId = []
    data = []
    lastX = 0
    for i in m:
        x, y, c = i.x, i.y, i.c
        while x != lastX:
            rowPos.append(len(colId))
            lastX += 1
        colId.append(y)
        data.append(c)
    return CSR(rowPos, colId, data)
        

def COO2CSC(mat):
    @dataclass(init=True, order=True)
    class Unit:
        y:int
        x:int
        c:float=field(compare=False)
    m = [Unit(x=x,y=y,c=c) for x,y,c in mat]
    m.sort()
    colPos = []
    rowId = []
    data = []
    lastY = 0
    for i in m:
        x, y, c = i.x, i.y, i.c
        while y != lastY:
            colPos.append(len(rowId))
            lastY += 1
        rowId.append(x)
        data.append(c)
    return CSC(colPos, rowId, data) 

def readMatrix(file):
    '''Return COO: List[(x,y,c)]'''
    with open(file, "r") as fin:
        data = fin.readlines()
    mat = []
    for s in data[1:]:
        x, y, c = s.strip().split(' ')
        mat.append((int(x), int(y), float(c)))
    return mat

def printMatrix(file, mat):
    '''Read COO: List[(x,y,c)]'''
    with open(file, "w") as fout:
        fout.writelines(
            [str(len(mat))+"\n"] +
            ["%d %d %.6f\n"%(x,y,c) for (x,y,c) in mat]
        )

def timer(f):
    def timedfunc(*args, **kwargs):
        t0 = time.process_time()
        r = f(*args, **kwargs)
        return ((time.process_time() - t0), r)
    return timedfunc

def COOMinus(a,b):
    aMap = {(x,y):c for x,y,c in a}
    bMap = {(x,y):c for x,y,c in b}
    aSet = {(x,y) for x,y,c in a}
    bSet = {(x,y) for x,y,c in b}
    result = []
    for x,y in aSet.intersection(bSet):
        result.append((x,y,aMap[(x,y)]-bMap[(x,y)]))
    for x,y in aSet.difference(bSet):
        result.append((x,y,aMap[(x,y)]))
    for x,y in bSet.difference(aSet):
        result.append((x,y,bMap[(x,y)]))
    return result

def COORMSE(a, b):
    err = 0.0
    diff = COOMinus(a,b)
    for x,y,c in diff:
        err += c**2 
    return math.sqrt(err / len(diff))
    
def segmentedRMSE(a, b, aMin, aMax, seg):
    nSeg = math.ceil((aMax - aMin)/seg)

    sums = [0 for i in range(nSeg)]
    cnts = [0 for i in range(nSeg)]
    labels = ["%d"%(aMin+i*seg) for i in range(nSeg+1)]

    bMap = {(x,y):c for x,y,c in b}
    for x,y,c in a:
        s = math.floor((c-aMin)/seg)
        cnts[s] += 1
        if (x,y) in bMap:
            sums[s] += (c - bMap[(x,y)] )**2
        else:
            sums[s] += c**2
    return labels, [math.sqrt(s/c) if c!=0 else 0.0 for s,c in zip(sums, cnts)]

def segmentedAcc(a, b, aMin, aMax, seg, th):
    nSeg = math.ceil((aMax - aMin)/seg)

    hits = [0 for i in range(nSeg)]
    cnts = [0 for i in range(nSeg)]
    labels = ["%d"%(aMin+i*seg) for i in range(nSeg+1)]

    bMap = {(x,y):c for x,y,c in b}
    for x,y,c in a:
        s = math.floor((c-aMin)/seg)
        cnts[s] += 1
        if (x,y) in bMap:
            z = bMap[(x,y)]
        else:
            z = 0.0
        if abs(z-c)/abs(c) <= th:
            hits[s] += 1

    return labels, [s/c*100 if c!=0 else 100 for s,c in zip(hits, cnts)]

        

       
