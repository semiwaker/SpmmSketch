import numpy as np
from numpy.random import default_rng
import time
import math
from spmmSketch.util import readMatrix, printMatrix, timer, COO2CSC, COO2CSR

rng = default_rng()

@timer
def multiplyCOO(a, b):
    aCSC = COO2CSC(a)
    bCSR = COO2CSR(b)
    result = {}
    lastA = 0
    lastB = 0
    cnt = 0
    for cPos, rPos in zip(aCSC.colPos, bCSR.rowPos):
        cnt += 1
        if cnt %1000 == 0:
            print(cnt)
        for i in range(lastA, cPos):
            for j in range(lastB, rPos):
                x = aCSC.rowId[i]
                y = bCSR.colId[j]
                c = aCSC.data[i]*bCSR.data[j]
                if (x,y) in result:
                    result[(x,y)] += c
                else:
                    result[(x,y)] = c
        lastA = cPos
        lastB = rPos
    result=[(k[0], k[1], v) for k, v in result.items() if abs(v)>1e-6]
    result.sort()
    return result

def makeUniformData(density, w, h):
    visit = set()
    result = []
    n = int(w*h*density + rng.normal(scale=100))
    for i in range(n):
        if i%1000 == 0:
            print(i)
        x = math.floor(rng.uniform(0, h))
        y = math.floor(rng.uniform(0, w))
        while (x,y) in visit:
            x = math.floor(rng.uniform(0, h))
            y = math.floor(rng.uniform(0, w))
        visit.add((x,y))
        result.append((x,y, rng.uniform(-1, 1)))

    return result

def makeGaussianData(density, w, h):
    visit = set()
    result = []
    n = int(w*h*density + rng.normal(scale=100))
    for i in range(n):
        if i%1000 == 0:
            print(i)
        x = math.floor(rng.uniform(0, h))
        y = math.floor(rng.uniform(0, w))
        while (x,y) in visit:
            x = math.floor(rng.uniform(0, h))
            y = math.floor(rng.uniform(0, w))
        visit.add((x,y))
        result.append((x,y, rng.normal()))

    return result

def makeBiasedGaussianData(density, w, h):
    visit = set()
    result = []
    n = int(w*h*density + rng.normal(scale=100))
    for i in range(n):
        if i%1000 == 0:
            print(i)
        x = math.floor(rng.uniform(0, h))
        y = math.floor(rng.uniform(0, w))
        while (x,y) in visit:
            x = math.floor(rng.uniform(0, h))
            y = math.floor(rng.uniform(0, w))
        visit.add((x,y))
        result.append((x,y, rng.normal(10.0, 5.0)))

    return result

if __name__ == "__main__":
    n = 1000000
    m = 1000000
    k = 1000000
    print("Make A")
    a = makeBiasedGaussianData(1e-6, n, m)
    print(len(a))
    printMatrix("data/gaussianLarge2A.txt", a)
    # a = readMatrix("data/gaussianA_large.txt")
    print("Make B")
    b = makeBiasedGaussianData(1e-6, m, k)
    print(len(b))
    printMatrix("data/gaussianLarge2B.txt", b)
    # b = readMatrix("data/gaussianB_large.txt")
    print("Make C")
    t, c = multiplyCOO(a, b)
    print(t)
    printMatrix("data/gaussianLarge2C.txt", c)
