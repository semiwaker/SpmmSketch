import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from spmmSketch.util import *

def draw(inFile, outFile, c):
    a = readMatrix(inFile)
    values = [c for (x,y,c) in a]
    
    plt.hist(values, 1000, log=True, color=c)
    plt.ylim(top=1e4)
    plt.savefig(outFile)

def draw2(outFile):
    a = readMatrix("data/gaussianLarge2A.txt")
    b = readMatrix("data/gaussianLarge2B.txt")
    aCSC = COO2CSC(a)
    bCSR = COO2CSR(b)
    values = []
    lastA = 0
    lastB = 0
    for cPos, rPos in zip(aCSC.colPos, bCSR.rowPos):
        for i in range(lastA, cPos):
            for j in range(lastB, rPos):
                c = aCSC.data[i]*bCSR.data[j]
                values.append(c)
        lastA = cPos
        lastB = rPos
    
    plt.hist(values, 1000, log=True)
    plt.ylim(top=1e4)
    plt.savefig(outFile)

def drawSegError(inFile, outFile, c):
    a = readMatrix("data/gaussianLarge2C.txt")
    b = readMatrix(inFile)
    labels, err = segmentedAcc(a,b, -250, 700, 50, 0.05)
    
    plt.plot(err, color=c)
    plt.xticks(ticks=[i-0.5 for i in range(len(err)+1)], labels=labels, rotation=45)
    plt.savefig(outFile)

if __name__=="__main__":
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    drawSegError("out/result2.txt", "segErr.png", colors[0])
    drawSegError("out/result4.txt", "segErr.png", colors[1])
    drawSegError("out/result3.txt", "segErr.png", colors[2])
    # drawSegError("out/result3.txt", "segErr.png", colors[0])
    # drawSegError("out/result5.txt", "segErr.png", colors[1])
    # drawSegError("out/result5_dec.txt", "segErr.png", colors[2])
    # drawSegError("out/result6.txt", "segErr.png", colors[3])
    # draw("out/result2.txt", "histHash.png", colors[1])
    # draw("data/gaussianLarge2C.txt", "histHash.png", colors[0])
    # draw("out/result3.txt", "histHash.png", colors[1])
    # draw("out/result4.txt", "histHash.png", colors[3])