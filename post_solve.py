import json
import argparse
import os
import sys
import operator
from pyeda.inter import *
import numpy as np
from collections import OrderedDict
import re



def genTruthTable(outputs,inbits,outbits = 0,x = []):
    if outbits < 1:
        outbits = max(outputs).bit_length()
    ttout_entries = 2**int(inbits)

    ttouts = np.empty([ttout_entries, outbits], dtype = str)
    for i in range(ttout_entries):
        try:   
            _tmp = outputs[i]
            if isinstance(_tmp,float):
                _tmp = x.index(_tmp)
            ttout = np.array(list('{0:0{1}b}'.format(_tmp,outbits)))
            ttouts[i] = ttout
        except (IndexError, KeyError) as e:
            ttouts[i] = '-'*outbits


    transposed_tt = ttouts.transpose()
    tt_stringlist = [];
    for i,tt_list in enumerate(transposed_tt):
        ttstring= ''.join(list(tt_list))
        tt_stringlist.append(ttstring)

    return tt_stringlist,ttouts

def genBoolExpr(outputs,nbits, varname, outbits = -1, x = []):
    X = ttvars(varname,int(nbits))
    truth_strings,tt = genTruthTable(outputs,nbits,outbits,x)
    logfun = []
    if outbits < 1:
        outbits = clog2(max(outputs))
    for i in range(outbits):
        ttable = truthtable(X,truth_strings[i])
        logfun.append(ttable)
    fm = espresso_tts(*logfun)
    return fm,tt


def genLzBoolExpr(x,nbits,uLy,uLy2y):
    lzMap = genLzTable(x,uLy,uLy2y)
    return genBoolExpr(lzMap,((nbits - 1)*2),"z",nbits,x)
    

def genLzTable(x,uLy,uLy2y):
    return dict(uLy2y)

def main():

    opfun = {
        '+': operator.add,
        '*': operator.mul,
        '/': operator.truediv,
        '-': operator.sub
    }

    parser = argparse.ArgumentParser(description='Process some integers.')    
    parser.add_argument('input')
    parser.add_argument('--bits')
    parser.add_argument('--prefix')
    args = parser.parse_args()

    inbits = int(args.bits)

    solution = json.load(open(args.input,"r"))
    Lx1 = solution['Lx1']
    Lx2 = solution['Lx2']
    Ly  = solution['Ly']
    uLy = list(dict.fromkeys((Ly)))
    uLy2y = solution['uLy2y']
    y   = solution['y']
    x   = solution['x1']
    m1,mx1 = genBoolExpr(Lx1,int(args.bits),"x")
    m2,mx2 = genBoolExpr(Lx2,int(args.bits),"y")
    m3,mx3 = genLzBoolExpr(x,int(args.bits),uLy,uLy2y)

    mnaive,mxnaive = genBoolExpr(genNaiveTable(x,y,inbits),int(2*inbits),"y")

    printRecap(x,y,Lx1,Lx2,Ly,uLy,uLy2y,m1,m2,m3,mx1,mx2,mx3,mnaive,mxnaive,int(args.bits))
    printFinalOpListing(x,Lx1,Lx2,uLy,uLy2y,solution['op'],opfun[solution['op']])



def printTruthTab(mx,outbits):
    for i,x in enumerate(mx):
        print('{0:0{1}b}'.format(i,outbits),"\t",''.join(list(mx[i])))

def printZTruthTab(mx,outbits, x,Lx1,Lx2,uLy,uLy2y):
    for i,x__ in enumerate(mx):
        try:
            x_ = x[int(''.join(list(mx[i])),2)]
        except:
            x_ = "-"
            print("",end="")
        print('{0:0{1}b}'.format(i,outbits),"\t",
                ''.join(list(mx[i])),"\t",
                x_
                
            )    

def countGates(boolExpr):
    andGates = len(re.findall("And", boolExpr))
    orGates  = len(re.findall("Or",  boolExpr))
    return andGates + orGates

def printFinalOpListing(x,Lx1,Lx2,uLy,uLy2y,op,opfun):
    print("\n===============")
    print("x: ",x)
    print("Lx: ",Lx1)
    print("Ly: ",Lx2)
    print("Lz:", uLy)
    print("Lz2z:", uLy2y)
    print("")
    ymap = dict(uLy2y)
    for i,lx in enumerate(Lx1):
        for j,ly in enumerate(Lx2):
            zindex = Lx1[i]+Lx2[j]
            print(x[i],op,x[j],"=",opfun(x[i],x[j]))
            print(Lx1[i],"+",Lx2[j],"=",zindex)
            print("Lz[",zindex,"] = ",ymap[zindex])
            print()
            print("-----------------")

def printOpTable(x,Lx1,Lx2,uLy,uLy2y):

    ymap = dict(uLy2y)

    print("",end="\t")
    for x_ in x:
        print(x_,end="\t")
    print("\n")
    for i,lx in enumerate(Lx1):
        print(x[i],end="\t")
        for j,ly in enumerate(Lx2):
            zindex = Lx1[i]+Lx2[j]
            print(uLy2y[uLy.index(zindex)][0],end="\t")
        print("")

def genNaiveTable(x,y,inbits):
    x = np.array(x)
    xf = np.concatenate(([0],x,[np.inf],-x))
    of = []
    for i,x1 in enumerate(xf):
        for j,x2 in enumerate(xf):
            ib = '{0:0{1}b}'.format(i,inbits)
            jb = '{0:0{1}b}'.format(j,inbits)
            out = x1+x2
            minIdx = np.argmin((xf-out)**2)
            of.append(minIdx)
    of = [x.item() for x in of]
    return of
# We will use paper nomenclature here
# x1 => x; x2 => y; y => z
def printRecap(x,y,Lx1,Lx2,Ly,uLy,uLy2y,m1,m2,m3,mx1,mx2,mx3,mn,mxn,inbits):
    print("\n===============\n")
    print("POSIT DOMAIN")
    print("Operands (x,y):")
    print(x,"\n")
    print("Result (z):")
    print(y)
    print("\n===============\n")
    print("MAPPING")
    print("x => Lx")
    print(x,"=>",Lx1,"\n")
    print("y => Ly")
    print(x,"=>",Lx2,"\n")
    
    print("Lz => z")
    print("[",end="")
    for ul in uLy2y[:-1]:
        print(ul[0],end=", ")
    print(uLy2y[-1][0],end="")
    print("] => ", end ="")

    print("[",end="")
    for ul in uLy2y[:-1]:
        print(ul[1],end=", ")
    print(uLy2y[-1][1],end="")
    print("] => ", end ="")

    print("Lx+Ly: ")
    print("\t","\t".join(map(str,Lx2)))
    print("")
    for i,l in enumerate(np.reshape(Ly,[len(Lx1), len(Lx2)])):
        print(Lx1[i],"\t","\t".join(map(str,list(l))))
    
    #print("\nLz[Lx+Ly]:\n")

    #printOpTable(x,Lx1,Lx2,uLy,uLy2y)
    #printFinalOpListing(x,Lx1,Lx2,uLy,uLy2y,"*")


    print("\nLz => z [uniques only]")
    print(uLy,"=> ",end="")
    print("[",end="")
    for i,ly_ in enumerate(uLy):
        print(uLy2y[i],end="")
        if i < len(uLy) - 1:
            print(", ",end="")

    print("]\n")
    print("Full Lz: ", len(Ly))
    print("Unique Lz: ", len(uLy))
    print("Max Lz: ", max(Ly))
    print("Min Lz: ", min(Ly))
    print("Max-min Lz: ", max(Ly) - min(Ly))
    print("% saved using uniques: ", 100*(len(Ly)-len(uLy))/len(Ly),"%")

    print("\n===============\n")

    print("TRUTH EXPR")
    print("x\t Lx")
    printTruthTab(mx1,inbits)
    print(str(m1))
    print("Gates: ",countGates(str(m1)))

    print("\ny\t Ly")
    printTruthTab(mx2,inbits)
    print(m2)
    print("Gates: ",countGates(str(m2)))


    print("\nLz\t z\t z (real)")
    printZTruthTab(mx3,(inbits - 1)*2, x,Lx1,Lx2,uLy,uLy2y)
    print(m3)
    print("Gates: ",countGates(str(m3)))

    print("\nLz\t z")
    printTruthTab(mxn,2*inbits)
    print(mn)
    print("Gates: ",countGates(str(mn)))


if __name__ == '__main__':
    main()
