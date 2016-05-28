import numpy as np
import sys

def readData(path):
    with open(path) as f:
        D,N_true,N_false=[int(i) for i in f.readline().strip().split()]
    data=np.loadtxt(path,skiprows=1)
    label=np.array([1]*N_true+[0]*N_false,int)
    return data,label

def parseArg():
    firstArg=sys.argv[1]
    args=dict()
    if firstArg=='-v':
        args['verbose']=True
    else:
        args['verbose']=False
    args['T']=sys.argv[1+args['verbose']]
    args['size']=sys.argv[2+args['verbose']]
    args['Xtr'],args['Ytr']=readData(sys.argv[3+args['verbose']])
    args['Xte'],args['Yte']=readData(sys.argv[4+args['verbose']])
    return args
    
class bagofbi:
    def __init__(self,initFunc):
        """
        self.p stands for parameters, and data
        initFunc is user defined, feed in parameters
        dictionary, contains all parameters and data
        designed according to caffe style
        """
        self.p=initFunc()
        #for i in self.p:
        #    print self.p[i]



if __name__=="__main__":
    model=bagofbi(parseArg)
