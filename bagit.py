import numpy as np
import sys

def readData(path):
    with open(path) as f:
        D,N_true,N_false=[int(i) for i in f.readline().strip().split()]
    data=np.loadtxt(path,skiprows=1)
    n=len(data)
    label=np.array([1]*N_true+[0]*N_false,bool)
    return data,label,n,D

def arr2str(arr):
    s=''
    if len(arr.shape)==1:

        return ' '.join(map(str,arr))
    else:
        for subarr in arr:
            s+=arr2str(subarr)+'\n'
    return s

def parseArg():
    firstArg=sys.argv[1]
    args=dict()
    if firstArg=='-v':
        args['verbose']=True
    else:
        args['verbose']=False
    args['T']=int(sys.argv[1+args['verbose']])
    args['size']=int(sys.argv[2+args['verbose']])
    args['Xtr'],args['Ytr'],args['ntr'],args['D']=readData(sys.argv[3+args['verbose']])
    args['Xte'],args['Yte'],args['nte'],args['D']=readData(sys.argv[4+args['verbose']])
    return args
    
class bagofbi:
    log=''
    status=None
    show=dict()
    debug=False

    def __init__(self,initFunc):
        """
        self.p: stands for parameters, it contains these:
                T: number of classifiers
                size: size of bootstrap sample sets
                Xtr Xte: points,
                Ytr Yte: labels
                ntr nte: number of points
                D: dimension
        initFunc: user defined, feed in parameters
                dictionary, contains all parameters and data
                designed according to caffe style
        """
        self.p=initFunc()
        print "Positive examples:",str(sum(self.p['Yte']))
        print "Negative examples:",str(self.p['nte']-sum(self.p['Yte']))

    def bagging(self): 
        """
        create bags of classifiers
        self.w, weight matrix
        self.b, bias vector
        """
        #frozen random seeds, used for debug
        if self.debug:
            np.random.seed(0xE3A8)
        
        #bootstrap sample index matrix D, T by size
        idx=np.random.randint(0,high=self.p['ntr'],
            size=(self.p['T'],self.p['size']))
        # some sample sets may only have one class, deal with this at the end of bagging, for convinience of computing
        # T by size mat
        all_label=self.p['Ytr'][idx]
                
        # bootstrap sample cube
        D=self.p['Xtr'][idx]

        if self.p['verbose']:
            self.log='\n'
            for i in range(D.shape[0]):
                self.log+="Bootstrap sample set "+str(i+1)+':\n'
                for j in range(D.shape[1]):
                    self.log+= arr2str(D[i,j])+' - '+str(all_label[i,j])+'\n'
                self.log+='\n'

        idx=np.dstack([all_label]*self.p['D'])
        #positive centroids,
        pos=(D*idx).mean(1)
        #negative centroids
        neg=(D*(~idx)).mean(1)
        # for one class centroids will be 0, deal with this later

        # w matrix, b vector
        self.w=pos-neg
        self.b=(self.w*((pos+neg)/2)).sum(1)
        
        #deal with one class case
        #All positive case: push b to -inf, 
        self.b[(~all_label).sum(1)==0]=-np.inf
        #All negative case: push b to +inf, so test examples will always be negative
        self.b[all_label.sum(1)==0]=np.inf

        if self.debug:
            print self.w,self.b

    def predict(self):
        #predicions are nte by T matrix, >0 makes boundary false
        prediction=(self.p['Xte'].dot(self.w.T)-self.b)>0
        #sum along T to vote, votes is nte by 1 vec
        votes=prediction.sum(1)
        
        self.res=votes>=(self.p['T']/2.0)
        # result is nte by 1 vec, >= makes ties true class
        fp=self.res & (~self.p['Yte'])
        fn=(~self.res) & self.p['Yte']

        self.status=np.array(['correct']*self.p['nte'],dtype=object)
        self.status[fp]='false positive'
        self.status[fn]='false negative'
        self.show['False positives']=int(sum(fp))
        self.show['False negatives']=int(sum(fn))
        for i in self.show:
            print i,':',self.show[i]
        if self.p['verbose']:
            self.log+='Classification:\n'
            for i in range(self.p['nte']):
                self.log+=arr2str(self.p['Xte'][i])+' - '+str(self.res[i])+\
                       ' ('+self.status[i]+')\n'
        print self.log



if __name__=="__main__":
    model=bagofbi(parseArg)
    model.debug=True
    model.bagging()
    model.predict()
