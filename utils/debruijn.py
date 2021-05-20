import numpy as np

class DeBruijnGraph:
    def chop(self,st,k):
        a=np.empty([0])
        b=np.empty([0])
        c=np.empty([0])
        for i in range(0, len(st)-(k-1)):
            a=np.append(a,[st[i:i+k]],axis=0)
            b=np.append(b,[st[i:i+k-1]],axis=0)
            c=np.append(c,st[i+1:i+k])
        return a,b,c

    def generate(self,st,k):
        if k<=1 :
            print("invalid value of k returning empty graph")
            return
        if len(st)<k:
            print("insufficient size of string input returning empty graph")
            return
        a,b,c= self.chop(st,k)
        hash={b[0]:0}
        j=1
        for i in range (0,a.shape[0]):
            if c[i] in hash :
               self.edge_index=np.append(self.edge_index,[[hash[b[i]]],[hash[c[i]]]],axis=1)
            else:
                hash[c[i]]=j
                j=j+1
                self.edge_index=np.append(self.edge_index,[[hash[b[i]]],[hash[c[i]]]],axis=1) 
        for h in hash:
            self.x=np.append(self.x,[[h]],axis=0)

    def reverse(self): #gives back the DNA sequence from the graph
        #print(self.edge_index.shape)
        if self.edge_index.shape[1]==0 or self.x.shape[0]==0:
            return ''
        a=self.x[self.edge_index[0][0]][0]
        b=self.x[self.edge_index[1][0]][0]
        kmer=a[0:len(a)-1]+b
        st=kmer
        for i in range (1,self.edge_index.shape[1]):
            a=self.x[self.edge_index[0][i]][0]
            b=self.x[self.edge_index[1][i]][0]
            kmer=a[0:len(a)-1]+b
            st=st+kmer[len(kmer)-1]
        return st

    def one_hot_encode(self, seq):
    	mapping = dict(zip("ACGT", range(4)))    
    	seq2 = [mapping[i] for i in seq]
    	return np.eye(4)[seq2]    
    
    def __init__(self,st,k):
        x=np.empty([0,1])
        edge_index=np.empty([2,0],dtype=int)
        self.x=x
        self.edge_index=edge_index
        self.generate(st,k)