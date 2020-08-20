import autograd.numpy as np
from autograd import grad

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm


class node():
    def __init__(self,depth=1,tree_depth=2,path=False,xdim=4,ydim=3):
        self.xdim = xdim
        self.ydim = ydim
        self.tree_depth = tree_depth
        self.depth = depth
        self.node_params = np.random.normal(size=(2,self.xdim))
        self.node_params[1,:] = np.random.uniform(size=self.xdim) /2
        #self.node_params = np.ones((2,self.xdim))*0.5
        self.path_prob = None
        self.is_leaf = False
        
        if not path:
            self.path = [0]
        else:
            self.path = path
        
        if self.depth < self.tree_depth:
            
            path_l = [i for i in self.path]
            path_l.append(0)
            path_r = [i for i in self.path]
            path_r.append(1)
            
            self.l_child = node(depth = self.depth+1, tree_depth = self.tree_depth, \
                                path = path_l,xdim = self.xdim, ydim = self.ydim)
            self.r_child = node(depth = self.depth+1, tree_depth = self.tree_depth, \
                                path = path_r,xdim = self.xdim, ydim = self.ydim)
            
            self.forward = self.forward_inner
            self.forward_hard = self.forward_inner_hard
        else:
            self.l_child = None
            self.r_child = None
            self.is_leaf = True
            
            #self.node_params = np.array([1.0/self.ydim] * self.ydim)
            self.node_params = np.random.dirichlet([1.0] * self.ydim).reshape(self.ydim)
            
            self.forward = self.forward_leaf
            self.forward_hard = self.forward_leaf_hard
            
        self.children = [self.l_child,self.r_child]
        
        
    @staticmethod
    def soft_step(x,t=1.0):
        
        return np.where(x>=0, 1/(1+np.exp(-x*t)),
                       np.exp(x*t) / (1+np.exp(x*t)))
    
    '''
    def single(self,x,alpha,theta):
        
        t = (alpha*(self.soft_step(x-theta,t=2.0)) + \
                (1-alpha)*(1-self.soft_step(x-theta,t=2.0)))
        return t
    '''
    def single(self,x,alpha,theta):
        
        return self.soft_step(x-theta,alpha)
    
    def cov_splits(self,X,params):
    
        p = [self.single(cov,params[0,i],params[1,i]) \
                for i,cov in enumerate(X)]
        return p
    
    def split_prob(self,X,params):
        
        p = [self.single(cov,params[0,i],params[1,i]) \
                for i,cov in enumerate(X)]
        
        #return np.sum(p)/self.xdim
        return np.prod(p)
    
    def forward_inner(self,X,tree_params):
        
        param = self.select_params(self.path,tree_params)
        p = self.split_prob(X,param)
        
        l_path = self.l_child.forward(X,tree_params) + np.log(1-p)
        r_path = self.r_child.forward(X,tree_params) + np.log(p)
        
        return np.log(np.exp(l_path) + np.exp(r_path))
    
    def forward_leaf(self, X, tree_params):
        
        param = self.select_params(self.path,tree_params)
        
        e_x = np.exp(param - np.max(param))

        return np.log(e_x / e_x.sum(axis=0))

    def forward_inner_hard(self,X,tree_params):
        
        param = self.select_params(self.path,tree_params)
        
        ps = []
        for i in range(self.xdim):
            if np.sign(param[0,i]) == 1.0:
                ps.append(X[i]>param[1,i])
            else:
                ps.append(X[i]<param[1,i])
                
        if np.all(ps):
            return self.r_child.forward_hard(X,tree_params)
        else:
            return self.l_child.forward_hard(X,tree_params)
    
    def forward_leaf_hard(self, X, tree_params):
        
        param = self.select_params(self.path,tree_params)

        return (param == np.max(param)).astype(int)
    
    @staticmethod
    def select_params(path,tree_params):
        
        l = len(path)
        idx = 2**(l-1)
        
        path_p = path[1:]
        path_p.reverse()
        
        binar = [(2**i)*v for i,v in enumerate(path_p)]
        idx += sum(binar)
        idx -= 1
        return tree_params[idx]
    
    
class soft_tree():
    
    def __init__(self,tree_depth,xdim=4,ydim=3):
        self.tree_depth = tree_depth
        self.params = []
        
        self.xdim = xdim
        self.ydim = ydim
        
        self.build_tree(depth = tree_depth)
        self.get_params()
        
    def build_tree(self,depth = 1):
        self.nodes = [None] * depth
        self.root = node(depth = 1,tree_depth = self.tree_depth, \
                         xdim = self.xdim, ydim = self.ydim)
        
        self.nodes[0] = [self.root]
        
        for i in range(self.tree_depth - 1):
            
            nod_list = []
            for nod in self.nodes[i]:
                nod_list.append(nod.l_child)
                nod_list.append(nod.r_child)
            self.nodes[i+1] = nod_list            
        return
    
    def get_params(self):
        
        for nod_list in self.nodes:
            for nod in nod_list:
                self.params.append(nod.node_params)
        return
    
    def push_params(self):
        
        for nod_list in self.nodes:
            for nod in nod_list:
                nod.node_params = nod.select_params(nod.path, self.params)
        return
    
    def forward(self,X,params = None):
        if not params:
            params = self.params
        ys = [np.exp(self.root.forward(x,params)).reshape(self.ydim) for x in X]
        return np.array(ys)
    
    def forward_hard(self,X,params = None):
        if not params:
            params = self.params
        ys = [self.root.forward_hard(x,params).reshape(self.ydim) for x in X]
        return np.array(ys)
    
    
    def ce_loss(self,params):
        preds = self.forward(self.X_train,params)
        return -np.sum(np.log(preds) * self.y_train)

    @staticmethod
    def update_step(params,grads,l_rate):
    
        l = len(params) 
        updated_params = []
        for i in range(l):
            par = params[i] - (l_rate * grads[i])
            updated_params.append(par)
        return updated_params
    
    def train(self,train_x,train_y,iters,l_rate=0.01):
        self.X_train = train_x
        self.y_train = train_y
        
        grad_p = grad(self.ce_loss)
        
        self.loss_history = []

        self.par_history = []
        self.par_history.append(self.params)
        
        out_file = open("model_training.txt","w+")
        out_file.write('Iter : Training Loss \n')
        out_file.close()
        
        flag = False
        
        for it in tqdm(range(iters),desc='Optimising Parameters'):
            grads = grad_p(self.params)
            self.params = self.update_step(self.params,grads,l_rate)
    
            self.par_history.append(self.params)
            lo = self.ce_loss(self.params)
            
            
            with open('model_training.txt', 'a') as out_file:
                out_file.write(f'{it}    : {lo}\n')
                
            if (not flag) & it > 0:
                if lo > self.loss_history[-1]:
                    print(f'WARNING - Loss not decreasing at iter: {it}')
                    flag = True
                
            self.loss_history.append(lo)
            
        self.diagnose()
        self.push_params()
        
        return
        
    def diagnose(self):
        
        idx_min = np.argmin(self.loss_history)
            
        plt.plot(self.loss_history,label='Training Loss')
        plt.xlabel('Iteration')
        
        if idx_min != (len(self.loss_history)-1):
            
            print("WARNING - Potential convergence issues")
            self.params = self.par_history[idx_min]
            plt.axvline(idx_min,color='red', label = f'Min at {idx_min}')
        
        plt.legend()
        plt.show()
        return
