#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np


# In[3]:


class Value:
    def __init__(self,data,_children=(),_op="",label=""):
        self.data=data
        self._prev=set(_children)
        self.grad=0
        self._backward= lambda: None
        self._op=_op
        self.label=label

    def __repr__(self):
        return f"Value is {self.data}"

    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out=Value(self.data + other.data,(self,other),"+")
        def _backward():
            #self.grad=(local gradient) * (global gradient)
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward    
        
        return out
    def __radd__(self,other):
        return self+other

    def __sub__(self,other):
        return self + (-other)
    def __rsub__(self,other):
        return self + (-other)    
   

    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data,(self,other),"*")
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward=_backward    
        return out

    def __pow__(self,other):
        assert isinstance(other,(int, float)) # support for int and float only
        out = Value(self.data**other,(self,),f"**{other}")

        def _backward():
            self.grad += other*(self.data**(other-1)) * out.grad
        out._backward=_backward
        return out
        
    def __rmul__(self,other):
        return self*other

    def __truediv__(self,other):
        return self * (other**-1)
        
    def tanh(self):
        n=self.data
        t=(math.exp(2*n)-1)/(math.exp(2*n)+1)
        out = Value(t,_children=(self,),_op='tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward  = _backward    
        
        return out

    def exp(self):
        n=self.data
        t=math.exp(n)
        out = Value(t,_children=(self,),_op='exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward  = _backward    
        
        return out
    def backward(self):
        topo=[]
        visited=set()
        def topo_ord(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    topo_ord(child)
                topo.append(v)         
        topo_ord(self)
        self.grad=1.0
        for node in reversed(topo):
            node._backward()
        


# In[ ]:




