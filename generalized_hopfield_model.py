#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 19:20:24 2025

@author: hector
"""

import numpy as np 

# should i think about tupules to couple training memories with their respective label? 

class GHN:
    def __init__(self, M_len, L_len, K, n_deg, Temp, m_deg, dt, momentum, rand_mean=-0.003, rand_std=0.00001, Mems=None, Labels=None):
        
        self.K = K #total number of patterns
        self.N_M = M_len #length of every row that contains the i-th pattern [visible variable] 
        self.N_L = L_len #length of every row that contains the i-th label [hidden variable]
        self.n = n_deg #power of the ReLU function, GHN superparameter
        self.T = Temp #temperature of the ReLU function, normalized the dot product of the argument  
        self.m = m_deg #power used in the cost function, normally m = n 
        self.dt = dt #rate at which the labels and memories are evolved while learning
        self.momentum = momentum #sneaky parameter that smoothens the learning protocol, explained in the original GHN
        self.rand_mean = rand_mean #normal mean for initializing the memory and label values at t = 0 
        self.rand_std = rand_std #normal standard dev for initializing the memory and label values at t = 0 
        
        if Mems == None:
            self.Ms = np.random.normal(self.rand_mean,self.rand_std, (self.K , self.N_M))
        else:
            self.Ms = Mems
            
        if Labels == None: 
            self.Ls = np.random.normal(self.rand_mean,self.rand_std, (self.K , self.N_L))
        else:
            self.Ls = Labels 
            
        self.dMs = np.zeros( ( self.K, self.N_M ) )
        self.dLs = np.zeros( ( self.K, self.N_L ) )
        
    def f_n(self, x, n):
        return np.power( (x + np.abs(x))/2.0 , n)
            
    def output(self, inpt_vec):
        pre_out = np.divide( self.f_n( np.dot(self.Ms, inpt_vec), self.n), np.power(self.T, self.n))
        out = np.repeat(pre_out, self.N_L).reshape( (self.K, self.N_L) )
        return np.tanh( np.sum( np.multiply(self.Ls, out ), axis=0)  )
    
    def f_cost(self, inpt_vec, train_L):
        return np.sum( np.power(train_L - self.output(inpt_vec), 2*self.m ))
    
    def gradC_M(self, train_Ms, train_Ls):
        gCM = np.zeros_like(self.Ms)
        for i in range(train_Ls.shape[0]):
            V_ = train_Ms[i,:]
            L_ = train_Ls[i,:]
            
            out = self.output(V_)
            gCM_1 = np.power( L_ - out, 2*self.m-1)*(1-out**2)*self.Ls*2*self.m
            gCM_2 = self.n/np.power(self.T, self.n)*self.f_n(np.dot(self.Ms, V_), self.n-1)
            gCM_3 = np.sum( gCM_1*np.repeat(gCM_2, self.N_L).reshape((self.K,self.N_L)), axis=1)
            gCM += np.repeat(gCM_3, self.N_M).reshape((self.K, self.N_M))*V_
            
        return -1*gCM
        
    
    def gradC_L(self, train_Ms, train_Ls):
        gCL = np.zeros_like(self.Ls)
        for i in range(train_Ls.shape[0]):
            V_ = train_Ms[i,:]
            L_ = train_Ls[i,:]
            
            out = self.output(V_)
            gCL_1 = np.power( L_ - out, 2*self.m-1)*(1-out**2)*2*self.m
            gCL_2 = np.divide(self.f_n(np.dot(self.Ms, V_), self.n), np.power(self.T, self.n))
            gCL += np.repeat(gCL_2, self.N_L).reshape((self.K,self.N_L))*gCL_1
            
        return -1*gCL
        

    def train_protocol(self, train_Ms, train_Ls, noise_mean, noise_std):
    
        grad_C_M = self.gradC_M( train_Ms, train_Ls )
        norm_gCM = np.repeat( np.max( np.abs(grad_C_M), axis=1 ), self.N_M).reshape( ( self.K, self.N_M ) )
        normed_gCM = np.divide( grad_C_M, norm_gCM, out=np.zeros_like( grad_C_M ), where=norm_gCM != 0 )
        
        self.dMs = normed_gCM + np.random.normal(noise_mean, noise_std, (self.K, self.N_M))
        self.Ms -=  self.dt*self.dMs 
        norm_Ms = np.repeat( np.max( np.abs(self.Ms).clip(min=1.0), axis=1 ), self.N_M ).reshape( ( self.K, self.N_M ) )
        self.Ms = np.divide( self.Ms, norm_Ms )
        
        grad_C_L = self.gradC_L( train_Ms, train_Ls )
        norm_gCL = np.repeat( np.max( np.abs(grad_C_L), axis=1 ), self.N_L).reshape( ( self.K, self.N_L ) )
        normed_gCL = np.divide( grad_C_L, norm_gCL, out=np.zeros_like( grad_C_L ) , where=norm_gCL != 0 )
        
        self.dLs = normed_gCL + np.random.normal(noise_mean, noise_std, (self.K, self.N_L))  
        self.Ls = np.clip( self.Ls - self.dt*self.dLs , -1, 1) 

        

    
    
    
