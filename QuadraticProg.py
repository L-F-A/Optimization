from cvxopt import matrix as cvxmat, sparse as cvxsparse, spmatrix as cvxspmatrix
from cvxopt.solvers import qp as cvxQP, options as cvxopt
import numpy as np

def quadprog(H, f, Aeq=None, beq=None, Aineq=None, bineq=None, lb=None, ub=None,abstol=None,reltol=None,maxiters=None,returnVar='yes'):
    """
    minimize:
            (1/2)*x'*H*x + f^T*x
    subject to:
            Aeq*x = beq
	    Aineq <= bineq 
            lb <= x <= ub

##########################################################################################################################
#        Matlab style quadratic programming using cvxopt with equality, inequality and lower and upper bounds.           #
#                                                                                                                        #
#             I modified slightly the code from https://gist.github.com/garydoranjr/1878742 to fit what I wanted                  #
#                                                                                                                        #
#                     Louis-Francois Arsenault, Columbia University la2518@columbia.edu (2013-2017)                      #
##########################################################################################################################
#                                                                                                                        #
#       INPUTS:                                                                                                          #
#                                                                                                                        #
#         H     : numpy 2d array([[h_11,h_12,...,h_1n],[h_21,h_22,...,h_2n],...,[h_n1,h_n2,...,h_nn]])                   #
#		  can also be a scalar which mean H is the scalar H time the identity matrix                             #
#															 #
#         f     : numpy 1d array([f_1,f_2,f_3,...,f_n])                                                                  #
#                                                                                                                        #
#         Aeq   : numpy 2d array([[aeq_11,aeq_12,...,aeq_1n],[aeq_21,aeq_22,...,aeq_2n],...,[aeq_p1,aeq_p2,...,aeq_pn]]) #
#                                                                                                                        #
#         beq   : numpy 1d array([beq_1,beq_2,beq_3,...,beq_p])                                                          #
#                                                                                                                        #
#         Aineq : numpy 2d array([[ai_11,ai_12,...,ai_1n],[ai_21,ai_22,...,ai_2n],...,[ai_l1,ai_l2,...,ai_ln]])          #
#                                                                                                                        #
#         bineq : numpy 1d array([bi_1,bi_2,bi_3,...,bi_l])                                                              #
#                                                                                                                        #
#         lb    : numpy 1d array([lb_1,lb_2,...,lb_n])                                                                   #
#                                                                                                                        #
#         ub    : numpy 1d array([ub_1,ub_2,...,ub_n])                                                                   #
#                                                                                                                        #
#	  Optionnal:													 #
#	  abstol   : absolute tolerance											 #
#	  reltol   : relative tolerance											 #
#	  maxiters : max number of iterations										 #
#															 #
##########################################################################################################################
#	Note that as long as f,beq,bineq,lb and ub are somewhat 1d like arrays like shape =  (m,1) or (1,m),		 #
#			         they will be reshaped to have the correct (m,) shape !!                                 #
##########################################################################################################################
"""
    P, q, G, h, A, b = _convert(H, f, Aeq, beq,Aineq,bineq, lb, ub)
    
    cvxopt['show_progress'] = False
    #Use options given. If None, use default
    if abstol != None:
	cvxopt['abstol'] = abstol
    if reltol != None:
	cvxopt['reltol'] = reltol
    if maxiters != None:
	cvxopt['maxiters'] = maxiters
    
    #Only returns the values
    results = cvxQP(P, q, G, h, A, b)

    #Convert back to NumPy array
    if returnVar is 'yes':
    	return np.array(results['x'])
    else:
	return results

def _convert(H, f, Aeq=None, beq=None,Aineq=None,bineq=None, lb=None, ub=None):                                                                                  
    
    #Convert everything to                                                                                              
    #cvxopt-style matrices
                                                                                              
    #P and q are always given
    if isinstance(H,np.ndarray) is False:
	#H is a scalar meaning H*I, where I identity matrix
	P=H*speye(len(f))
    else: 
    	P = cvxmat(H,tc='d')
    q = cvxmat(f, tc='d')

    if Aeq is None:                                                                                                    
        A = None
	b = None #if no Aeq, no beq                                                                                                       
    else:
	if Aeq.ndim == 1:
		Aeq.shape = (1,Aeq.size)
	elif (Aeq.shape[1]==1):
		Aeq = Aeq.transpose() 
        A = cvxmat(Aeq,tc='d')
	if beq.ndim != 1:
		beq.shape = (beq.size,)
	b = cvxmat(beq,tc='d')                                                                                                

    if Aineq is None:
	if lb is None:
		if ub is None:
			G = None
			h = None
		else:
			if ub.ndim != 1:
		                ub.shape = (ub.size,)
			n=ub.size
			G = cvxsparse(speye(n),tc='d')
			h = cvxmat(ub,tc='d')
	else:
		if lb.ndim != 1:
                	lb.shape = (lb.size,)
		if ub is None:
			n=lb.size
			G = cvxsparse(-speye(n),tc='d')
			h = cvxmat(-lb,tc='d')
		else:
			if ub.ndim != 1:
                                ub.shape = (ub.size,)
			n = lb.size
    			G = cvxsparse([speye(n), -speye(n)],tc='d')
    			h = cvxmat(np.hstack([ub,-lb]),tc='d')
    else:
	if bineq.ndim != 1:
        	bineq.shape = (bineq.size,)
	if lb is None:
		if ub is None:
			G = cvxmat(Aineq,tc='d')
			h = cvxmat(bineq,tc='d')
		else:
			if ub.ndim != 1:
                                ub.shape = (ub.size,)
			n = ub.size
			Gt = cvxmat(Aineq,tc='d')
			G = cvxmat([Gt,speye(n)],tc='d')	
			h = cvxmat(np.hstack([bineq,ub]),tc='d')
	else:
		if lb.ndim != 1:
                	lb.shape = (lb.size,)
		if ub is None:
			n = lb.size
			Gt = cvxmat(Aineq,tc='d')
                        G = cvxmat([Gt,-speye(n)],tc='d')
                        h = cvxmat(np.hstack([bineq,-lb]),tc='d')
		else:
			if ub.ndim != 1:
                                ub.shape = (ub.size,)
			n = lb.size
			Gt = cvxmat(Aineq,tc='d')
                        G = cvxmat([Gt,speye(n),-speye(n)],tc='d')
                        h = cvxmat(np.hstack([bineq,ub,-lb]),tc='d')
    return P, q, G, h, A, b 

def speye(n):

#    Create a sparse identity matrix

    r = range(n)
    return cvxspmatrix(1.0, r, r)
