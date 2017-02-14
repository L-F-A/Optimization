import numpy as np
import scipy as sp
import warnings


def NewtonAdaptStep(func,dfunc,ddfunc,w,tol,iteMax,args=None):
#Newton method with adaptive stepsize

	eta=1.
	lam=1e-10
	n=len(w)
	Imat=lam*np.identity(n)
	Delta=np.ones_like(w)

	if args is None:
		ite=0
                while np.max(Delta)>tol and ite<=iteMax:
                        if ite==iteMax:
                                warnings.warn('Maximum number of iterations reached: no convergence')
                                break
                        ite+=1
                        #Solving the equation (ddfunc+lam*I)Delta=-dfunc
                        #Using Cholesky decomposition
                        B=ddfunc(w)+Imat
                        C=sp.linalg.cho_factor(B,check_finite=False)
                        Delta=sp.linalg.cho_solve((C,False),-dfunc(w),check_finite=False)

                        ite1=0
                        while ite1==0 or eta*np.max(Delta)>tol/1000.:

                                ww=w+eta*Delta

                                if func(ww)<=func(w):
                                        w=ww.copy()
                                        eta=np.sqrt(eta)
                                        ite1=1
                                else:
                                        eta=0.1*eta

	else:
		ite=0
		while np.max(Delta)>tol and ite<=iteMax:
			if ite==iteMax:
				warnings.warn('Maximum number of iterations reached: no convergence')
				break
			ite+=1
			#Solving the equation (ddfunc+lam*I)Delta=-dfunc
                        #Using Cholesky decomposition
                        B=ddfunc(w,*args)+Imat
                        C=sp.linalg.cho_factor(B,check_finite=False)
                        Delta=sp.linalg.cho_solve((C,False),-dfunc(w,*args),check_finite=False)		

			ite1=0
			while ite1==0 or eta*np.max(Delta)>tol/1000.:

                                ww=w+eta*Delta

                                if func(ww,*args)<=func(w,*args):
                                        w=ww.copy()
                                        eta=np.sqrt(eta)
                                        ite1=1
                                else:
                                        eta=0.1*eta

	return w,ite
				

def NewtonAdaptDamp(func,dfunc,ddfunc,w,tol,iteMax,args=None):
#Newton method with adaptive damping lambda (Levenberg-Marquardt)	
	lam=1e-10
        n=len(w)
        Imat=np.identity(n)
        Delta=np.ones_like(w)

	if args is None:

		ite=0
                stop_crit=0
                while stop_crit==0:
			ite+=1
			#Solving the equation (ddfunc+lam*I)Delta=-dfunc
                        #Using Cholesky decomposition
                        B=ddfunc(w)+lam*Imat
                        C=sp.linalg.cho_factor(B,check_finite=False)
                        Delta=sp.linalg.cho_solve((C,False),-dfunc(w),check_finite=False)

                        if func(w+Delta) <= func(w):
                                w=w+Delta
                                lam=0.2*lam
                        else:
                                lam=10.*lam

                        if lam<1. and np.max(Delta)<tol:
                                stop_crit=1
			elif ite==iteMax:
				warnings.warn('Max number of iterations reached: no convergence')
				stop_crit=1

	else:
		ite=0
		stop_crit=0
		while stop_crit==0:
			ite+=1
			#Solving the equation (ddfunc+lam*I)Delta=-dfunc
                        #Using Cholesky decomposition
                        B=ddfunc(w,*args)+lam*Imat
                        C=sp.linalg.cho_factor(B,check_finite=False)
                        Delta=sp.linalg.cho_solve((C,False),-dfunc(w,*args),check_finite=False)

                        if func(w+Delta,*args) <= func(w,*args):
                                w=w+Delta
                                lam=0.2*lam
                        else:
                                lam=10.*lam

                        if lam<1. and np.max(Delta)<tol:
                                stop_crit=1
			elif ite==iteMax:
                                warnings.warn('Max number of iterations reached: no convergence')
                                stop_crit=1

	
	return w,ite
