import numpy as np
import warnings

#########################################################################################################
#					   IMPORTANT NOTE						#
#Every method splits cases of function with or without supplementary arguments at the very beginning	# 
#as to avoid if-else operations for it at each iteration in the while loops. This means just		# 
#copy-pasting the code twice in each method a changing func(w) to func(w,*args) etc. Perhaps not as neat# 
#as possible, but might save computation time.								#
#########################################################################################################

def GradDesc0(dfunc,w,eta,tol,tol_rel,iteMax,args=None):
######################################################################################################### 
#                            Vanilla gradient descent with constant stepsize	                        #
#                                                                                                       #
#       INPUTS:                                                                                         #
#               dfunc   : Function giving the derivative of func to be minimized                        #
#               w       : Initial value for w                                                           #
#               eta     : Initial stepsize                                                              #
#               tol     : Absolute tolerance                                                            #
#               tol_rel : Relative tolerance                                                            #
#               args    : Tuple containing all other argument that func and dfunc take                  #
#                                                                                                       #
#       OUPUTS:                                                                                         #
#               w       : Solution for w where func is minimum                                          #
#               ite     : How many iterations is took                                                   #
#########################################################################################################

	ite=0
        ite_conv=0
	ite_follow=0
        err=1.
        err_rel=1.

	if args is None:#split function with or without supplementary arguments at the very beginning as 
			#to avoid if operation for it at each iteration in the while loop

		while (err>tol or err_rel>tol_rel) and (ite_conv<5):#tol conditions respected for 5 
								    #iterations in a row

			if ite==iteMax:
				warnings.warn('Maximum number of iterations reached: no convergence')
				break

			ite+=1
			g=dfunc(w)
			w0=w.copy()
                	w=w-eta*g

                	err=np.linalg.norm(w-w0)
                	err_rel=err/np.linalg.norm(w0+np.finfo(float).eps)

                	if err<tol or err_rel<tol_rel:#Making certain that it is really for 5 iterations
						      #in a row and not 5 non consequtive ones

				if ite_follow==0:
					ite_follow=ite
                        		ite_conv+=1
				elif ite==ite_follow+1:
					ite_follow=ite
					ite_conv+=1
				else:
					ite_follow=0
					ite_conv=0
		
	else:
		while (err>tol or err_rel>tol_rel) and (ite_conv<5):#tol conditions respected for 5 
								    #iterations in a row

			if ite==iteMax:
                                warnings.warn('Maximum number of iterations reached: no convergence')
                                break

                        ite+=1
                        g=dfunc(w,*args)
                        w0=w.copy()
                        w=w-eta*g

			err=np.linalg.norm(w-w0)
                        err_rel=err/np.linalg.norm(w0+np.finfo(float).eps)

                        if err<tol or err_rel<tol_rel:#Making certain that it is really for 5 iterations
                                                      #in a row and not 5 non consequtive ones
                                
				if ite_follow==0:
                                        ite_follow=ite
                                        ite_conv+=1
                                elif ite==ite_follow+1:
                                        ite_follow=ite
                                        ite_conv+=1
                                else:
                                        ite_follow=0
                                        ite_conv=0

	return w,ite
	
def GradDesc(func,dfunc,w,eta,tol,tol_rel,iteMax,args=None):
######################################################################################################### 
#				Gradient descent with stepsize adaptation				#
# 	  		  see Marc Toussaint U Stuttgart: Intro to Optimization				#
#    https://ipvs.informatik.uni-stuttgart.de/mlr/marc/teaching/13-Optimization/02-gradientMethods.pdf	#
#													#
#	INPUTS:												#
#		func    : Function to be minimized							#
#		dfunc   : Function giving the derivative of func					#
#		w       : Initial value for w								#
#		eta     : Initial stepsize								#
#		tol     : Absolute tolerance								#
#		tol_rel : Relative tolerance								#
#		args    : Tuple containing all other argument that func and dfunc take			#
#													#
#	OUPUTS:												#
#		w	: Solution for w where func is minimum						#
#		ite	: How many iterations is took							#
#########################################################################################################

	ite=0
	ite_conv=0
	ite_follow=0
	err=1.
	err_rel=1.

	if args is None:
		
		while (err>tol or err_rel>tol_rel) and (ite_conv<5):#tol conditions respected for 5 
								    #iterations in a row

			if ite==iteMax:
                                warnings.warn('Maximum number of iterations reached: no convergence')
                                break

			ite+=1
			
			g=dfunc(w)
                        y=w-eta*g/np.linalg.norm(g)
                        w0=w.copy()
                        if func(y) <= func(w):
                                w=y.copy()
                                eta=1.2*eta     #1.2 Magic number from Marc Toussaint U Stuttgart: Intro 
						#to Optimization
                        else:
                                eta=0.5*eta     #0.5 Magic number from Marc Toussaint U Stuttgart: Intro 
						#to Optimization

			err=np.linalg.norm(y-w0)
                	err_rel=err/np.linalg.norm(w0+np.finfo(float).eps)

			if err<tol or err_rel<tol_rel:#Making certain that it is really for 5 iterations
                                                      #in a row and not 5 non consequtive ones

                                if ite_follow==0:
                                        ite_follow=ite
                                        ite_conv+=1
                                elif ite==ite_follow+1:
                                        ite_follow=ite
                                        ite_conv+=1
                                else:
                                        ite_follow=0
                                        ite_conv=0

	else:

		while (err>tol or err_rel>tol_rel) and (ite_conv<5):#tol conditions respected for 5 
                                                                    #iterations in a row

			if ite==iteMax:
                                warnings.warn('Maximum number of iterations reached: no convergence')
                                break

                        ite+=1

                        g=dfunc(w,*args)
                        y=w-eta*g/np.linalg.norm(g)
                        w0=w.copy()
                        if func(y,*args) <= func(w,*args):
                                w=y.copy()
                                eta=1.2*eta     #1.2 Magic number from Marc Toussaint U Stuttgart: Intro 
                                                #to Optimization
                        else:
                                eta=0.5*eta     #0.5 Magic number from Marc Toussaint U Stuttgart: Intro 
                                                #to Optimization

                        err=np.linalg.norm(y-w0)
                        err_rel=err/np.linalg.norm(w0+np.finfo(float).eps)

			if err<tol or err_rel<tol_rel:#Making certain that it is really for 5 iterations
                                                      #in a row and not 5 non consequtive ones
                                if ite_follow==0:
                                        ite_follow=ite
                                        ite_conv+=1
                                elif ite==ite_follow+1:
                                        ite_follow=ite
                                        ite_conv+=1
                                else:
                                        ite_follow=0
                                        ite_conv=0                

	return w,ite	



def Rprop(dfunc,w,eta,tol,tol_rel,iteMax,args=None):
######################################################################################################### 
#                                   Resilient Back Propagation		                                #
#                         see Marc Toussaint U Stuttgart: Intro to Optimization                         #
#    https://ipvs.informatik.uni-stuttgart.de/mlr/marc/teaching/13-Optimization/02-gradientMethods.pdf  #
#                                                                                                       #
#       INPUTS:                                                                                         #
#               dfunc   : Function giving the derivative of the function to be minimized                #
#               w       : Initial value for w                                                           #
#               eta     : Initial stepsize                                                              #
#               tol     : Absolute tolerance                                                            #
#               tol_rel : Relative tolerance                                                            #
#               args    : Tuple containing all other argument that func and dfunc take                  #
#                                                                                                       #
#       OUPUTS:                                                                                         #
#               w       : Solution for w where func is minimum                                          #
#               ite     : How many iterations is took                                                   #
#########################################################################################################
        ite=0
        ite_conv=0
	ite_follow=0
	n=len(w)
        err=1.
        err_rel=1.

	g0=np.zeros(n)
	eta=eta*np.ones(n)

	if args is None:
		while (err>tol or err_rel>tol_rel) and (ite_conv<5):#tol conditions respected for 5 
								     #iterations in a row
			if ite==iteMax:
                                warnings.warn('Maximum number of iterations reached: no convergence')
                                break

			ite+=1
			g=dfunc(w)

			w0=w.copy()
                	for i in range(n):
                        	if g[i]*g0[i]>0:
                                	eta[i]=1.2*eta[i]
                                	w[i]=w[i]-eta[i]*np.sign(g[i])
                                	g0[i]=g[i]
                        	elif g[i]*g0[i]<0.:
                                	eta[i]=0.5*eta[i]
                                	w[i]=w[i]-eta[i]*np.sign(g[i])
                                	g0[i]=0.
                        	else:
                                	w[i]=w[i]-eta[i]*np.sign(g[i])
                                	g0[i]=g[i]

			err=np.linalg.norm(w-w0)
			err_rel=err/np.linalg.norm(w0+np.finfo(float).eps)

			if err<tol or err_rel<tol_rel:#Making certain that it is really for 5 iterations
                                                      #in a row and not 5 non consequtive ones

                                if ite_follow==0:
                                        ite_follow=ite
                                        ite_conv+=1
                                elif ite==ite_follow+1:
                                        ite_follow=ite
                                        ite_conv+=1
                                else:
                                        ite_follow=0
                                        ite_conv=0
		
	else:

		while (err>tol or err_rel>tol_rel) and (ite_conv<5):#tol conditions respected for 5 
                                                                     #iterations in a row
                        if ite==iteMax:
                                warnings.warn('Maximum number of iterations reached: no convergence')
                                break

                        ite+=1
                        g=dfunc(w,*args)

                        w0=w.copy()
                        for i in range(n):
                                if g[i]*g0[i]>0:
                                        eta[i]=1.2*eta[i]
                                        w[i]=w[i]-eta[i]*np.sign(g[i])
                                        g0[i]=g[i]
                                elif g[i]*g0[i]<0.:
                                        eta[i]=0.5*eta[i]
                                        w[i]=w[i]-eta[i]*np.sign(g[i])
                                        g0[i]=0.
                                else:
                                        w[i]=w[i]-eta[i]*np.sign(g[i])
                                        g0[i]=g[i]

                        err=np.linalg.norm(w-w0)
                        err_rel=err/np.linalg.norm(w0+np.finfo(float).eps)

                        if err<tol or err_rel<tol_rel:#Making certain that it is really for 5 iterations
                                                      #in a row and not 5 non consequtive ones

                                if ite_follow==0:
                                        ite_follow=ite
                                        ite_conv+=1
                                elif ite==ite_follow+1:
                                        ite_follow=ite
                                        ite_conv+=1
                                else:
                                        ite_follow=0
                                        ite_conv=0
	
	return w,ite


def GradDescSteep(dfunc,w,eta,tol,tol_rel,iteMax,args=None):
#using stepsize eta_n from https://en.wikipedia.org/wiki/Gradient_descent
	err=1.
	err_rel=1.
	ite=0
	ite_conv=0
	ite_follow=0

	w_00=w.copy()

	if args is None:
		g_00=dfunc(w)

		while (err>tol or err_rel>tol_rel) and (ite_conv<5):

			if ite==iteMax:
                                warnings.warn('Maximum number of iterations reached: no convergence')
                                break

			ite+=1
			g0=dfunc(w)

			if ite==0:
                        	w=w-eta*g0
			else:
                        	delta=np.dot(w-w_00,g0-g_00)/np.dot(g0-g_00,g0-g_00)
                        	w_00=w.copy()
                        	w=w-delta*g0
                        	g_00=g0.copy()

			err=np.linalg.norm(w-w_00)
                	err_rel=err/np.linalg.norm(w_00+np.finfo(float).eps)

			if err<tol or err_rel<tol_rel:#Making certain that it is really for 5 iterations
                                                      #in a row and not 5 non consequtive ones
                        	if ite_follow==0:
                                	ite_follow=ite
                                	ite_conv+=1
                        	elif ite==ite_follow+1:
                                	ite_follow=ite
                                	ite_conv+=1
                        	else:
                                	ite_follow=0
                                	ite_conv=0			

	else:
		g_00=dfunc(w,*args)

		while (err>tol or err_rel>tol_rel) and (ite_conv<5):

			if ite==iteMax:
                                warnings.warn('Maximum number of iterations reached: no convergence')
                                break

                        ite+=1
			g0=dfunc(w,*args)
                        if ite==1:
                                w=w-eta*g0
                        else:
                                delta=np.dot(w-w_00,g0-g_00)/np.dot(g0-g_00,g0-g_00)
                                w_00=w.copy()
                                w=w-delta*g0
                                g_00=g0.copy()

                        err=np.linalg.norm(w-w_00)
                        err_rel=err/np.linalg.norm(w_00+np.finfo(float).eps)

                        if err<tol or err_rel<tol_rel:#Making certain that it is really for 5 iterations
                                                      #in a row and not 5 non consequtive ones
                                if ite_follow==0:
                                        ite_follow=ite
                                        ite_conv+=1
                                elif ite==ite_follow+1:
                                        ite_follow=ite
                                        ite_conv+=1
                                else:
                                        ite_follow=0
                                        ite_conv=0


	return w,ite
