import numpy as np
import warnings

#################################################################################################
#		   Various implementations of stochastic gradient descent			#
#												#
#	 Louis-Francois Arsenault, Columbia Universisty (2013-2017), la2518@columbia.edu	#
#################################################################################################
#												#
#					IMPORTANT NOTES:					#
#												#
#					      (1)						#
#    There is a lot of copy-pasting of the same code over an over rather than the possibility	# 
#    of one version with many if-else checking. I separate the problem at the very beginning 	#
#    by checking if the gradient takes zero, one or multiple extra parameters throught the 	#
#    variable args in addition to w, the variable we are looking for, the data X and y. I then  #
#    check if ones want to use one point stochastic gradient descent or mini-batchs. By doing so# 
#    at the beginning I avoid testing at every iteration the nature of the problem which should #
#    save some time, although not that much probably.						#
#												#
#					      (2)						#
#    The methods AdaGradDecay, AdaDelta, Adam and AdaMax are directly implemented following the	# 
#    Matlab implementations of Mikhail Pak at https://github.com/mp4096/adawhatever		#
#################################################################################################


def StochGrad(dfunc,w,X,y,Nlearn,args,damp,eta=1e-2,tol=1e-6,epochMax=5,Nbatch=1,WarnM=False):
#################################################################################################
#		Stochastic Gradient Descent with learning et0/(1+et0*damp*t)			#
#			For details on that stepsize, see:	 				#
#		http://cilvr.cs.nyu.edu/diglib/lsml/bottou-sgd-tricks-2012.pdf 			#
#												#
#	I did not put the possibility of fixed learning rate because it does not respect the 	#
#		theoretical conditions, but of course modifying the code is trivial.		#
#################################################################################################
#												#
#	INPUTS:											#
#		dfunc    : Function returning the value of the gradient				#				
#		w        : First guess of the variable being optimized				#
#		X        : Features matrix of the learning set					#	
#		y        : Vector of the values of the learning set				#
#		Nlearn   : How many examples in the learning set				#	
#		args     : Contains all the extra parameters that dfunc takes in addition of X	# 
#			   and y, can be be None, a scalar or a tuple				#
#		damp     : The constant used in the stepsize, usually lambda the 		#	
#			   regularization constant						#
#		eta      : The starting stepsize						#
#		tol      : Tolerance on the difference of two iterations of w that will trigger #
#			   convergence
#		epochMax : Maximum number of epochs possible					#
#		Nbatch   : Nbatch=1 for sgd with one example, Nbatch = k for mini-batch of size	# 
#			   as close as possible to k						#
#		WarnM    : On screen warning message of no convergence after epochMax iterations#
#												#
#	OUTPUTS:										#
#		w	 : Final values of the vector w						#
#		ep	 : Number of full epoch done						#
#		ite	 : Total number of iterations						#
#		mess	 : Warning message regarding convergence				#
#################################################################################################

	err=1.
	idx=range(Nlearn)
	ep=0
	ite=0
	w0=w.copy()
	eta0=eta
	
	mess='Calculation converged before maximum number of epochs'

	if (type(args) is float) or (args is None):

		if Nbatch==1:

			while err>tol:

				if ep==epochMax:
                        		mess='No convergenge after '+str(epochMax)+' epochs'
					if WarnM is True:
                        			warnings.warn(mess)
                                	break

				ep+=1#Which epoch

				np.random.shuffle(idx)#Create the order for one epoch

				for r_sp in range(Nlearn):#loop over all the data
	
					ite+=1
					w=w0-eta*dfunc(w0,X[r_sp,:],y[r_sp],args)
					err=np.linalg.norm(w-w0)

					if err<tol and ep>1:
						break

					w0=w.copy()
					
					eta=eta0/(1.+eta0*damp*ite)

		else:#Mini-batch
		
			while err>tol:

                        	if ep==epochMax:
                                	mess='No convergenge after '+str(epochMax)+' epochs'
					if WarnM is True:
                                		warnings.warn(mess)
                                	break

                        	ep+=1#Which epoch

                        	np.random.shuffle(idx)#Create the order for one epoch
				sp=np.array_split(idx,np.floor( Nlearn/float(Nbatch) ) )

				for r_sp in range(len(sp)):#loop over all the mini-batch
					ite+=1
					idData=sp[r_sp]
					nn=len(idData)
				
                                	w=w0-eta*dfunc(w0,X[idData,:],y[idData],args)/float(nn)
                                	err=np.linalg.norm(w-w0)

					if err<tol and ep>1:
                                        	break

                                	w0=w.copy()
					
                                	eta=eta0/(1.+eta0*damp*ite)		

	else:
	
		if Nbatch==1:

			while err>tol:

				if ep==epochMax:
                        		mess='No convergenge after '+str(epochMax)+' epochs'
					if WarnM is True:
                        			warnings.warn(mess)
                                	break

				ep+=1#Which epoch

				np.random.shuffle(idx)#Create the order for one epoch

				for r_sp in range(Nlearn):#loop over all the data
	
					ite+=1
					w=w0-eta*dfunc(w0,X[r_sp,:],y[r_sp],*args)
					err=np.linalg.norm(w-w0)

					if err<tol and ep>1:
						break

					w0=w.copy()
					
					eta=eta0/(1.+eta0*damp*ite)

		else:#Mini-batch
		
			while err>tol:

                        	if ep==epochMax:
                                	mess='No convergenge after '+str(epochMax)+' epochs'
					if WarnM is True:
                                		warnings.warn(mess)
                                	break

                        	ep+=1#Which epoch

                        	np.random.shuffle(idx)#Create the order for one epoch
				sp=np.array_split(idx,np.floor( Nlearn/float(Nbatch) ) )

				for r_sp in range(len(sp)):#loop over all the mini-batch
					ite+=1
					idData=sp[r_sp]
					nn=len(idData)
				
                                	w=w0-eta*dfunc(w0,X[idData,:],y[idaDta],*args)/float(nn)
                                	err=np.linalg.norm(w-w0)

					if err<tol and ep>1:
                                        	break

                                	w0=w.copy()
					
                                	eta=eta0/(1.+eta0*damp*ite)

	return w,ep,ite,mess


def StochAdaGrad(dfunc,w,X,y,Nlearn,args,eta=1e-2,fudge=1e-10,tol=1e-6,epochMax=5,Nbatch=1,WarnM=False):
#################################################################################################
#                     Stochastic Gradient Descent with AdaGrad stepsize				#
#												#
#		         General implementation idea taken from 				#
#		 https://github.com/benbo/adagrad/blob/master/adagrad.py 			#
#					    and							#
#https://xcorr.net/2014/01/23/adagrad-eliminating-learning-rates-in-stochastic-gradient-descent/#
#												#
#################################################################################################
#                                                                                               #
#       INPUTS:                                                                                 #
#               dfunc    : Function returning the value of the gradient                         #
#               w        : First guess of the variable being optimized                          #
#               X        : Features matrix of the learning set			                #       
#               y        : Vector of the values of the learning set                             #
#               Nlearn   : How many examples in the learning set                                #       
#               args     : Contains all the extra parameters that dfunc takes in addition of X  # 
#                          and y, can be be None, a scalar or a tuple                           #
#               eta      : Constant multiplicative stepsize	                                #
#		fudge	 : Constant value for numerical stability				#
#               tol      : Tolerance on the difference of two iterations of w that will trigger #
#			   convergence       							#
#               epochMax : Maximum number of epochs possible                                    #
#               Nbatch   : Nbatch=1 for sgd with one example, Nbatch = k for mini-batch of size # 
#                          as close as possible to k                                            #
#               WarnM    : On screen warning message of no convergence after epochMax iterations#
#                                                                                               #
#       OUTPUTS:                                                                                #
#               w        : Final values of the vector w                                         #
#               ep       : Number of full epoch done                                            #
#               ite      : Total number of iterations                                           #
#               mess     : Warning message regarding convergence                                #
#################################################################################################

	err=1.
	idx=range(Nlearn)
	ep=0
	ite=0
	w0=w.copy()
	gi=np.zeros(len(w))

	mess='Calculation converged before maximum number of epochs'

	if (type(args) is float) or (args is None):

		if Nbatch==1:

			while err>tol:

				if ep==epochMax:
                        		mess='No convergenge after '+str(epochMax)+' epochs'
					if WarnM is True:
                        			warnings.warn(mess)
                                	break

				ep+=1#Which epoch

				np.random.shuffle(idx)#Create the order for one epoch

				for r_sp in range(Nlearn):#loop over all the data
	
					ite+=1
					grad=dfunc(w0,X[r_sp,:],y[r_sp],args)
					gi+=grad**2
				
					w=w0-eta*grad/(fudge+np.sqrt(gi))
					err=np.linalg.norm(w-w0)

					if err<tol and ep>1:
						break

					w0=w.copy()

		else:#Mini-batch
		
			while err>tol:

                        	if ep==epochMax:
                                	mess='No convergenge after '+str(epochMax)+' epochs'
					if WarnM is True:
                                		warnings.warn(mess)
                                	break

                        	ep+=1#Which epoch

                        	np.random.shuffle(idx)#Create the order for one epoch
				sp=np.array_split(idx,np.floor( Nlearn/float(Nbatch) ) )

				for r_sp in range(len(sp)):#loop over all the mini-batch
					ite+=1
					idData=sp[r_sp]
					nn=len(idData)
					
					grad=dfunc(w0,X[idData,:],y[idData],args)/float(nn)
					gi+=grad**2
				
                                	w=w0-eta*grad/(fudge+np.sqrt(gi))
                                	err=np.linalg.norm(w-w0)

					if err<tol and ep>1:
                                        	break

                                	w0=w.copy()

	else:
	
		if Nbatch==1:

			while err>tol:

				if ep==epochMax:
                        		mess='No convergenge after '+str(epochMax)+' epochs'
					if WarnM is True:
                        			warnings.warn(mess)
                                	break

				ep+=1#Which epoch

				np.random.shuffle(idx)#Create the order for one epoch

				for r_sp in range(Nlearn):#loop over all the data
	
					ite+=1
					grad=dfunc(w0,X[r_sp,:],y[r_sp],*args)
					gi+=grad**2
							
					w=w0-eta*grad/(fudge+np.sqrt(gi))
					err=np.linalg.norm(w-w0)

					if err<tol and ep>1:
						break

					w0=w.copy()

		else:#Mini-batch
		
			while err>tol:

                        	if ep==epochMax:
                                	mess='No convergenge after '+str(epochMax)+' epochs'
					if WarnM is True:
                                		warnings.warn(mess)
                                	break

                        	ep+=1#Which epoch

                        	np.random.shuffle(idx)#Create the order for one epoch
				sp=np.array_split(idx,np.floor( Nlearn/float(Nbatch) ) )

				for r_sp in range(len(sp)):#loop over all the mini-batch
					ite+=1
					idData=sp[r_sp]
					nn=len(idData)
					grad=dfunc(w0,X[idData,:],y[idaDta],*args)/float(nn)
					gi+=grad**2
				
                                	w=w0-eta*grad/(fudge+np.sqrt(gi))
                                	err=np.linalg.norm(w-w0)

					if err<tol and ep>1:
                                        	break

                                	w0=w.copy()

	return w,ep,ite,mess

def StochAdaGradDecay(dfunc,w,X,y,Nlearn,args,eta=1e-2,beta=1e-2,fudge=1e-10,tol=1e-6,epochMax=5,Nbatch=1,WarnM=False):
#################################################################################################
#                   Stochastic Gradient Descent with AdaGrad Decay learning rate		#
#												#
#		             General implementation idea taken from 				#
#	         https://github.com/mp4096/adawhatever/blob/master/AdaGradDecay.m		#
#################################################################################################
#                                                                                               #
#       INPUTS:                                                                                 #
#               dfunc    : Function returning the value of the gradient                         #
#               w        : First guess of the variable being optimized                          #
#               X        : Features matrix of the learning set			                #       
#               y        : Vector of the values of the learning set                             #
#               Nlearn   : How many examples in the learning set                                #       
#               args     : Contains all the extra parameters that dfunc takes in addition of X  # 
#                          and y, can be be None, a scalar or a tuple                           #
#               eta      : Constant multiplicative stepsize	                                #
#		beta     : Decay rate for historical gradients					#
#		fudge	 : Constant value for numerical stability				#
#               tol      : Tolerance on the difference of two iterations of w that will trigger #
#			   convergence							        #
#               epochMax : Maximum number of epochs possible                                    #
#               Nbatch   : Nbatch=1 for sgd with one example, Nbatch = k for mini-batch of size # 
#                          as close as possible to k                                            #
#               WarnM    : On screen warning message of no convergence after epochMax iterations#
#                                                                                               #
#       OUTPUTS:                                                                                #
#               w        : Final values of the vector w                                         #
#               ep       : Number of full epoch done                                            #
#               ite      : Total number of iterations                                           #
#               mess     : Warning message regarding convergence                                #
#################################################################################################

	err=1.
	idx=range(Nlearn)
	ep=0
	ite=0
	w0=w.copy()
	gi=np.zeros(len(w))	

	mess='Calculation converged before maximum number of epochs'

	if (type(args) is float) or (args is None):
		
		if Nbatch==1:

			while err>tol:

				if ep==epochMax:
                        		mess='No convergenge after '+str(epochMax)+' epochs'
					if WarnM is True:
                        			warnings.warn(mess)
                                	break

				ep+=1#Which epoch

				np.random.shuffle(idx)#Create the order for one epoch

				for r_sp in range(Nlearn):#loop over all the data
	
					ite+=1
					grad=dfunc(w0,X[r_sp,:],y[r_sp],args)
					if ite==0:
						gi+=grad**2
					else:
						gi=beta*gi + (1.-beta)*grad**2
				
					w=w0-eta*grad/(fudge+np.sqrt(gi))
					err=np.linalg.norm(w-w0)

					if err<tol and ep>1:
						break

					w0=w.copy()

		else:#Mini-batch
		
			while err>tol:

                        	if ep==epochMax:
                                	mess='No convergenge after '+str(epochMax)+' epochs'
					if WarnM is True:
                                		warnings.warn(mess)
                                	break

                        	ep+=1#Which epoch

                        	np.random.shuffle(idx)#Create the order for one epoch
				sp=np.array_split(idx,np.floor( Nlearn/float(Nbatch) ) )

				for r_sp in range(len(sp)):#loop over all the mini-batch
					ite+=1
					idData=sp[r_sp]
					nn=len(idData)
					
					grad=dfunc(w0,X[idData,:],y[idData],args)/float(nn)
					if ite==0:
                                                gi+=grad**2
                                        else:
                                                gi=beta*gi + (1.-beta)*grad**2
				
                                	w=w0-eta*grad/(fudge+np.sqrt(gi))
                                	err=np.linalg.norm(w-w0)

					if err<tol and ep>1:
                                        	break

                                	w0=w.copy()

	else:
	
		if Nbatch==1:

			while err>tol:

				if ep==epochMax:
                        		mess='No convergenge after '+str(epochMax)+' epochs'
					if WarnM is True:
                        			warnings.warn(mess)
                                	break

				ep+=1#Which epoch

				np.random.shuffle(idx)#Create the order for one epoch

				for r_sp in range(Nlearn):#loop over all the data
	
					ite+=1
					grad=dfunc(w0,X[r_sp,:],y[r_sp],*args)
					if ite==0:
                                                gi+=grad**2
                                        else:
                                                gi=beta*gi + (1.-beta)*grad**2
							
					w=w0-eta*grad/(fudge+np.sqrt(gi))
					err=np.linalg.norm(w-w0)

					if err<tol and ep>1:
						break

					w0=w.copy()

		else:#Mini-batch
		
			while err>tol:

                        	if ep==epochMax:
                                	mess='No convergenge after '+str(epochMax)+' epochs'
					if WarnM is True:
                                		warnings.warn(mess)
                                	break

                        	ep+=1#Which epoch

                        	np.random.shuffle(idx)#Create the order for one epoch
				sp=np.array_split(idx,np.floor( Nlearn/float(Nbatch) ) )

				for r_sp in range(len(sp)):#loop over all the mini-batch
					ite+=1
					idData=sp[r_sp]
					nn=len(idData)
					grad=dfunc(w0,X[idData,:],y[idaDta],*args)/float(nn)
					if ite==0:
                                                gi+=grad**2
                                        else:
                                                gi=beta*gi + (1.-beta)*grad**2
				
                                	w=w0-eta*grad/(fudge+np.sqrt(gi))
                                	err=np.linalg.norm(w-w0)

					if err<tol and ep>1:
                                        	break

                                	w0=w.copy()

	return w,ep,ite,mess

def StochAdaDelta(dfunc,w,X,y,Nlearn,args,eta=1e-2,beta=1e-2,fudge=1e-10,tol=1e-6,epochMax=5,Nbatch=1,WarnM=False):
#################################################################################################
#                   Stochastic Gradient Descent with AdaDelta learning rate			#
#												#
#		             General implementation idea taken from 				#
#	           https://github.com/mp4096/adawhatever/blob/master/Adadelta.m			#
#			    Method developped in ArXiv:1212.5701v1 				#
#################################################################################################
#                                                                                               #
#       INPUTS:                                                                                 #
#               dfunc    : Function returning the value of the gradient                         #
#               w        : First guess of the variable being optimized                          #
#               X        : Features matrix of the learning set			                #       
#               y        : Vector of the values of the learning set                             #
#               Nlearn   : How many examples in the learning set                                #       
#               args     : Contains all the extra parameters that dfunc takes in addition of X  # 
#                          and y, can be be None, a scalar or a tuple                           #
#               eta      : Constant multiplicative stepsize	                                #
#		beta     : Decay rate for moving average					#
#		fudge	 : Constant value for numerical stability				#
#               tol      : Tolerance on the difference of two iterations of w that will trigger #
#			   convergence							        #
#               epochMax : Maximum number of epochs possible                                    #
#               Nbatch   : Nbatch=1 for sgd with one example, Nbatch = k for mini-batch of size # 
#                          as close as possible to k                                            #
#               WarnM    : On screen warning message of no convergence after epochMax iterations#
#                                                                                               #
#       OUTPUTS:                                                                                #
#               w        : Final values of the vector w                                         #
#               ep       : Number of full epoch done                                            #
#               ite      : Total number of iterations                                           #
#               mess     : Warning message regarding convergence                                #
#################################################################################################

	err=1.
	idx=range(Nlearn)
	ep=0
	ite=0
	w0=w.copy()
	accG=np.zeros(len(w))
	accD=np.zeros(len(w))	

	mess='Calculation converged before maximum number of epochs'

	if (type(args) is float) or (args is None):
		
		if Nbatch==1:

			while err>tol:

				if ep==epochMax:
                        		mess='No convergenge after '+str(epochMax)+' epochs'
					if WarnM is True:
                        			warnings.warn(mess)
                                	break

				ep+=1#Which epoch

				np.random.shuffle(idx)#Create the order for one epoch

				for r_sp in range(Nlearn):#loop over all the data
	
					ite+=1
					grad=dfunc(w0,X[r_sp,:],y[r_sp],args)
					
					accG=beta*accG + (1.-beta)*grad**2
					gradMod=-(np.sqrt(accD+fudge)/np.sqrt(accG+fudge))*grad
					accD=beta*accD + (1.-beta)*gradMod**2

					w=w0+gradMod
					err=np.linalg.norm(w-w0)

					if err<tol and ep>1:
						break

					w0=w.copy()

		else:#Mini-batch
		
			while err>tol:

                        	if ep==epochMax:
                                	mess='No convergenge after '+str(epochMax)+' epochs'
					if WarnM is True:
                                		warnings.warn(mess)
                                	break

                        	ep+=1#Which epoch

                        	np.random.shuffle(idx)#Create the order for one epoch
				sp=np.array_split(idx,np.floor( Nlearn/float(Nbatch) ) )

				for r_sp in range(len(sp)):#loop over all the mini-batch
					ite+=1
					idData=sp[r_sp]
					nn=len(idData)
					
					grad=dfunc(w0,X[idData,:],y[idData],args)/float(nn)

					accG=beta*accG + (1.-beta)*grad**2
                                        gradMod=-(np.sqrt(accD+fudge)/np.sqrt(accG+fudge))*grad
                                        accD=beta*accD + (1.-beta)*gradMod**2

                                        w=w0+gradMod 
                                	err=np.linalg.norm(w-w0)

					if err<tol and ep>1:
                                        	break

                                	w0=w.copy()

	else:
	
		if Nbatch==1:

			while err>tol:

				if ep==epochMax:
                        		mess='No convergenge after '+str(epochMax)+' epochs'
					if WarnM is True:
                        			warnings.warn(mess)
                                	break

				ep+=1#Which epoch

				np.random.shuffle(idx)#Create the order for one epoch

				for r_sp in range(Nlearn):#loop over all the data
	
					ite+=1
					grad=dfunc(w0,X[r_sp,:],y[r_sp],*args)

					accG=beta*accG + (1.-beta)*grad**2
                                        gradMod=-(np.sqrt(accD+fudge)/np.sqrt(accG+fudge))*grad
                                        accD=beta*accD + (1.-beta)*gradMod**2

                                        w=w0+gradMod 
					err=np.linalg.norm(w-w0)

					if err<tol and ep>1:
						break

					w0=w.copy()

		else:#Mini-batch
		
			while err>tol:

                        	if ep==epochMax:
                                	mess='No convergenge after '+str(epochMax)+' epochs'
					if WarnM is True:
                                		warnings.warn(mess)
                                	break

                        	ep+=1#Which epoch

                        	np.random.shuffle(idx)#Create the order for one epoch
				sp=np.array_split(idx,np.floor( Nlearn/float(Nbatch) ) )

				for r_sp in range(len(sp)):#loop over all the mini-batch
					ite+=1
					idData=sp[r_sp]
					nn=len(idData)
					grad=dfunc(w0,X[idData,:],y[idaDta],*args)/float(nn)

					accG=beta*accG + (1.-beta)*grad**2
                                        gradMod=-(np.sqrt(accD+fudge)/np.sqrt(accG+fudge))*grad
                                        accD=beta*accD + (1.-beta)*gradMod**2

                                        w=w0+gradMod 
                                	err=np.linalg.norm(w-w0)

					if err<tol and ep>1:
                                        	break

                                	w0=w.copy()

	return w,ep,ite,mess

def StochAdam(dfunc,w,X,y,Nlearn,args,eta=1e-2,beta1=1e-2,beta2=1e-2,fudge=1e-10,tol=1e-6,epochMax=5,Nbatch=1,WarnM=False):
#################################################################################################
#                   Stochastic Gradient Descent with AdaDelta step size				#
#												#
#		             General implementation idea taken from 				#
#	           https://github.com/mp4096/adawhatever/blob/master/Adadelta.m			#
#			     Method developped in ArXiv:1412.6980v8				#
#################################################################################################
#                                                                                               #
#       INPUTS:                                                                                 #
#               dfunc    : Function returning the value of the gradient                         #
#               w        : First guess of the variable being optimized                          #
#               X        : Features matrix of the learning set			                #       
#               y        : Vector of the values of the learning set                             #
#               Nlearn   : How many examples in the learning set                                #       
#               args     : Contains all the extra parameters that dfunc takes in addition of X  # 
#                          and y, can be be None, a scalar or a tuple                           #
#               eta      : Constant multiplicative step size	                                #
#		beta1    : Decay rate for 1st moment						#
#		beta2    : Decay rate for 2nd moment						#
#		fudge	 : Constant value for numerical stability				#
#               tol      : Tolerance on the difference of two iterations of w that will trigger #
#			   convergence							        #
#               epochMax : Maximum number of epochs possible                                    #
#               Nbatch   : Nbatch=1 for sgd with one example, Nbatch = k for mini-batch of size # 
#                          as close as possible to k                                            #
#               WarnM    : On screen warning message of no convergence after epochMax iterations#
#                                                                                               #
#       OUTPUTS:                                                                                #
#               w        : Final values of the vector w                                         #
#               ep       : Number of full epoch done                                            #
#               ite      : Total number of iterations                                           #
#               mess     : Warning message regarding convergence                                #
#################################################################################################

	err=1.
	idx=range(Nlearn)
	ep=0
	ite=0
	w0=w.copy()

	m=np.zeros(len(w))
	v=np.zeros(len(w))	

	mess='Calculation converged before maximum number of epochs'

	if (type(args) is float) or (args is None):
		
		if Nbatch==1:

			while err>tol:

				if ep==epochMax:
                        		mess='No convergenge after '+str(epochMax)+' epochs'
					if WarnM is True:
                        			warnings.warn(mess)
                                	break

				ep+=1#Which epoch

				np.random.shuffle(idx)#Create the order for one epoch

				for r_sp in range(Nlearn):#loop over all the data
	
					ite+=1
					grad=dfunc(w0,X[r_sp,:],y[r_sp],args)
					
					m=beta1*m+(1.-beta1)*grad
					v=beta2*v+(1.-beta2)*grad**2
					mhat=m/(1.-beta1**ite)
					vhat=v/(1.-beta2**ite)

					w=w0-eta*mhat/(np.sqrt(vhat)+fudge)
					err=np.linalg.norm(w-w0)

					if err<tol and ep>1:
						break

					w0=w.copy()

		else:#Mini-batch
		
			while err>tol:

                        	if ep==epochMax:
                                	mess='No convergenge after '+str(epochMax)+' epochs'
					if WarnM is True:
                                		warnings.warn(mess)
                                	break

                        	ep+=1#Which epoch

                        	np.random.shuffle(idx)#Create the order for one epoch
				sp=np.array_split(idx,np.floor( Nlearn/float(Nbatch) ) )

				for r_sp in range(len(sp)):#loop over all the mini-batch
					ite+=1
					idData=sp[r_sp]
					nn=len(idData)
					
					grad=dfunc(w0,X[idData,:],y[idData],args)/float(nn)

					m=beta1*m+(1.-beta1)*grad
                                        v=beta2*v+(1.-beta2)*grad**2
                                        mhat=m/(1.-beta1**ite)
                                        vhat=v/(1.-beta2**ite)

                                        w=w0-eta*mhat/(np.sqrt(vhat)+fudge)
                                	err=np.linalg.norm(w-w0)

					if err<tol and ep>1:
                                        	break

                                	w0=w.copy()

	else:
	
		if Nbatch==1:

			while err>tol:

				if ep==epochMax:
                        		mess='No convergenge after '+str(epochMax)+' epochs'
					if WarnM is True:
                        			warnings.warn(mess)
                                	break

				ep+=1#Which epoch

				np.random.shuffle(idx)#Create the order for one epoch

				for r_sp in range(Nlearn):#loop over all the data
	
					ite+=1
					grad=dfunc(w0,X[r_sp,:],y[r_sp],*args)

					m=beta1*m+(1.-beta1)*grad
                                        v=beta2*v+(1.-beta2)*grad**2
                                        mhat=m/(1.-beta1**ite)
                                        vhat=v/(1.-beta2**ite)

                                        w=w0-eta*mhat/(np.sqrt(vhat)+fudge)
					err=np.linalg.norm(w-w0)

					if err<tol and ep>1:
						break

					w0=w.copy()

		else:#Mini-batch
		
			while err>tol:

                        	if ep==epochMax:
                                	mess='No convergenge after '+str(epochMax)+' epochs'
					if WarnM is True:
                                		warnings.warn(mess)
                                	break

                        	ep+=1#Which epoch

                        	np.random.shuffle(idx)#Create the order for one epoch
				sp=np.array_split(idx,np.floor( Nlearn/float(Nbatch) ) )

				for r_sp in range(len(sp)):#loop over all the mini-batch
					ite+=1
					idData=sp[r_sp]
					nn=len(idData)
					grad=dfunc(w0,X[idData,:],y[idaDta],*args)/float(nn)

					m=beta1*m+(1.-beta1)*grad
                                        v=beta2*v+(1.-beta2)*grad**2
                                        mhat=m/(1.-beta1**ite)
                                        vhat=v/(1.-beta2**ite)

                                        w=w0-eta*mhat/(np.sqrt(vhat)+fudge)
                                	err=np.linalg.norm(w-w0)

					if err<tol and ep>1:
                                        	break

                                	w0=w.copy()

	return w,ep,ite,mess


def StochAdaMax(dfunc,w,X,y,Nlearn,args,eta=1e-2,beta1=1e-2,beta2=1e-2,tol=1e-6,epochMax=5,Nbatch=1,WarnM=False):
#################################################################################################
#                   Stochastic Gradient Descent with AdaMax step size				#
#												#
#		             General implementation idea taken from 				#
#		    https://github.com/mp4096/adawhatever/blob/master/Adamax.m			#
#			     Method developped in ArXiv:1412.6980v8				#
#################################################################################################
#                                                                                               #
#       INPUTS:                                                                                 #
#               dfunc    : Function returning the value of the gradient                         #
#               w        : First guess of the variable being optimized                          #
#               X        : Features matrix of the learning set			                #       
#               y        : Vector of the values of the learning set                             #
#               Nlearn   : How many examples in the learning set                                #       
#               args     : Contains all the extra parameters that dfunc takes in addition of X  # 
#                          and y, can be be None, a scalar or a tuple                           #
#               eta      : Constant multiplicative step size	                                #
#		beta1    : Decay rate for 1st moment						#
#		beta2    : Decay rate for weighted infinity norm				#
#               tol      : Tolerance on the difference of two iterations of w that will trigger #
#			   convergence							        #
#               epochMax : Maximum number of epochs possible                                    #
#               Nbatch   : Nbatch=1 for sgd with one example, Nbatch = k for mini-batch of size # 
#                          as close as possible to k                                            #
#               WarnM    : On screen warning message of no convergence after epochMax iterations#
#                                                                                               #
#       OUTPUTS:                                                                                #
#               w        : Final values of the vector w                                         #
#               ep       : Number of full epoch done                                            #
#               ite      : Total number of iterations                                           #
#               mess     : Warning message regarding convergence                                #
#################################################################################################

	err=1.
	idx=range(Nlearn)
	ep=0
	ite=0
	w0=w.copy()

	m=np.zeros(len(w))
	u=np.zeros(len(w))	

	mess='Calculation converged before maximum number of epochs'

	if (type(args) is float) or (args is None):
		
		if Nbatch==1:

			while err>tol:

				if ep==epochMax:
                        		mess='No convergenge after '+str(epochMax)+' epochs'
					if WarnM is True:
                        			warnings.warn(mess)
                                	break

				ep+=1#Which epoch

				np.random.shuffle(idx)#Create the order for one epoch

				for r_sp in range(Nlearn):#loop over all the data
	
					ite+=1
					grad=dfunc(w0,X[r_sp,:],y[r_sp],args)
					
					m=beta1*m+(1.-beta1)*grad
					u=np.max(beta2*u,np.abs(grad))
					mhat=m/(1.-beta1**ite)
		
					w=w0-eta*mhat/u
					err=np.linalg.norm(w-w0)

					if err<tol and ep>1:
						break

					w0=w.copy()

		else:#Mini-batch
		
			while err>tol:

                        	if ep==epochMax:
                                	mess='No convergenge after '+str(epochMax)+' epochs'
					if WarnM is True:
                                		warnings.warn(mess)
                                	break

                        	ep+=1#Which epoch

                        	np.random.shuffle(idx)#Create the order for one epoch
				sp=np.array_split(idx,np.floor( Nlearn/float(Nbatch) ) )

				for r_sp in range(len(sp)):#loop over all the mini-batch
					ite+=1
					idData=sp[r_sp]
					nn=len(idData)
					
					grad=dfunc(w0,X[idData,:],y[idData],args)/float(nn)

					m=beta1*m+(1.-beta1)*grad
                                        u=np.max(beta2*u,np.abs(grad))
                                        mhat=m/(1.-beta1**ite)

                                        w=w0-eta*mhat/u
                                	err=np.linalg.norm(w-w0)

					if err<tol and ep>1:
                                        	break

                                	w0=w.copy()

	else:
	
		if Nbatch==1:

			while err>tol:

				if ep==epochMax:
                        		mess='No convergenge after '+str(epochMax)+' epochs'
					if WarnM is True:
                        			warnings.warn(mess)
                                	break

				ep+=1#Which epoch

				np.random.shuffle(idx)#Create the order for one epoch

				for r_sp in range(Nlearn):#loop over all the data
	
					ite+=1
					grad=dfunc(w0,X[r_sp,:],y[r_sp],*args)

					m=beta1*m+(1.-beta1)*grad
                                        u=np.max(beta2*u,np.abs(grad))
                                        mhat=m/(1.-beta1**ite)

                                        w=w0-eta*mhat/u
					err=np.linalg.norm(w-w0)

					if err<tol and ep>1:
						break

					w0=w.copy()

		else:#Mini-batch
		
			while err>tol:

                        	if ep==epochMax:
                                	mess='No convergenge after '+str(epochMax)+' epochs'
					if WarnM is True:
                                		warnings.warn(mess)
                                	break

                        	ep+=1#Which epoch

                        	np.random.shuffle(idx)#Create the order for one epoch
				sp=np.array_split(idx,np.floor( Nlearn/float(Nbatch) ) )

				for r_sp in range(len(sp)):#loop over all the mini-batch
					ite+=1
					idData=sp[r_sp]
					nn=len(idData)
					grad=dfunc(w0,X[idData,:],y[idaDta],*args)/float(nn)

					m=beta1*m+(1.-beta1)*grad
                                        u=np.max(beta2*u,np.abs(grad))
                                        mhat=m/(1.-beta1**ite)

                                        w=w0-eta*mhat/u
                                	err=np.linalg.norm(w-w0)

					if err<tol and ep>1:
                                        	break

                                	w0=w.copy()

	return w,ep,ite,mess
