# -*- coding: utf-8 -*-

# Source code to perform the classification task for exponential model (Model 1 with K=2 class, see Section 5 for more details)

import numpy as np

import scipy
from scipy.optimize import minimize

import tick
from tick.hawkes import SimuHawkes, HawkesExpKern, HawkesKernelExp
                      

exec(open('classif_func.py').read())  # python3, load functions used to perform classification procedures 

# HyperParam -------------------------------------------
    
Kclass = 2
p = np.ones(Kclass)/Kclass 
Tmax = 20
dt = 0.01

initparam = [np.log(0.5),0,0,0,0] 
nTrain = 100 
nTest = 1000

# Kernel Param -------------------------------------------

mu = 1. 
param = [None] * Kclass
param[0] = np.array([mu, 0.7,1.3])
param[1] = np.array([mu,0.2,3])


kernelSim = [None] * Kclass
kernelBayes = [None] * Kclass

for k in range(Kclass):
   kernelSim[k] = HawkesKernelExp(param[k][1], param[k][2])
   kernelBayes[k] = kernelSim[k].get_values     

##############################################################
# Risks computation -------------------------------------------
##############################################################

# TrainSample
listJumptimesTrain, Ytrain = simulHawkesTraj(Klass = Kclass, mu = mu, Tmax = Tmax, nSample = nTrain, kernelSim = kernelSim)
# TestSample
listJumptimesTest, Ytest = simulHawkesTraj(Klass = Kclass, mu = mu, Tmax = Tmax, nSample = nTest, kernelSim = kernelSim)


#################### Estimation of p_k ######################################
#############################################################################

pHat = np.zeros(Kclass)
for k in range(Kclass):
      pHat[k]=np.mean(Ytrain==k)

#################### BAYES ##################################################
#############################################################################

piStar = phiFtestim_expo(Jumptimes = listJumptimesTest, Kclass = Kclass, param = param, p = p, Tmax = Tmax)
bayesPred = np.argmax(piStar, axis = 1)
ErrorBayes = np.mean(bayesPred != Ytest)


################### PERM ####################################################
############################################################################# 
thetaHat = estimParamErm(Xtrain = listJumptimesTrain, Ytrain =Ytrain, p= pHat, Tmax=Tmax, Kclass=Kclass, init = initparam)
piEstimPErm = phiFtestim_expo(Jumptimes=listJumptimesTest, Kclass= Kclass, param= thetaHat, p= pHat, Tmax= Tmax)
PermPred = np.argmax(piEstimPErm, axis = 1)
ErrorPErm = np.mean(PermPred != Ytest)

#################### PG ##################################################### 
#############################################################################

paramLS = [None]*Kclass
for k in range(Kclass):
      classk = (np.where(Ytrain==k))
      classk = np.array(classk[0])
      listJumptimesK = [None]*len(classk)
      for i in range(len(classk)):
            listJumptimesK[i] = [listJumptimesTrain[classk[i]]] # tick format
      learnerLSK = HawkesExpKern(decays= 0.5, gofit = 'least-squares') 
      learnerLSK.fit(listJumptimesK)
      paramLS[k] = [float(learnerLSK.baseline[0]), float(learnerLSK.adjacency[0]), learnerLSK.decays]


piEstimPlugIn  = phiFtestim_expo(Jumptimes= listJumptimesTest, Kclass=Kclass, param = paramLS, p=pHat, Tmax= Tmax)
plugInPred = np.argmax(piEstimPlugIn, axis = 1)
ErrorPG = np.mean(plugInPred != Ytest)

#################################################################################
#################################################################################

print(ErrorBayes)
print(ErrorPErm)
print(ErrorPG)






