# -*- coding: utf-8 -*-

def simulHawkesTraj(Klass, mu, Tmax, nSample, kernelSim):
   # Kclass: number of classes
   # Tmax: observation time horizon
   # nSample: size of the learning sample
   # kernelSim: kernel function used to simulate the paths
   
   Y = np.random.choice(range(0,Kclass), size=nSample, replace=True)
   listJumptimes = [None]*nSample
   for k in range(0,Kclass):
         idx  = np.where(Y == k)[0]
         for i in idx:
             hawkes = SimuHawkes(n_nodes=1, end_time= Tmax, verbose=False)
             hawkes.set_kernel(0, 0, kernelSim[k])
             hawkes.set_baseline(0, mu)
             hawkes.simulate()
             listJumptimes[i] = hawkes.timestamps[0]
   return(listJumptimes, Y);


def phiFtestim_expo(Jumptimes, Kclass, param, p, Tmax):
   # Jumptimes: list of jumps times 
   # Kclass: number of classes
   # param: liste of parameters (size of the list: Kclass)
   # Tmax: observation time horizon


  n = len(Jumptimes)
  out = np.zeros([n,Kclass])

  for k in range(n):
          for l in range(Kclass):
              N = len(Jumptimes[k])
              paraml = param[l]
              if (N > 0):
                  term1 = np.zeros(N)
                  for i in range(1,N) : 
                      term1[i] = (1 +  term1[i-1]) * np.exp(-paraml[2]*(Jumptimes[k][i] - Jumptimes[k][i-1]))
                  termlog = (paraml[0] + (paraml[1]*paraml[2] * term1))
                  termlog = termlog*(termlog>0)+ 0.00001
                  estimFT = np.sum(np.log(termlog)) - paraml[0]*Tmax +paraml[1]*np.sum(np.exp(-paraml[2]*(Jumptimes[k][N-1] - Jumptimes[k])) - 1)
              if (N==0):
                  estimFT = - paraml[0] * Tmax
              out[k,l] = p[l]* np.exp(estimFT)
     
          out[k, :] =  out[k, :]/sum(out[k, :])
  

  return out;





def estimParamErm(Xtrain, Ytrain, p, Tmax, Kclass, init):
    # Xtrain: list of jump-times paths of the learning sample
    # Ytrain: array of labels for the learning sample
    # p: vector of probability weights of Y
    # Tmax: observation time horizon
    # Kclass: number of classes
    # init: initial guess for the optimization task
    
    Z  = -np.ones((np.size(Ytrain), Kclass))         
    for k in range(nTrain):
        Z[k,Ytrain[k]]= 1;
            
    def computePhiRisk(theta):
                mu = np.ones(Kclass)*np.exp(theta[0])
                mu.shape=(Kclass,1)
                alphBeta = theta[range(1,np.size(theta))]
                alphBeta.shape = (Kclass,2)
                alphBeta[:,0] = 1/(1+np.exp(-alphBeta[:,0]))                         
                alphBeta[:,1] = np.exp(alphBeta[:,1])
                thetaAux = np.concatenate((mu,alphBeta), axis = 1)
                
                scoresTraj = phiFtestim_expo(Jumptimes= Xtrain, Kclass=Kclass, param = thetaAux, p=p, Tmax= Tmax)
               
                TargetF = 2*scoresTraj - 1
                phiRisk = np.mean(np.sum((1-TargetF*Z)**2, axis = 1))
                return phiRisk;

    OptimRisk = minimize(computePhiRisk, init, method = 'BFGS') 
        
    thetaHat = OptimRisk.x
    muHat = np.ones(Kclass)*np.exp(thetaHat[0])
    muHat.shape=(Kclass,1)
    alphBetaHat = thetaHat[range(1,np.size(thetaHat))]
    alphBetaHat.shape = (Kclass,2)
    alphBetaHat[:,0] = 1/(1+np.exp(-alphBetaHat[:,0]))                                             
    alphBetaHat[:,1] = np.exp(alphBetaHat[:,1])
    thetaAuxHat = np.concatenate((muHat,alphBetaHat), axis = 1)
           
    return(thetaAuxHat);














