# This file is used to define the BayesianParameterEstimation class
# Developed by Morolake Omoya and Henry Burton in April 2022


# Immport relevant packages
import numpy as np
from scipy.stats import expon
from scipy.stats import norm
from scipy.stats import gamma
import math

#############################################################################
#        Define a class that performs Bayesian parameter estimation         # 
#    using the Conjugate Prior (CP) and Markov Chain Monte Carlo (MCMC)     #
#############################################################################


class BayesianParameterEstimation(object):
    """
    This class contains the following two functions:
    (1) Perform Bayesian parameter estimation using the conjugate prior method
    (2) Perform Bayesian parameter estimation using Markov Chain Monte Carlo
    """

    def __init__(self,data,NSamples,lambdaPriorMean,lambdaPriorCOV):
        """
        This function initializes the attributes of a BayesianParameterEstimation instance. 
        """
        
        # Attach inputs to object
        self.data = data
        self.NSamples = NSamples
        self.lambdaPriorMean = lambdaPriorMean
        self.lambdaPriorCOV = lambdaPriorCOV
        
        # Define relevant attributes of the class
        self.posteriorCP = {}
        self.posteriorMCMC = {}
        self.marginal = {}
        
    def mcmcEstimation(self):
        """
        This function performs Markov Chain Monte Carlo estimation of the posterior distribution for the mean duration  
        (and associated lambda) using a gamma prior and exponential sampling distribution
    
        INPUTS
        NSamples:                          scalar value representing the number of samples/iterations generated in the MCMC chain
        data:                              N X 1 array of observed duration variable values
        lambdaPriorMean:                   scalar variable that is the prior mean of the duration lambda
        lambdaPriorCOV:                    scalar variable that is the prior coefficient of variation of the lambda duration
        
        OUTPUTS
        meanPosteriorSamples:              L X 1 array of the posterior sample mean duration values generated using MCMC
        meanPosteriorDensity:              L X 2 array of the probability density function of posterior distribution of the mean duration
                                           The first column contains the mean duration values and the second column contains the densities
        meanPosteriorMean:                 scalar variable that is the posterior mean of the mean duration
        meanPosteriorSD:                   scalar variable that is the posterior standard deviation of the mean duration 
        meanPosteriorCOV:                  scalar variable that is the posterior coefficient of variation of the mean duration 
        
        lambdaPosteriorSamples:            L X 1 array of the posterior sample duration-lambda values generated using MCMC
        lambdaPosteriorDensity:            L X 2 array of the probability density function of posterior distribution of the duration lambda. 
                                           The first column contains the lambda values and the second column contains the densities
        lambdaPosteriorMean:               scalar variable that is the posterior mean of the lambda corresponding to the mean duration
        lambdaPosteriorSD:                 scalar variable that is the posterior standard deviation of the lambda corresponding to
                                           the mean duration
        lambdaPosteriorCOV:                scalar variable that is the posterior coefficient of variation of the lambda corresponding to
                                           the mean duration        
        """
        
        # Define variables
        data = self.data
        lambdaPriorMean = self.lambdaPriorMean
        lambdaPriorCOV = self.lambdaPriorCOV
        NSamples = self.NSamples
        lambdaPriorSD = lambdaPriorCOV*lambdaPriorMean
        
        # Specify the current mean lambda
        lambdaMeanCurrent = lambdaPriorMean
        lambdaSDCurrent = lambdaPriorCOV*lambdaMeanCurrent
        
        
        # Compute prior gamma distribution parameters for lambda
        lambdaPriorAlpha = (lambdaPriorMean/lambdaPriorSD)**2
        lambdaPriorBeta = lambdaPriorAlpha/lambdaPriorMean
        
        # Initialize list to store posterior mean samples of mean duration
        meanPosteriorSamples = []
        
        # Initialize list to store posterior mean samples of duration lambda
        lambdaPosteriorSamples = []
        
        # Initialize list to store posterior mean samples of mean duration
        meanPosteriorSamples1 = []
        
        # Initialize list to store posterior mean samples of duration lambda
        lambdaPosteriorSamples1 = []
        
        # Loop over the number of MCMC iterations
        for i in range(NSamples):
            # Compute prior gamma distribution parameters for lambda
            lambdaAlphaCurrent = (lambdaMeanCurrent/lambdaSDCurrent)**2
            lambdaBetaCurrent = lambdaAlphaCurrent/lambdaMeanCurrent
            
            # Sample the "proposed" mean duration from the gamma distribution (prior)
            lambdaMeanProposed = gamma.rvs(lambdaAlphaCurrent, scale = 1/lambdaBetaCurrent)
            
            # Compute the log likelihood for the "current" and "proposed" mean 
            logLikLambdaMeanCurrent = np.sum(np.log(expon.pdf(data,scale = 1/lambdaMeanCurrent)))
            logLikLambdaMeanProposed = np.sum(np.log(expon.pdf(data,scale = 1/lambdaMeanProposed)))
            
            # Compute the log prior probability of "current" and "proposed" mean based on the gamma distribution
            logPriorProbLambdaMeanCurrent = np.log(gamma.pdf(lambdaMeanCurrent,lambdaPriorAlpha, scale = 1/lambdaPriorBeta))
            logPriorProbLambdaMeanProposed = np.log(gamma.pdf(lambdaMeanProposed,lambdaPriorAlpha, scale = 1/lambdaPriorBeta))
            
            # Compute the probability of acceptance
            probAccept = (logPriorProbLambdaMeanProposed + logLikLambdaMeanProposed) - (logPriorProbLambdaMeanCurrent + logLikLambdaMeanCurrent)            
            
            # Check for acceptance
            acceptProposed = np.random.rand() < np.exp(probAccept)
            
            if acceptProposed:
                # Append poster samples with proposed mean
                meanPosteriorSamples.append(1/lambdaMeanProposed)
                lambdaPosteriorSamples.append(lambdaMeanProposed) 
                
                # Set lambdaMeanCurrent = lambdaMeanProposed
                lambdaMeanCurrent = lambdaMeanProposed
                lambdaSDCurrent = lambdaPriorCOV*lambdaMeanCurrent
                
        # Compute mean, standard deviation and coefficient of variation of posterior samples
        meanPosteriorMean = np.mean(np.array(meanPosteriorSamples))
        meanPosteriorSD = np.std(np.array(meanPosteriorSamples))
        posteriorCOV = meanPosteriorSD/meanPosteriorMean        
        lambdaPosteriorMean = np.mean(np.array(lambdaPosteriorSamples))
        
        # Compute posterior gamma distribution parameters for lambda
        lambdaPosteriorAlpha = (lambdaPosteriorMean/np.std(np.array(lambdaPosteriorSamples)))**2
        lambdaPosteriorBeta = lambdaPosteriorAlpha/lambdaPosteriorMean
        
        
        # Store in dictionary
        self.posteriorMCMC = {'meanSamples': meanPosteriorSamples,
                           'lambdaSamples': lambdaPosteriorSamples,
                             'meanMean': meanPosteriorMean,
                             'meanSD': meanPosteriorSD,
                             'COV': posteriorCOV,
                             'lambdaMean': lambdaPosteriorMean,
                             'alpha': lambdaPosteriorAlpha,
                             'beta': lambdaPosteriorBeta}
        
        
    def cpEstimation(self):
        """
        This function performs conjugate prior (CP) estimation of the posterior distribution for the duration  lambda
        (and associated mean) using a gamma prior and exponential sampling distribution
    
        INPUTS
        data:                              N X 1 array of observed duration variable values
        lambdaPriorMean:                   scalar variable that is the prior mean of the duration lambda
        lambdaPriorCOV:                    scalar variable that is the prior coefficient of variation of the lambda duration
        
        OUTPUTS
        meanPosteriorMean:                 scalar variable that is the posterior mean of the mean duration
        meanPosteriorSD:                   scalar variable that is the posterior standard deviation of the mean duration 
        posteriorCOV:                      scalar variable that is the posterior coefficient of variation         
        lambdaPosteriorMean:               scalar variable that is the posterior mean of the lambda corresponding to the mean duration
        lambdaPosteriorSD:                 scalar variable that is the posterior standard deviation of the lambda corresponding to
                                           the mean duration      
        """
        
        # Define variables
        data = self.data
        lambdaPriorMean = self.lambdaPriorMean
        lambdaPriorCOV = self.lambdaPriorCOV
        
        # Compute prior standard deviation of lambda
        lambdaPriorSD = lambdaPriorCOV*lambdaPriorMean
        
        # Compute prior gamma distribution parameters for lambda
        lambdaPriorAlpha = (lambdaPriorMean/lambdaPriorSD)**2
        lambdaPriorBeta = lambdaPriorAlpha/lambdaPriorMean
        
        # Compute posterior gamma distribution parameters for the mean duration
        lambdaPosteriorBeta = lambdaPriorBeta + np.sum(data)
        lambdaPosteriorAlpha = lambdaPriorAlpha + len(data)
        
        # Compute posterior mean, standard deviation and coefficient of variation of the lambda duration
        lambdaPosteriorMean = lambdaPosteriorAlpha/lambdaPosteriorBeta
        lambdaPosteriorSD = math.sqrt(lambdaPosteriorAlpha)/lambdaPosteriorBeta
        posteriorCOV = lambdaPosteriorSD/lambdaPosteriorMean
        
        # Compute posterior mean, standard deviation and coefficient of variation of the mean duration
        meanPosteriorMean = 1/lambdaPosteriorMean
        meanPosteriorSD = posteriorCOV*meanPosteriorMean
        
               
        # Store in dictionary
        self.posteriorCP = {'meanMean': meanPosteriorMean,
                             'meanSD': meanPosteriorSD,
                             'lambdaMean': lambdaPosteriorMean,
                             'lambdaSD': lambdaPosteriorSD,
                             'COV': posteriorCOV,
                             'alpha': lambdaPosteriorAlpha,
                             'beta': lambdaPosteriorBeta}
        
    def marginalDistribution(self,lambdaPosteriorMean,posteriorCOV,durationMin,durationMax,numDurations,numLambdas,lambdaFactor):
        """
        This function computes the marginal distribution of the duration variable based on Equation 7 of Omoya et al. 2022
        
        INPUTS
        posteriorCOV:                      scalar variable that is the posterior coefficient of variation         
        lambdaPosteriorMean:               scalar variable that is the posterior mean of the lambda corresponding to the mean duration
        durationMin:                       Minimum considered value of the duration variable
        durationMax:                       Maximum considered value of the duration variable  
        numDurations:                      Size of duration vector
        numLambdas:                        Size of lambda vector
        lambdaFactor:                      Factor that will be used to define the minimm and maximum lambda based on the mean
        
        OUTPUTS
        durationPDFVector:                 Vector of pdf values for the marginal distribution of the duration variable
        durationVector:                    Vector of duration values 
        meanDuration:                      scalar value that is the mean value of the marginal distribution of the duration variable       
        stdDuration:                       scalar value that is the standard deviation of the marginal distribution of the duration variable    
        """
        
        # Define posterior distribution parameters
        lambdaPosteriorSD= lambdaPosteriorMean*posteriorCOV
        lambdaPosteriorAlpha = (lambdaPosteriorMean/lambdaPosteriorSD)**2
        lambdaPosteriorBeta = lambdaPosteriorAlpha/lambdaPosteriorMean
        
        # Size of duration increment
        durationInc = (durationMax - durationMin)/numDurations

        # Define vector for duration
        durationVector = np.linspace(durationMin,durationMax,numDurations)
        
        # Define minimum lambda
        lambdaMin = lambdaPosteriorMean/lambdaFactor

        # Define maximum lambda
        lambdaMax = lambdaPosteriorMean*lambdaFactor

        # Size of lambda increment
        lambdaInc = (lambdaMax - lambdaMin)/numLambdas

        # Define vector for lambda
        lambdaVector = np.linspace(lambdaMin,lambdaMax,numLambdas)

        # Compute lambda pdf
        lambdaPDF = gamma.pdf(lambdaVector,lambdaPosteriorAlpha, scale = 1/lambdaPosteriorBeta)
        
        # Initialize array used to store duration pdf
        durationPDFVector = np.zeros((len(durationVector),1))

        # Loop over the duration vector
        for i in range(len(durationVector)):
            # Extract current duration
            duration = durationVector[i]
    
            # Loop over the number of lambda values
            for j in range(len(lambdaVector)):
                # Compute pdf of duration given the current lambda
                durationPDF = expon.pdf(duration,scale = 1/lambdaVector[j])
        
                # Update durationPDFVector
                durationPDFVector[i] = durationPDFVector[i] + durationPDF*lambdaPDF[j]*lambdaInc
                
        # Initialize mean and standard deviation of duration
        meanDuration = 0
        stdDuration = 0

        # Loop over the duration vector
        for i in range(len(durationVector)):
            # Update meanDuration
            meanDuration += durationVector[i]*durationPDFVector[i]*durationInc

    
        # Loop over the duration vector
        for i in range(len(durationVector)):
            # Update stdDuration
            stdDuration += durationPDFVector[i]*durationInc*(meanDuration - durationVector[i])**2 

        # Update stdDuration
        stdDuration = math.sqrt(stdDuration)
        
               
        # Store in dictionary
        self.marginal = {'duration': durationVector,
                             'pdf': durationPDFVector,
                             'mean': meanDuration,
                             'SD': stdDuration}   