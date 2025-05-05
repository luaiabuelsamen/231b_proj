import numpy as np
import scipy as sp
#NO OTHER IMPORTS ALLOWED (However, you're allowed to import e.g. scipy.linalg)

def estInitialize():
    # Fill in whatever initialization you'd like here. This function generates
    # the internal state of the estimator at time 0. You may do whatever you
    # like here, but you must return something that is in the format as may be
    # used by your estRun() function as the first returned variable.
    #
    # The second returned variable must be a list of student names.
    # 
    # The third return variable must be a string with the estimator type

    #we make the internal state a list, with the first three elements the position
    # x, y; the angle theta; and our favorite color. 
    x = 0
    y = 0
    theta = 0
    color = 'green' 
    # note that there is *absolutely no prescribed format* for this internal state.
    # You can put in it whatever you like. Probably, you'll want to keep the position
    # and angle, and probably you'll remove the color.
    '''
    internalState = [x,
                     y,
                     theta, 
                     color
                     ]
    '''

    #        x_state, P, Evv, Eww, r, B = internalStateIn

    #Initial state mean and variance
    # x, y, theta

    theta = np.deg2rad(45) #we are heading north-east, which is approxiamtely 45 degrees from theta=0 being the x axis.


    #Process/ Meas noise variance
    Evv = np.diag([0.001, 0.001, 0.001,0,0])
    Eww = np.array([[1.08933973,1.53329122],[1.53329122,2.98795486]])  #straight from examineCalib.py  y GPS is noiser than the X, AND, the errors are correlated.

    #TODO: Augment as latent state
    r = 0.425
    B = 0.8



    internalStateDoug = []
    B =  0.8 # the baseline B is uncertain (to within approximately ±10%)
    var_B = B*(1/12)*((1.1-0.9)**2)   #treat the wheel base as a uniform random variable, over the range from B+10% to B-10%.
    r = 0.425
    var_r = r*(1/12)*((1.05-0.95)**2)   #Same idea with the wheel rim radius, but with +/- 5%.

    x_state = np.array([[0.0], [0.0], [theta],[r],[B] ])
    P = np.diag([1.0, 1.0, 0.1,var_r,var_B])        #variance on the initial state.  These, are guesses.




    internalState = [
        x_state,
        P,
        Evv,
        Eww,]
        

    studentNames = ['Luai Abuelsamen',
                    'Douglas Hutchings']
    
    # replace this with the estimator type. Use one of the following options:
    #  'EKF' for Extended Kalman Filter
    #  'UKF' for Unscented Kalman Filter
    #  'PF' for Particle Filter
    #  'OTHER: XXX' if you're using something else, in which case please
    #                 replace "XXX" with a (very short) description
    estimatorType = 'UKF'  
    
    return internalState, studentNames, estimatorType

