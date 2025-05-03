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



    internalStateDoug = []
    B =  0.8 # the baseline B is uncertain (to within approximately ±10%)
    var_B = B*(1/12)*((1.1-0.9)**2)   #treat the wheel base as a uniform random variable, over the range from B+10% to B-10%.
    r = 0.425
    var_r = r*(1/12)*((1.05-0.95)**2)   #Same idea with the wheel rim radius, but with +/- 5%.

    x = 0
    y = 0 
    #todo var?

    theta = np.deg2rad(45) #we are heading north-east, which is approxiamtely 45 degrees from theta=0 being the x axis.



    internalState = [x,
                     y,
                     theta, 
                     color
                     ]


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

