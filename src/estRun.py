import numpy as np
import scipy as sp
import scipy.linalg as linalg

#NO OTHER IMPORTS ALLOWED (However, you're allowed to import e.g. scipy.linalg)
np.set_printoptions(precision=3)


def estRun(time, dt, internalStateIn, steeringAngle, pedalSpeed, measurement):
    x, y, theta, internalStateOut = estRunLuai(time,dt,internalStateIn,steeringAngle,pedalSpeed,measurement)

    # In this function you implement your estimator. The function arguments
    # are:
    #  time: current time in [s] 
    #  dt: current time step [s]
    #  internalStateIn: the estimator internal state, definition up to you. 
    #  steeringAngle: the steering angle of the bike, gamma, [rad] 
    #  pedalSpeed: the rotational speed of the pedal, omega, [rad/s] 
    #  measurement: the position measurement valid at the current time step
    #
    # Note: the measurement is a 2D vector, of x-y position measurement.
    #  The measurement sensor may fail to return data, in which case the
    #  measurement is given as NaN (not a number).
    #
    # The function has four outputs:
    #  x: your current best estimate for the bicycle's x-position
    #  y: your current best estimate for the bicycle's y-position
    #  theta: your current best estimate for the bicycle's rotation theta
    #  internalState: the estimator's internal state, in a format that can be understood by the next call to this function

    # Example code only, you'll want to heavily modify this.
    # this internal state needs to correspond to your init function:






    #### OUTPUTS ####
    # Update the internal state (will be passed as an argument to the function
    # at next run), must obviously be compatible with the format of
    # internalStateIn:

    #print(measurement,x, y, theta)


    # DO NOT MODIFY THE OUTPUT FORMAT:
    return x, y, theta, internalStateOut 


def estRunLuai(time, dt, internalStateIn, steeringAngle, pedalSpeed, measurement):

    x_state, P, Evv, Eww, r, B = internalStateIn


    nx = 3

    #Propogate discretized dynamics
    def A(x, u):
        theta = x[2, 0]
        v_mag = 5 * r * u[1, 0]
        return np.array([
            [x[0, 0] + dt *  v_mag * np.cos(theta)],
            [x[1, 0] + dt *  v_mag * np.sin(theta)],
            [x[2, 0] + dt *  (v_mag / B) * np.tan(u[0, 0])]
        ])

    def H(x):
        x_c = x[0, 0] + 0.5 * B * np.cos(x[2, 0])
        y_c = x[1, 0] + 0.5 * B * np.sin(x[2, 0])
        return np.array([[x_c], [y_c]])

    u = np.array([[steeringAngle], [pedalSpeed]])

    # generate sigma points
    S = np.sqrt(nx)*np.linalg.cholesky(P)


    num_sigma_points = 2*nx+1 #luai included a sigma point at the current x state - very strange.

    #num_sigma_points = 2*nx #number of sigma points exactly corrisponding to ME231B's notes.
    
    sigma_points = np.zeros((nx, num_sigma_points))      #rows of sigma pints matrix corrispond to the states.  Columns is individual sigma points.

    if num_sigma_points ==  2*nx + 1:
        sigma_points[:, 0] = x_state[:, 0]

    for i in range(nx):                                                          
        if num_sigma_points ==  2*nx + 1:
            #need a bunch of off by 1 offsets to include the fact that hthe first sigma point is the mean
            sigma_points[:, i + 1] = (x_state + S[:, i:i+1])  [:, 0]
            sigma_points[:, i + 1 + nx] = (x_state - S[:, i:i+1])   [:, 0]

        else:
            sigma_points[:, i] = (x_state + S[:, i:i+1])  [:, 0]
            sigma_points[:, i + nx] = (x_state - S[:, i:i+1])   [:, 0]

    
    # propogate sigma points through system dynamics
    X_pred = np.zeros((nx, num_sigma_points))
    for i in range(num_sigma_points):
        X_pred[:, i:i+1] = A(sigma_points[:, i:i+1], u)

    # mean over sigma points
    x_pred = np.mean(X_pred, axis=1, keepdims=True)

    # find Pxx
    P_pred = Evv.copy()
    for i in range(num_sigma_points):
        dx = X_pred[:, i:i+1] - x_pred
        P_pred += (1 / (2 * nx)) * dx @ dx.T

    #propogate through measurement prediction
    Z_pred = np.zeros((2, num_sigma_points))
    for i in range(num_sigma_points):
        Z_pred[:, i:i+1] = H(X_pred[:, i:i+1])
    z_pred = np.mean(Z_pred, axis=1, keepdims=True)
        
    # Handle partial measurements
    valid_indices = [i for i, z in enumerate(measurement) if not np.isnan(z)]
    if valid_indices:
        # Slice measurement space
        z = np.array([[measurement[i]] for i in valid_indices])
        z_pred_used = z_pred[valid_indices, :]
        Z_pred_used = Z_pred[valid_indices, :]

        # Slice measurement noise
        Eww_used = Eww[np.ix_(valid_indices, valid_indices)]

        # Compute Pzz
        P_zz = Eww_used.copy()
        for i in range(num_sigma_points):
            dz = Z_pred_used[:, i:i+1] - z_pred_used
            P_zz += (1 / (2 * nx)) * dz @ dz.T

        # Compute cross-covariance
        P_xz = np.zeros((nx, len(valid_indices)))
        for i in range(num_sigma_points):
            dx = X_pred[:, i:i+1] - x_pred
            dx[2] = (dx[2] + np.pi) % (2 * np.pi) - np.pi
            dz = Z_pred_used[:, i:i+1] - z_pred_used
            P_xz += (1 / (2 * nx)) * dx @ dz.T

        # Kalman gain and update
        K = P_xz @ np.linalg.inv(P_zz)
        x_state = x_pred + K @ (z - z_pred_used)
        x_state[2] = (x_state[2] + np.pi) % (2 * np.pi) - np.pi
        P = P_pred - K @ P_zz @ K.T
    else:
        x_state = x_pred
        P = P_pred

    x, y, theta = float(x_state[0, 0]), float(x_state[1, 0]), float(x_state[2, 0])
    internalStateOut = [x_state, P, Evv, Eww, r, B]
    return x, y, theta, internalStateOut
