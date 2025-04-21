import numpy as np
import scipy as sp
import scipy.linalg as linalg

def estRun(time, dt, internalStateIn, steeringAngle, pedalSpeed, measurement):
    if isinstance(internalStateIn, list) and len(internalStateIn) == 4:
        x = 0.0
        y = 0.0
        theta = 0.0
        x_state = np.matrix([[x], [y], [theta]])
        
        r = 0.425
        B = 0.8
        
        Q = np.matrix([[0.01, 0, 0], 
                      [0, 0.01, 0], 
                      [0, 0, 0.005]])
        
        R = np.matrix([[0.1, 0], 
                      [0, 0.1]])
        
        P = np.matrix([[1.0, 0, 0], 
                      [0, 1.0, 0], 
                      [0, 0, 0.1]])
        
        alpha = 0.1
        beta = 2
        kappa = 0
    else:
        x_state = internalStateIn[0]
        P = internalStateIn[1]
        Q = internalStateIn[2]
        R = internalStateIn[3]
        alpha = internalStateIn[4]
        beta = internalStateIn[5]
        kappa = internalStateIn[6]
        r = internalStateIn[7]
        B = internalStateIn[8]
    
    nx = 3
    
    def f(x, u, w):
        x_k = x[0,0]
        y_k = x[1,0]
        theta_k = x[2,0]
        
        v = 5 * r * u[1,0]
        
        x_new = x_k + dt * v * np.cos(theta_k)
        y_new = y_k + dt * v * np.sin(theta_k)
        theta_new = theta_k + dt * (v/B) * np.tan(u[0,0])
        
        x_new += w[0,0]
        y_new += w[1,0]
        theta_new += w[2,0]
        
        return np.matrix([[x_new], [y_new], [theta_new]])
    
    def h(x, u, v):
        x_center = x[0,0] + 0.5 * B * np.cos(x[2,0])
        y_center = x[1,0] + 0.5 * B * np.sin(x[2,0])
        
        return np.matrix([[x_center + v[0,0]], [y_center + v[1,0]]])
    
    L_aug = nx + Q.shape[0] + R.shape[0]
    lambda_param = alpha**2 * (L_aug + kappa) - L_aug
    gamma = np.sqrt(L_aug + lambda_param)
    
    Wm = np.zeros(2*L_aug + 1)
    Wc = np.zeros(2*L_aug + 1)
    Wm[0] = lambda_param / (L_aug + lambda_param)
    Wc[0] = Wm[0] + (1 - alpha**2 + beta)
    for i in range(1, 2*L_aug + 1):
        Wm[i] = 1 / (2 * (L_aug + lambda_param))
        Wc[i] = Wm[i]
    
    sgnW0 = 1 if Wc[0] >= 0 else -1
    
    try:
        S = linalg.cholesky(P).T
    except:
        P = P + 1e-6 * np.eye(P.shape[0])
        S = linalg.cholesky(P).T
    
    ix = np.arange(0, nx)
    iq = np.arange(nx, nx + Q.shape[0])
    ir = np.arange(nx + Q.shape[0], nx + Q.shape[0] + R.shape[0])
    
    Sa = np.zeros((L_aug, L_aug))
    Sa[np.ix_(ix, ix)] = S
    Sa[np.ix_(iq, iq)] = linalg.cholesky(Q).T
    Sa[np.ix_(ir, ir)] = linalg.cholesky(R).T
    
    xa = np.vstack([x_state, np.zeros((Q.shape[0], 1)), np.zeros((R.shape[0], 1))])
    gsa = np.hstack((gamma * Sa.T, -gamma * Sa.T)) + xa * np.ones((1, 2 * L_aug))
    X = np.hstack([xa, gsa])
    
    u = np.matrix([[steeringAngle], [pedalSpeed]])
    
    Y = np.zeros((2, 2*L_aug+1))
    X_prop = np.zeros((nx, 2*L_aug+1))
    
    for j in range(2*L_aug+1):
        X_prop[:, j:j+1] = f(X[ix, j:j+1], u, X[iq, j:j+1])
        Y[:, j:j+1] = h(X_prop[:, j:j+1], u, X[ir, j:j+1])
    
    x_pred = np.zeros((nx, 1))
    for j in range(2*L_aug+1):
        x_pred += Wm[j] * X_prop[:, j:j+1]
    
    y_pred = np.zeros((2, 1))
    for j in range(2*L_aug+1):
        y_pred += Wm[j] * Y[:, j:j+1]
    
    ex = np.zeros((nx, 2*L_aug+1))
    ey = np.zeros((2, 2*L_aug+1))
    
    for j in range(2*L_aug+1):
        ex[:, j:j+1] = np.sqrt(np.abs(Wc[j])) * (X_prop[:, j:j+1] - x_pred)
        ey[:, j:j+1] = np.sqrt(np.abs(Wc[j])) * (Y[:, j:j+1] - y_pred)
    
    Pxy = np.zeros((nx, 2))
    for j in range(2*L_aug+1):
        Pxy += Wc[j] * (X_prop[:, j:j+1] - x_pred) * (Y[:, j:j+1] - y_pred).T
    
    try:
        S = linalg.cholesky(ex @ ex.T).T / np.sqrt(2)
        Syy = linalg.cholesky(ey @ ey.T).T / np.sqrt(2)
    except:
        ex_cov = ex @ ex.T
        ey_cov = ey @ ey.T
        ex_cov = (ex_cov + ex_cov.T) / 2 + 1e-6 * np.eye(ex_cov.shape[0])
        ey_cov = (ey_cov + ey_cov.T) / 2 + 1e-6 * np.eye(ey_cov.shape[0])
        S = linalg.cholesky(ex_cov).T / np.sqrt(2)
        Syy = linalg.cholesky(ey_cov).T / np.sqrt(2)
    
    S_before = S
    if not (np.isnan(measurement[0]) or np.isnan(measurement[1])):
        try:
            y_meas = np.matrix([[measurement[0]], [measurement[1]]])
            SyyTSyy = Syy.T @ Syy
            K = Pxy @ np.linalg.inv(SyyTSyy)
            
            x_state = x_pred + K @ (y_meas - y_pred)
            
            U = K @ Syy.T
            for j in range(U.shape[1]):
                u_vec = np.ravel(U[:, j])
                alpha_chol = 1.0
                for k in range(S.shape[0]):
                    v = S[k, k]**2 + sgnW0 * u_vec[k]**2 * alpha_chol**2
                    if v <= 0:
                        raise Exception
                    r = np.sqrt(v)
                    c = r / S[k, k]
                    s = u_vec[k] * alpha_chol / S[k, k]
                    S[k, k] = r
                    if k < S.shape[0] - 1:
                        S[k, k+1:] = (S[k, k+1:] + sgnW0 * s * u_vec[k+1:] * alpha_chol) / c
                        u_vec[k+1:] = u_vec[k+1:] - s * np.ravel(S[k, k+1:])
                    alpha_chol = alpha_chol / c
        except:
            x_state = x_pred
            S = S_before
    else:
        x_state = x_pred
    
    x = float(x_state[0, 0])
    y = float(x_state[1, 0])
    theta = float(x_state[2, 0])
    
    P = S.T @ S
    
    internalStateOut = [x_state, P, Q, R, alpha, beta, kappa, r, B]
    
    return x, y, theta, internalStateOut