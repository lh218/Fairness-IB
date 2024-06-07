import numpy as np
import pandas as pd
from sklearn import metrics


def measures_from_Yhat(Y, Z, Yhat=None, threshold=0.5):
    assert isinstance(Y, np.ndarray)
    assert isinstance(Z, np.ndarray)
    assert Yhat is not None
    assert isinstance(Yhat, np.ndarray)
    
    if Yhat is not None:
        Ytilde = (Yhat >= threshold).astype(np.float32)
    assert Ytilde.shape == Y.shape and Y.shape == Z.shape
    
    # Y_hat_1_Y_1 = sum((Ytilde == 1)&(Y == 1))
    # Y_hat_1_Y_0 = sum((Ytilde == 1)&(Y == 0))
    # Y_hat_0_Y_1 = sum((Ytilde == 0)&(Y == 1))
    # Y_hat_0_Y_0 = sum((Ytilde == 0)&(Y == 0))
    
    # P_Y_hat_1_Y_1 = float(Y_hat_1_Y_1) / (Y_hat_1_Y_1 + Y_hat_0_Y_1)
    # P_Y_hat_1_Y_0 = float(Y_hat_1_Y_0) / (Y_hat_1_Y_0 + Y_hat_0_Y_0)
    # P_Y_hat_0_Y_1 = float(Y_hat_0_Y_1) / (Y_hat_1_Y_1 + Y_hat_0_Y_1)
    # P_Y_hat_0_Y_0 = float(Y_hat_0_Y_0) / (Y_hat_1_Y_0 + Y_hat_0_Y_0)
    
    # Prob = [P_Y_hat_1_Y_1, P_Y_hat_1_Y_0, P_Y_hat_0_Y_1, P_Y_hat_0_Y_0]
    
    # Accuracy
    acc = (Ytilde == Y).astype(np.float32).mean()
    # DP  
    DDP = abs(np.mean(Ytilde[Z==0])-np.mean(Ytilde[Z==1]))

    # Y_Z0, Y_Z1 = Y[Z==0], Y[Z==1]
    # Y1_Z0 = Y_Z0[Y_Z0==1]
    # Y0_Z0 = Y_Z0[Y_Z0==0]
    # Y1_Z1 = Y_Z1[Y_Z1==1]
    # Y0_Z1 = Y_Z1[Y_Z1==0]
    
    # FPR, FNR = {}, {}
    # FPR[0] = np.sum(Ytilde[np.logical_and(Z==0, Y==0)])/len(Y0_Z0)
    # FPR[1] = np.sum(Ytilde[np.logical_and(Z==1, Y==0)])/len(Y0_Z1)

    # FNR[0] = np.sum(1 - Ytilde[np.logical_and(Z==0, Y==1)])/len(Y1_Z0)
    # FNR[1] = np.sum(1 - Ytilde[np.logical_and(Z==1, Y==1)])/len(Y1_Z1)
    
    # TPR_diff = abs((1-FNR[0]) - (1-FNR[1]))
    # FPR_diff = abs(FPR[0] - FPR[1])
    # DEO = TPR_diff + FPR_diff
    
    # EO
    Y_Z0, Y_Z1 = Y[Z==0], Y[Z==1]
    Y1_Z0 = Y_Z0[Y_Z0==1]
    Y0_Z0 = Y_Z0[Y_Z0==0]
    Y1_Z1 = Y_Z1[Y_Z1==1]
    Y0_Z1 = Y_Z1[Y_Z1==0]
    
    P_Y1_Z0 = np.sum(Ytilde[np.logical_and(Z==0, Y==1)])/len(Y1_Z0)
    P_Y1_Z1 = np.sum(Ytilde[np.logical_and(Z==1, Y==1)])/len(Y1_Z1)
    
    DEO = abs(P_Y1_Z0 - P_Y1_Z1)
    
    P_y1 = float(sum(Ytilde)/len(Ytilde))
    P_y0 = float((len(Ytilde) - sum(Ytilde))/len(Ytilde))
    
    Ytilde_Z0, Ytilde_Z1 = Ytilde[Z==0], Ytilde[Z==1]
    Err_Z0 = 1 - (Ytilde_Z0 == Y_Z0).astype(np.float32).mean()
    Err_Z1 = 1 - (Ytilde_Z1 == Y_Z1).astype(np.float32).mean()
    P_err = abs(Err_Z0 - Err_Z1)
    
    # data = [acc, F1_score, precision, recall, AUC_score, DDP, DEO, P_y0, P_y1]
    # columns = ['acc', 'F1', 'pre', 'rec', 'auc', 'DDP', 'DEO', 'P(Y=0|S)', 'P(Y=1|S)']
    data = [acc, DDP, DEO, P_y0, P_y1, Err_Z0, Err_Z1, P_err]
    columns = ['acc', 'DDP', 'DEO', 'P(Y=0|S)', 'P(Y=1|S)', 'Err_Z0', 'Err_Z1', 'P_err']
    return pd.DataFrame([data], columns=columns)