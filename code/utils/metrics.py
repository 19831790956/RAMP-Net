import numpy as np

def MAE(pred, true):
    return np.mean(np.abs(true - pred))

def MSE(pred, true):
    return np.mean((true - pred) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))
#[0,1] small
def SMAPE(pred, true):
    return 2.0 * np.mean(np.abs(pred - true) / (np.abs(pred) + np.abs(true)))*100
#[0,inf] small
def MAAPE(pred, true):
     result = np.mean(np.arctan(np.abs((true - pred) / true)))*100
     return result
#[-1,1] bigger
#person
def CORR(pred, true):
    m1, m2 = np.mean(true), np.mean(pred)
    numerator = np.sum((true - m1) * (pred - m2))
    denominator = np.sqrt(np.sum((true - m1) ** 2)) * np.sqrt(np.sum((pred - m2) ** 2))
    return np.mean(numerator / denominator)
#[-inf,1]  bigger
def R2(pred,true):
    r2 = 1 - np.sum((true - pred) ** 2) / np.sum((true - np.mean(true)) ** 2)
    return r2

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    smape = SMAPE(pred, true)
    maape = MAAPE(pred, true)
    r=CORR(pred, true)
    r2=R2(pred,true)



    return mae, mse, rmse, smape,maape,r,r2
