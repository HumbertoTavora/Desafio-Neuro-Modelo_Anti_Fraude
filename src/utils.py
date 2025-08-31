import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

def ks_statistic(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    ks = np.max(tpr - fpr)
    return ks, fpr, tpr, thresholds

def select_threshold_by_alert(y_true, y_scores, alert_rate=0.02):
    n = len(y_true)
    k = max(1, int(n * alert_rate))
    order = np.argsort(-y_scores)
    top_idx = order[:k]
    thresh = y_scores[order[k-1]]
    precision_topk = y_true.iloc[top_idx].mean()
    return float(thresh), float(precision_topk), k

def simulate_financial_impact(y_true, y_scores, vlr, threshold, profit_margin=0.05):
    y_pred = (y_scores >= threshold).astype(int)
    df_sim = pd.DataFrame({'y': y_true, 'score': y_scores, 'pred': y_pred, 'vlr': vlr})
    tp = df_sim[(df_sim.y==1) & (df_sim.pred==1)]
    tn = df_sim[(df_sim.y==0) & (df_sim.pred==0)]
    fp = df_sim[(df_sim.y==0) & (df_sim.pred==1)]
    fn = df_sim[(df_sim.y==1) & (df_sim.pred==0)]
    ganho_tn = tn.vlr.sum() * profit_margin
    perda_fp = fp.vlr.sum() * profit_margin
    perda_fn = fn.vlr.sum()
    ganho_tp = tp.vlr.sum()
    impacto = ganho_tn - perda_fp - perda_fn
    return {
        "lucro_bons_aprovados (TN)": ganho_tn,
        "perda_bons_bloqueados (FP)": perda_fp,
        "perda_fraudes_aprovadas (FN)": perda_fn,
        "ganho_fraudes_bloqueados (TP)": ganho_tp,
        "impacto_liquido": impacto,
        "precisao": (tp.shape[0] / max(1,(tp.shape[0]+fp.shape[0]))),
        "recall": (tp.shape[0] / max(1,(tp.shape[0]+fn.shape[0]))),
        "alerts": int((y_scores >= threshold).sum())
    }
