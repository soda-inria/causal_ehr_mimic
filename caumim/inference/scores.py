import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
    f1_score,
)
from sklearn.utils import check_consistent_length


def ipw_risk(y, a, hat_y, hat_e, trimming=None):
    if trimming is not None:
        clipped_hat_e = np.clip(hat_e, trimming, 1 - trimming)
    else:
        clipped_hat_e = hat_e
    ipw_weights = a / clipped_hat_e + (1 - a) / (1 - clipped_hat_e)
    return np.sum(((y - hat_y) ** 2) * ipw_weights) / len(y)


def r_risk(y, a, hat_m, hat_e, hat_tau):
    return np.mean(((y - hat_m) - (a - hat_e) * hat_tau) ** 2)


def u_risk(y, a, hat_m, hat_e, hat_tau):
    return np.mean(((y - hat_m) / (a - hat_e) - hat_tau) ** 2)


def w_risk(y, a, hat_e, hat_tau, trimming=None):
    if trimming is not None:
        clipped_hat_e = np.clip(hat_e, trimming, 1 - trimming)
    else:
        clipped_hat_e = hat_e
    pseudo_outcome = (y * (a - clipped_hat_e)) / (
        clipped_hat_e * (1 - clipped_hat_e)
    )
    return np.mean((pseudo_outcome - hat_tau) ** 2)


def ipw_r_risk(y, a, hat_mu_0, hat_mu_1, hat_e, hat_m, trimming=None):
    if trimming is not None:
        clipped_hat_e = np.clip(hat_e, trimming, 1 - trimming)
    else:
        clipped_hat_e = hat_e
    ipw_weights = a / clipped_hat_e + (1 - a) / (1 - clipped_hat_e)
    hat_tau = hat_mu_1 - hat_mu_0

    return np.sum(
        (((y - hat_m) - (a - hat_e) * (hat_tau)) ** 2) * ipw_weights
    ) / len(y)


def ipw_r_risk_oracle(y, a, hat_mu_0, hat_mu_1, e, mu_1, mu_0):
    m = mu_0 * (1 - e) + mu_1 * e
    return ipw_r_risk(
        y=y, a=a, hat_mu_0=hat_mu_0, hat_mu_1=hat_mu_1, hat_e=e, hat_m=m
    )


# ### metrics Utils ### #


def print_metrics_regression(y_true, predictions, verbose=1, elog=None):
    mad = mean_absolute_error(y_true, predictions)
    mse = mean_squared_error(y_true, predictions)
    mape = mean_absolute_percentage_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)

    if verbose:
        print("Mean absolute deviation (MAD) = {}".format(mad))
        print("Mean squared error (MSE) = {}".format(mse))
        print("Mean absolute percentage error (MAPE) = {}".format(mape))
        print("R^2 Score = {}".format(r2))

    return {
        "mad": mad,
        "mse": mse,
        "mape": mape,
        "r2": r2,
    }


def print_metrics_binary(y_true, prediction_probs, verbose=1, elog=None):
    if verbose:
        print("==> Binary scores:")
    prediction_probs = np.array(prediction_probs)
    prediction_probs = np.transpose(
        np.append([1 - prediction_probs], [prediction_probs], axis=0)
    )
    predictions = prediction_probs.argmax(axis=1)
    cf = confusion_matrix(y_true, predictions, labels=range(2))
    if elog is not None:
        elog.print("Confusion matrix:")
        elog.print(cf)
    elif verbose:
        print("Confusion matrix:")
        print(cf)
    cf = cf.astype(np.float32)

    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    prec0 = cf[0][0] / (cf[0][0] + cf[1][0])
    prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
    rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
    rec1 = cf[1][1] / (cf[1][1] + cf[1][0])

    auroc = roc_auc_score(y_true, prediction_probs[:, 1])
    (precisions, recalls, thresholds) = precision_recall_curve(
        y_true, prediction_probs[:, 1]
    )
    auprc = average_precision_score(y_true, prediction_probs[:, 1])
    f1macro = f1_score(y_true, predictions, average="macro")
    # calibration
    brier = brier_score_loss(y_true, prediction_probs[:, 1])
    bss = brier_skill_score(y_true, prediction_probs[:, 1])
    results = {
        "Accuracy": acc,
        "Precision class 0": prec0,
        "Precision class 1": prec1,
        "Recall class 0": rec0,
        "Recall class 1": rec1,
        "Area Under the Receiver Operating Characteristic curve (AUROC)": auroc,
        "Area Under the Precision Recall curve (AUPRC)": auprc,
        "F1 score (macro averaged)": f1macro,
        "Brier score": brier,
        "Brier Skill Score": bss,
    }
    if verbose:
        for key in results:
            print("{} = {}".format(key, results[key]))

    return {
        "acc": acc,
        "prec0": prec0,
        "prec1": prec1,
        "rec0": rec0,
        "rec1": rec1,
        "auroc": auroc,
        "auprc": auprc,
        "f1macro": f1macro,
        "brier": brier,
        "bss": bss,
    }


def brier_skill_score(y_true, y_prob):
    """
    Brier skill score : https://en.wikipedia.org/wiki/Brier_score
    Args:
        y_true ([type]): [description]
        y_prob ([type]): [description]
    """
    brier = brier_score_loss(y_true, y_prob)
    dummy_brier = brier_score_loss(
        y_true, np.repeat(y_true.mean(), len(y_true))
    )
    #
    bss = 1 - brier / dummy_brier
    return bss


def get_treatment_metrics(y_true, y_prob):
    """Only for binary treatment

    Args:
        y_true (_type_): _description_
        y_prob (_type_): _description_

    Returns:
        _type_: _description_
    """
    assert (
        y_prob.ndim == 1
    ), "y_prob should be a 1D array, with the score of the positive class."
    return {
        "bss": brier_skill_score(y_true, y_prob),
        "bs": brier_score_loss(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }
