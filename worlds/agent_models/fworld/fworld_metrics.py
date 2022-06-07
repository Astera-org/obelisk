
from typing import Any, Dict, List, AnyStr

import pandas as pd
import numpy as np
from scipy.special import kl_div
from sklearn.metrics import confusion_matrix as c_m
from sklearn.metrics import precision_score

def calc_kl(x:List[int],y:List[int]):
    data_table = pd.DataFrame()
    data_table["ground_truth"] = y
    data_table["predicted"] = x
    ground_truth_distribution = data_table["ground_truth"].value_counts(normalize=True).sort_index().values
    prediction_distributed = data_table["predicted"].value_counts(normalize=True).sort_index().values
    kl_divergence = kl_div(prediction_distributed,ground_truth_distribution)
    return np.sum(kl_divergence)
def calc_performance_overtime(x:List[int],y:List[int]):
    pass
def calc_precision(predicted:List[int],ground_truth:List[int]): #calculate the weighted precisoin
    f1 = precision_score(ground_truth,predicted, average="macro")
    return f1
def calc_confusion_matrix(predicted:List[int],ground_truth:List[int], labels:List[AnyStr]):
    confusion_matrix = c_m(ground_truth,predicted)
    return pd.DataFrame(confusion_matrix,index=labels,columns=labels)
def calc_reward(rewards:List[int]):
    pass