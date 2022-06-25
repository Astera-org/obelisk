
from typing import Any, Dict, List, AnyStr

import pandas as pd
import numpy as np
from scipy.special import kl_div
from sklearn.metrics import confusion_matrix as c_m
from sklearn.metrics import precision_score

def calc_kl(x:List[int],y:List[int])->float:
    data_table = pd.DataFrame()
    data_table["ground_truth"] = y
    data_table["predicted"] = x
    ground_truth_valuecounts = data_table["ground_truth"].value_counts(normalize=True)
    predicted_valuecounts = data_table["predicted"].value_counts(normalize=True)

    #in the case where predicted exists, but kldivergence says it doesn't (effectively, ignoring NANs) since p(x)/0
    for i in data_table["predicted"].unique(): #if values show up in predicted, but don't exist in ground truth ignore, a bit of a hack
        if (i in data_table["ground_truth"].unique()) == False:
            predicted_valuecounts.drop([i],inplace=True)
    #in the case where exists in ground truth but never appears in predicted 0/p(y)
    for i in data_table["ground_truth"].unique(): #if values show up in predicted, but don't exist in ground truth ignore, a bit of a hack
        if (i in data_table["predicted"].unique()) == False:
            predicted_valuecounts[i] = 0.0
    predicted_valuecounts = predicted_valuecounts/predicted_valuecounts.sum() #renormalize after deleting

    assert len(set(predicted_valuecounts.index).intersection(ground_truth_valuecounts.index)) == len(ground_truth_valuecounts.index), "should have same number of classes"

    ground_truth_distribution = ground_truth_valuecounts.sort_index().values
    prediction_distributed = predicted_valuecounts.sort_index().values
    kl_divergence = kl_div(prediction_distributed,ground_truth_distribution)

    return kl_divergence.sum()
def calc_performance_overtime(x:List[int],y:List[int]):
    pass
def calc_precision(predicted:List[int],ground_truth:List[int]): #calculate the weighted precisoin
    f1 = precision_score(ground_truth,predicted, average="macro")
    return f1
def calc_confusion_matrix(predicted:List[int],ground_truth:List[int], labels:List[AnyStr])->pd.DataFrame:
    confusion_matrix = c_m(ground_truth,predicted)
    return pd.DataFrame(confusion_matrix,index=labels,columns=labels)
def calc_reward(rewards:List[int])->float:
    return np.mean(rewards)