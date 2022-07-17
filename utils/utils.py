from typing import Optional, Tuple

import missingno as msn
import numpy as np
import pandas as pd
import plotly.io as pio

#pio.templates.default = "plotly_dark"
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (PrecisionRecallDisplay, confusion_matrix, f1_score, precision_score,
                             recall_score,classification_report)


def get_shap_feat_importance(shap_values, data):
    """
    Get Feature Importance from SHAP
    """
    shap_sum = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame([data.columns.tolist(), shap_sum.tolist()]).T
    importance_df.columns = ["features", "shap_importance"]
    importance_df = importance_df.sort_values("shap_importance", ascending=False)
    return importance_df


def check_missing(data: pd.DataFrame, plot: bool = False) -> str:
    """
    The function checks if the data has missing features and displays them
    """

    if plot:
        msn.matrix(data, figsize=(45, 8), sparkline=False)
    missing_total = data.isna().sum().sum()
    print(f"There are in total {missing_total} missing values in the data.")
    if missing_total != 0:
        print(
            f"Following are the list of columns with their corresponding missing number of values.\
          \n{data.isna().sum()[data.isna().sum()>0]}"
        )

def plot_all_dist(data,num_cols,show_static_image=False):
    fig = make_subplots(rows=16, cols=6,
    subplot_titles=tuple(num_cols))

    k=0

    for i in range(1,17):
            for j in range(1,7):
                    fig.add_trace(go.Histogram(x=data[num_cols[k]],name=num_cols[k]),
                    row=i, col=j)
                    k+=1
                    if k ==94:
                            break
                    
    fig.update_layout(width=1200, height=2000,xaxis_tickangle=-90,showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            font=dict(size=10)
            )
    fig.update_annotations(font_size=8)

    if show_static_image:
        fig.show("svg")
    else:
        fig.show()



def plot_dist_bar(
    data1: pd.Series,
    data2: pd.Series,
    title: str,
    index1: Optional[list] = None,
    index2: Optional[list] = None,
    custom_index: bool = False,
    show_static_image=False
) -> go.Figure:
    """
    Uses plotly to plot bar charts with a simplified interface
    """
    fig = make_subplots(1, 2)

    if custom_index:
        fig.append_trace(go.Bar(x=index1, y=data1.values, name="values"), row=1, col=1)

        fig.append_trace(
            go.Bar(x=index2, y=np.round(data2.values, 2) * 100, name="%"), row=1, col=2
        )

    else:
        fig.append_trace(
            go.Bar(x=data1.index, y=data1.values, name="values"), row=1, col=1
        )

        fig.append_trace(
            go.Bar(x=data2.index, y=np.round(data2.values, 2) * 100, name="%"),
            row=1,
            col=2,
        )

    fig.update_layout(
        title=title,
        width=1000,
        height=500,
        yaxis_title="frequency",
        yaxis2_title="%",
        xaxis_tickangle=-90,
        xaxis2_tickangle=-90,
    )

    if show_static_image:
        fig.show("svg")
    else:
        fig.show()

def plot_all_scatter(data, show_static_image=False):
    num_cols = data.select_dtypes(include=np.number).columns
    target0 = data[data["bankruptcy"]==0]
    target1 = data[data["bankruptcy"]==1]
    fig = make_subplots(rows=16, cols=6,
    subplot_titles=tuple(num_cols))

    k=0

    for i in range(1,17):
            for j in range(1,7):
                    fig.add_trace(go.Scatter(x=target0[num_cols[k]],y=target1[num_cols[k]],
                    mode="markers",marker_color=data["bankruptcy"]),
                    row=i, col=j)
                    k+=1
                    if k ==94:
                            break
                    
    fig.update_layout(width=1200, height=2400,xaxis_tickangle=-90,showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            font=dict(size=10)
            )
    fig.update_annotations(font_size=8)

    if show_static_image:
        fig.show("svg")
    else:
        fig.show()





def get_dist(data: pd.DataFrame, col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    perc = data[col].value_counts(normalize=True)
    num = data[col].value_counts()
    return (num, perc)


def read_xlsx(path: str, sheet_name: str) -> pd.DataFrame:
    """
    This function read the XLSX files given a sheetname
    """
    try:
        data = pd.read_excel(open(path, "rb"), sheet_name=sheet_name)
    except FileNotFoundError:
        print("file not found")
    return data


def model_performance(y_true, y_pred, threshold, title, model_type):
    if model_type=="lightgbm":
        fig = PrecisionRecallDisplay.from_predictions(y_true, y_pred, name=title)
        pr_score = precision_score(y_pred=y_pred > threshold, y_true=y_true)
        rec_score = recall_score(y_pred=y_pred > threshold, y_true=y_true)
        _f1_score = f1_score(y_pred=y_pred > threshold, y_true=y_true)
        class_report = classification_report(y_pred=y_pred > threshold, y_true=y_true)
        
        
        print(f"Precision_Score: {pr_score},\
        Recall Score: {rec_score},\
        f1_score: {_f1_score},\n\
        confusion matrix: \n{confusion_matrix(y_true=y_true, y_pred=y_pred>threshold)}\n\
        classification report: \n{class_report}"
        )
        
    else:
        fig = PrecisionRecallDisplay.from_predictions(y_true, y_pred[:, 1], name=title)
        pr_score = precision_score(y_pred=y_pred[:, 1] > threshold, y_true=y_true)
        rec_score = recall_score(y_pred=y_pred[:, 1] > threshold, y_true=y_true)
        _f1_score = f1_score(y_pred=y_pred[:, 1] > threshold, y_true=y_true)
        class_report = classification_report(y_pred=y_pred[:, 1] > threshold, y_true=y_true)


        print(f"Precision_Score: {pr_score},\
                Recall Score: {rec_score},\
                f1_score: {_f1_score},\n\
                confusion matrix: \n{confusion_matrix(y_true=y_true, y_pred=y_pred[:, 1] >threshold)}\n\
                classification report: \n{class_report}")    
        return fig

def detect_outliers_iqr(data):
    outliers_list=[]
    outliers_ind=[]
    q1 = np.quantile(data, q=0.25)
    q3 = np.quantile(data, q=0.75)
    # print(q1, q3)
    IQR = q3-q1
    lwr_bound = q1-(3*IQR)
    upr_bound = q3+(3*IQR)
    # print(lwr_bound, upr_bound)
    for i,k in zip(data,data.index): 
        if (i<lwr_bound or i>upr_bound):
            outliers_list.append(i)
            outliers_ind.append(k)
    return outliers_list,outliers_ind


def treat_outliers(data,outliers_cols):
    for c in outliers_cols:
        val, ind = detect_outliers_iqr(data[c])
        median_ = np.median(data[c])
        upper = np.quantile(data[c],0.95)
        lower = np.quantile(data[c],0.05)
        data[c][ind] = data[c][ind].apply(lambda x: upper if x>median_ else (lower if x<median_ else x))
    return data
    