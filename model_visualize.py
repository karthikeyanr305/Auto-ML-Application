import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
#from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
import seaborn as sb
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_curve, roc_auc_score, confusion_matrix, auc
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import SMOTE
import joblib
import shap

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, confusion_matrix, f1_score

def plot_shap(model, X_train, X_test):
    # Initialize the SHAP Explainer
    explainer = shap.Explainer(model, X_train)

    # Compute SHAP values for the test set
    shap_values = explainer(X_test)

    # Convert SHAP values to DataFrame
    shap_df = pd.DataFrame(shap_values.values, columns=X_test.columns)

    
    # Calculate mean absolute SHAP values
    mean_shap = shap_df.abs().mean().sort_values(ascending=False)
    
    # Convert mean_shap Series to DataFrame for better handling
    mean_shap_df = mean_shap.reset_index()
    mean_shap_df.columns = ['Feature', 'Mean SHAP Value']

    # Sort the DataFrame by 'Mean SHAP Value' in descending order
    mean_shap_df = mean_shap_df.sort_values(by='Mean SHAP Value', ascending=False)

    # Create the bar plot using Plotly Express
    fig_shap = px.bar(mean_shap_df, x='Mean SHAP Value', y='Feature',
                labels={'Feature': 'Feature', 'Mean SHAP Value': 'Mean SHAP Value'},
                title='Mean Absolute SHAP Values',
                orientation='h', text='Mean SHAP Value')  # Add text parameter for displaying values

    # Adjust text annotation position
    fig_shap.update_traces(texttemplate='%{text:.2f}', textposition='outside')

    # Update layout for a clearer view
    fig_shap.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',
                    xaxis_title='Mean SHAP Value',
                    yaxis_title='Feature')
    
    st.plotly_chart(fig_shap, use_container_width=True)


def plot_hist_score(y, y_score):
    # The histogram of scores compared to true labels
    fig_hist = px.histogram(
        x=y_score, color=y, nbins=50,
        title='Histogram of Scores',
        labels=dict(color='True Labels', x= 'Score', y= 'Count')
    )

    #fig_hist.show()
    st.plotly_chart(fig_hist, use_container_width=True)

def plot_threshold(fpr, tpr, thresholds):

    # Evaluating model performance at various thresholds
    df_thres = pd.DataFrame({
        'False Positive Rate': fpr,
        'True Positive Rate': tpr
    }, index=thresholds)
    df_thres.index.name = "Thresholds"
    df_thres.columns.name = "Rate"

    fig_thresh = px.line(
        df_thres, title='TPR and FPR at every threshold',
        width=700, height=500
    )

    fig_thresh.update_yaxes(scaleanchor="x", scaleratio=1)
    fig_thresh.update_xaxes(range=[0, 1], constrain='domain')
    #fig_thresh.show()
    st.plotly_chart(fig_thresh, use_container_width=True)


def plot_auc_roc(fpr, tpr):
    fig_auc = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
    )
    fig_auc.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig_auc.update_yaxes(scaleanchor="x", scaleratio=1)
    fig_auc.update_xaxes(constrain='domain')
    #fig.show()
    st.plotly_chart(fig_auc, use_container_width=True)

def plot_pr_curve(precision, recall, fpr, tpr):

    fig_pr = px.area(
    x=recall, y=precision,
    title=f'Precision-Recall Curve (AUC={auc(recall, precision):.4f})',
    labels=dict(x='Recall', y='Precision'),
    width=700, height=500
    )
    fig_pr.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=1, y1=0
    )
    fig_pr.update_yaxes(scaleanchor="x", scaleratio=1)
    fig_pr.update_xaxes(constrain='domain')

    #fig_pr.show()
    st.plotly_chart(fig_pr, use_container_width=True)

def plot_confusion_matrix(cm):

    # Convert the confusion matrix to DataFrame for better labeling in Plotly Express
    df_cm = pd.DataFrame(cm, index=['Authentic', 'Fraud'], columns=['Predicted Authentic', 'Predicted Fraud'])

    # Create the heatmap using Plotly Express
    fig_cm = px.imshow(df_cm,
                    labels=dict(x="Predicted Label", y="True Label", color="Number of Samples"),
                    x=['Authentic', 'Fraud'],
                    y=['Authentic', 'Fraud'],
                    text_auto=True)  # This will annotate the heatmap with the numeric values

    # Add titles
    fig_cm.update_layout(title="Confusion Matrix",
                    xaxis_title="Predicted Label",
                    yaxis_title="True Label")

    # Show the plot
    #fig_cm.show()
    st.plotly_chart(fig_cm, use_container_width=True)


def visualize_lr(LR, X, y, X_train, X_test, y_train, y_test):

    LR_pred = LR.predict(X_test)
    classification_report_str = classification_report(y_test, LR_pred)
    accuracy_lr = accuracy_score(y_test, LR_pred)
    precision_lr = precision_score(y_test, LR_pred)
    recall_lr = recall_score(y_test, LR_pred)
    # F1 Score: The weighted average of Precision and Recall.
    f1_lr = f1_score(y_test, LR_pred)

    st.markdown("<h3 style='text-align: center;color: #5fb4fb;'><u>METRICS</u></h3>", unsafe_allow_html=True)
    #st.text(classification_report_str)

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Accuracy", f"{accuracy_lr:.2f}")
    metric2.metric("Precision", f"{precision_lr:.2f}")
    metric3.metric(" Recall", f"{recall_lr:.2f}")
    metric4.metric("F-1 Score", f"{f1_lr:.2f}")

    st.markdown('<hr style="border-top: 1px solid; width: 75%; margin: 0 auto;">', unsafe_allow_html=True)

    #st.markdown('<h3 style="color: #5fb4fb;">PLOTS:</h3>', unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center;color: #5fb4fb;'><u>PLOTS</u></h3>", unsafe_allow_html=True)

    #Histogram of Scores
    y_score = LR.predict_proba(X)[:, 1]

    # AUC ROC 
    fpr, tpr, thresholds = roc_curve(y, y_score)

    # Precision Recall Curve
    precision, recall, thresholds2 = precision_recall_curve(y, y_score)

    # Confusion Matrix
    cm = confusion_matrix(y_test, LR_pred)

    top_left, top_right = st.columns(2)
    mid_left, mid_right = st.columns(2)
    bottom_left, bottom_right  = st.columns(2)

    with top_left:
        plot_shap(LR, X_train, X_test)

    with top_right:
        plot_hist_score(y, y_score)

    with mid_left:
        plot_threshold(fpr, tpr, thresholds)
    
    with mid_right:
        plot_confusion_matrix(cm)
    
    with bottom_left:
        plot_auc_roc(fpr, tpr)

    with bottom_right:
        plot_pr_curve(precision, recall, fpr, tpr)