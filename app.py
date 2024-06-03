import streamlit as st
import pandas as pd
import numpy as np
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_curve, roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_curve
import time

#import models:
from clean_auto import clean_auto
from model_visualize import visualize_lr
from model_files.logisticRegression import logisticRegression
from model_files.decisionTree import decisionTree
from model_files.randomForestClassifier import randomForestClassifier
from model_files.adaBoost import adaBoost
from model_files.XGBoost import XGBoost
from model_files.naiveBayes import naiveBayes
import joblib

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(layout="wide")


def call_cust(custom_list, file):
    loaded_model = joblib.load(file)
    #custom_list1 = [int(float((i))) for i in custom_list]
    custom_list1 = np.array(custom_list)
    test1 = custom_list1.reshape(1,-1)
    result = loaded_model.predict(test1)
    result1 = loaded_model.predict_proba(test1)
    
    
    if int(result) == 0:
        output = "Readmission not required within 30 days!"
    else:
        output = "Readamission required within 30 days!"
    return output, result1.astype(float)

def basicEDA(df):
    import os
    df = df.replace('?', np.nan)
    df.drop_duplicates(inplace=True)
    count = df['race'].value_counts()

    st.markdown("<h6 style='text-align: center;color: #5fb4fb;'>Here's some EDA of your dataset!</h6>", unsafe_allow_html=True)

    plt.bar(count.index, count.values)
    plt.title('Distribution of Race in Patients')
    plt.xlabel('Race')
    plt.ylabel('Count of Patients')

    #Save the plot as an image
    plt.savefig("race_distribution.png")

    plt.figure(figsize=(8, 6))
    sb.countplot(x='race', hue='readmitted', data=df)
    plt.title('Readmission Count by Race')
    plt.xlabel('Race')
    plt.ylabel('Count')

    #Save the plot as an image
    plt.savefig("race_readmit_distribution.png")

    col1, col2 = st.columns([1, 1])
    #First Image
    with col1:
        st.image("race_distribution.png", use_column_width=True, caption="Distribution of Race in Patients")

    #Second Image
    with col2:
        st.image("race_readmit_distribution.png", use_column_width=True, caption="Readmission Count by Race")

    os.remove("race_distribution.png")
    os.remove("race_readmit_distribution.png")



    count = df['time_in_hospital'].value_counts()
    plt.bar(count.index, count.values)
    plt.title('Distribution of time in hospital')
    plt.xlabel('Time in hospital (days)')
    plt.ylabel('Count of Patients')

    #Save the plot as an image
    plt.savefig("time_hospital_distribution.png")

    plt.figure(figsize=(8, 6))
    sb.countplot(x='age', hue='readmitted', data=df)
    plt.title('Readmission Count by Age')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.show()

    #Save the plot as an image
    plt.savefig("age_readmit_distribution.png")

    col1, col2 = st.columns([1, 1])
    #First Image
    with col1:
        st.image("time_hospital_distribution.png", use_column_width=True, caption="Distribution of time in hospital")

    #Second Image
    with col2:
        st.image("age_readmit_distribution.png", use_column_width=True, caption="Readmission Count by Age")

    os.remove("time_hospital_distribution.png")
    os.remove("age_readmit_distribution.png")


    #distribution of readmission
    sb.countplot(x='readmitted', data=df)
    plt.title('Distribution of Readmission')
    plt.xlabel('Readmission class')
    plt.ylabel('Count')
    
    #Save the plot as an image
    plt.savefig("readmit_distribution.png")

    #distribution of patients by age and gender
    plt.figure(figsize=(12,6))
    sb.countplot(x='age', hue='gender', data=df)

    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.title('Gender and Age Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.2)

    #Save the plot as an image
    plt.savefig("gender_age_distribution.png")
    
    col1, col2 = st.columns([1, 1])
    #First Image
    with col1:
        st.image("readmit_distribution.png", use_column_width=True, caption="Distribution of time in hospital")

    #Second Image
    with col2:
        st.image("gender_age_distribution.png", use_column_width=True, caption="Readmission Count by Age")

    os.remove("readmit_distribution.png")
    os.remove("gender_age_distribution.png")    

    st.markdown('<hr style="border-top: 1px solid; width: 75%; margin: 0 auto;">', unsafe_allow_html=True)

    eda_option = st.selectbox("Do you want to peform more EDA on the dataset?", ["--Select--", "Yes", "No"])

    if eda_option == "Yes":
        sb.boxplot(x=df["num_medications"], y=df['age'])
        plt.title('Box Plot of Number of Medications by Age')
        plt.xlabel('Number of Medications')
        plt.ylabel('Age')

        #Save the plot as an image
        plt.savefig("num_medications_boxplot.png")

        sb.boxplot(x=df["num_lab_procedures"], y=df['age'])
        plt.title('Box Plot of Number of Lab Procedures by Age')
        plt.xlabel('Number of Lab Procedures')
        plt.ylabel('Age')

        # Save the plot as an image
        plt.savefig("num_lab_boxplot.png")


        col1, col2 = st.columns([1, 1])
        #First Image
        with col1:
            st.image("num_medications_boxplot.png", use_column_width=True, caption="Box Plot of Number of Medications by Age")

        #Second Image
        with col2:
            st.image("num_lab_boxplot.png", use_column_width=True, caption="Box Plot of Number of Lab Procedures by Age")

        os.remove("num_medications_boxplot.png")
        os.remove("num_lab_boxplot.png")    




        correlation_matrix = df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))

        #Plotting correlation matrix as a heatmap
        sb.heatmap(correlation_matrix, annot=True, ax=ax)

        plt.title('Correlation Matrix')
        plt.xlabel('Features')
        plt.ylabel('Features')

        #Save the plot as an image
        plt.savefig("correlation_matrix.png")


        num_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']
        corr_matrix = df[num_cols].corr()

        #Create a figure and axes
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot the correlation matrix heatmap
        sb.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)

        # Set the title and labels
        plt.title('Correlation Analysis')
        plt.xlabel('Features')
        plt.ylabel('Features')

        #Save the plot as an image
        plt.savefig('correlation_heatmap.png')

        col1, col2 = st.columns([1, 1])
        #First Image
        with col1:
            st.image("correlation_heatmap.png", use_column_width=True, caption="Correlation Matrix Heatmap of numerical features")

        #Second Image
        with col2:
            st.image("correlation_matrix.png", use_column_width=True, caption="Correlation Matrix")

        os.remove("correlation_matrix.png")
        os.remove("correlation_heatmap.png")   





        st.markdown('<hr style="border-top: 1px solid; width: 75%; margin: 0 auto;">', unsafe_allow_html=True) 


        race_readmit_probs = df.groupby('race')['readmitted'].value_counts(normalize=True).mul(100)
        st.markdown("<h6 style='text-align: center;'>Probability of readmission by race</h6>", unsafe_allow_html=True)
        st.text(race_readmit_probs)

        st.markdown('<hr style="border-top: 1px solid; width: 75%; margin: 0 auto;">', unsafe_allow_html=True)
    
    interactive_option = st.selectbox("Do you want to enter into interactive EDA mode?", ["--Select--", "Yes", "No"])
    import plotly.graph_objs as go
    import plotly.express as px
    if interactive_option == "Yes":

        #Distribution of Race in Patients
        count = df['race'].value_counts()
        fig1 = px.bar(df, x=count.index, y=count.values, labels={'x': 'Race', 'y': 'Count of Patients'})
        fig1.update_layout(title='Distribution of Race in Patients')

        #Readmission Count by Race
        fig2 = px.histogram(df, x='race', color='readmitted')
        fig2.update_layout(title='Readmission Count by Race', xaxis_title='Race', yaxis_title='Count')

        #Display the plots
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)

        #Distribution of time in hospital
        count = df['time_in_hospital'].value_counts()
        fig3 = px.bar(df, x=count.index, y=count.values, labels={'x': 'Time in hospital (days)', 'y': 'Count of Patients'})
        fig3.update_layout(title='Distribution of time in hospital')

        #Readmission Count by Age
        fig4 = px.histogram(df, x='age', color='readmitted')
        fig4.update_layout(title='Readmission Count by Age', xaxis_title='Age', yaxis_title='Count')

        #Display the plots
        st.plotly_chart(fig3, use_container_width=True)
        st.plotly_chart(fig4, use_container_width=True)

        #Distribution of Readmission
        fig5 = px.histogram(df, x='readmitted')
        fig5.update_layout(title='Distribution of Readmission', xaxis_title='Readmission class', yaxis_title='Count')

        #Gender and Age Distribution
        fig6 = px.histogram(df, x='age', color='gender', barmode='group')
        fig6.update_layout(title='Gender and Age Distribution', xaxis_title='Age', yaxis_title='Count')

        #Display plots
        st.plotly_chart(fig5, use_container_width=True)
        st.plotly_chart(fig6, use_container_width=True)

        int_eda_option = st.selectbox("Do you want to view more interactive EDA?", ["--Select--", "Yes", "No"])

        if int_eda_option == "Yes":
            fig1 = px.box(df, x='num_medications', y='age', labels={'x': 'Number of Medications', 'y': 'Age'})
            fig1.update_layout(title='Box Plot of Number of Medications by Age')

            #Box Plot of Number of Lab Procedures by Age
            fig2 = px.box(df, x='num_lab_procedures', y='age', labels={'x': 'Number of Lab Procedures', 'y': 'Age'})
            fig2.update_layout(title='Box Plot of Number of Lab Procedures by Age')

            #Display plots
            st.plotly_chart(fig1, use_container_width=True)
            st.plotly_chart(fig2, use_container_width=True)

def plot_treemap(fraud_counts):



    fig_treemap = px.treemap(fraud_counts, path=[px.Constant('All Categories'), 'category'], values='Fraud count',
                 title='Treemap of Fraud Counts by Category',
                 color='Fraud count',
                 color_continuous_scale='Blues')
    fig_treemap.update_layout(margin=dict(t=50, l=25, r=25, b=25))

    st.plotly_chart(fig_treemap, use_container_width=True)

def plot_pie(dataset):
    labels = ["Authentic Transaction", "Fraudulent Transaction"]
    values = dataset["is_fraud"].value_counts()
    colors = ['mediumturquoise', 'lightgreen']
    # pull is given as a fraction of the pie radius
    
    fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0, 0.2], domain={'x': [0.05, 0.95], 'y': [0.01, 0.95]})])

    fig_pie.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=12,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
    
    # Update layout with title adjustments
    fig_pie.update_layout(
        title={
            'text': 'Authentic vs Fraudulent Transactions',
            'y':0.97,  # moves the title upwards
            'x':0.5,  # centers the title
            'xanchor': 'center',
            'yanchor': 'top'
        }

    )
    #fig.show()
    
    # Plot!
    time.sleep(0.5)
    st.plotly_chart(fig_pie, use_container_width=True)

def plot_combined(fraud_stats):
    # Create subplots
    fig_comb = make_subplots(specs=[[{"secondary_y": True}]])

    # Add bar chart for the count of fraud transactions
    fig_comb.add_trace(
        go.Bar(x=fraud_stats['year_month'], y=fraud_stats['fraud_count'], name='Fraud Count'),
        secondary_y=False,
    )

    # Add area chart for the sum of fraud amounts
    fig_comb.add_trace(
        go.Scatter(x=fraud_stats['year_month'], y=fraud_stats['fraud_amount'], name='Fraud Amount', fill='tozeroy'),
        secondary_y=True,
    )

    # Add figure titles and axis labels
    fig_comb.update_layout(
        title_text='Fraud Transactions Analysis Over Time',
        xaxis_title='Year-Month',
        legend_title_text='Metrics'
    )

    # Set y-axes titles
    fig_comb.update_yaxes(title_text='Count of Fraud Transactions', secondary_y=False)
    fig_comb.update_yaxes(title_text='Sum of Fraud Amounts', secondary_y=True)

    # Show the figure
    st.plotly_chart(fig_comb, use_container_width=True)

def plot_geo_map(fraud_counts):

    # Create the scatter geo plot
    fig_geo = go.Figure(data=go.Scattergeo(
        lon = fraud_counts['long'],
        lat = fraud_counts['lat'],
        text = fraud_counts['city'] + ', ' + fraud_counts['state'] + ': ' + fraud_counts['fraud_count'].astype(str) + ' frauds',
        mode = 'markers',
        marker = dict(
            size = fraud_counts['fraud_count'],
            sizemode = 'area',
            sizeref = 2.*max(fraud_counts['fraud_count'])/(40.**2),
            sizemin = 4,
            color = fraud_counts['fraud_count'],
            colorscale = 'Reds',
            line_width = 1,
            line_color='black',
            showscale=True
        )
    ))

    # Update layout for a clearer view
    fig_geo.update_layout(
        title = 'Number of Fraudulent Transactions by City in the US',
        geo_scope='usa',  # limit map scope to USA
    )

    # Show the figure
    st.plotly_chart(fig_geo, use_container_width=True)

def plot_geo_map2(fraud_stats):

    # Create the scatter geo plot
    fig_geo2 = go.Figure(data=go.Scattergeo(
        lon = fraud_stats['long'],
        lat = fraud_stats['lat'],
        text = fraud_stats['city'] + ', ' + fraud_stats['state'] +
            ': ' + fraud_stats['fraud_count'].astype(str) + ' frauds, $' +
            fraud_stats['amt'].astype(str),
        mode = 'markers',
        marker = dict(
            size = fraud_stats['amt'],
            sizemode = 'area',
            sizeref = 2.*max(fraud_stats['amt'])/(40.**2),
            sizemin = 4,
            color = fraud_stats['fraud_count'],
            colorscale = 'Reds',
            line_width = 1,
            line_color='black',
            showscale=True
        )
    ))

    # Update layout for a clearer view
    fig_geo2.update_layout(
        title = 'Fraudulent Transactions by City in the US',
        geo_scope='usa',  # limit map scope to USA
    )

    # Show the figure
    #fig_geo2.show()
    st.plotly_chart(fig_geo2, use_container_width=True)

def stream_clean():
    clean_text0 = "Cleaning...\n"
    clean_text1 = "1.Formatting Data types.\n"
    clean_text2 = "2.Dropping unnecessary columns.\n"
    clean_text3 = "3.Formatting missing values.\n"
    clean_text4 = "4.Encoding cateogrical features."
    #clean_text = clean_text0 + clean_text1 + clean_text2 + clean_text3 + clean_text4 
    clean_text = [clean_text0, clean_text1, clean_text2, clean_text3, clean_text4]
    
    for clean in clean_text:
        for word in clean.split(" "):
            yield word + " "
            time.sleep(0.1)
        yield '\n'

# Creating function for streaming imblance caution
def stream_imbalance_caution():
    imbalance_caution = "There seems to be a class imbalance. Would you like to improve the model performance by using SMOTE?\n"
    imbalance = [imbalance_caution]       
    for clean in imbalance:
        for word in clean.split(" "):
            yield word + " "
            time.sleep(0.1)
        yield '\n'

def basicEDA2(dataset0):

    # converting to datetime format
    dataset0["trans_date_trans_time"] = pd.to_datetime(dataset0["trans_date_trans_time"])
    dataset0["dob"] = pd.to_datetime(dataset0["dob"])

    #Dropping unnecessary columns
    #dataset.drop(columns=['Unnamed: 0','cc_num','first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num','trans_date_trans_time'],inplace=True)
    reqd_cols = ['category', 'amt', 'gender', 'lat', 'long', 'city_pop', 'job', 'unix_time', 'merch_lat', 'merch_long', 'is_fraud']
    

    dataset = dataset0.copy(deep=True)

    time.sleep(1)
    
    dataset = dataset[reqd_cols]

    #Drop all rows that contain missing values 
    dataset = dataset.dropna(ignore_index=True)

    #Encoding cateogrical variables
    encoder = LabelEncoder()
    #dataset["merchant"] = encoder.fit_transform(dataset["merchant"])
    dataset["category"] = encoder.fit_transform(dataset["category"])
    dataset["gender"] = encoder.fit_transform(dataset["gender"])
    dataset["job"] = encoder.fit_transform(dataset["job"])

    st.write_stream(stream_clean)

    view_clean  = st.checkbox("View Sample Dataset After Cleaning")

    if view_clean:
        st.header('Sample Data - Cleaned')
        st.write(dataset.head())


    st.write("\n---- Should show columns and data types.---------\n")

    # Filter and group by 'category' to count 'isFraud = 1' instances
    fraud_counts_tree = dataset0[dataset0['is_fraud'] == 1].groupby('category').size().reset_index(name='Fraud count')

    fraud_df = dataset0[dataset0['is_fraud']== 1]

    fraud_df['year_month'] = fraud_df['trans_date_trans_time'].dt.to_period('M')

    # Group by year_month to calculate counts and sums
    fraud_stats = fraud_df.groupby(['year_month']).agg({
        'is_fraud': 'count',  # Count of fraud transactions
        'amt': 'sum'      # Sum of amounts for fraud transactions
    }).rename(columns={'is_fraud': 'fraud_count', 'amt': 'fraud_amount'})

    
    # Reset index to make 'year_month' a column
    fraud_stats.reset_index(inplace=True)
    fraud_stats['year_month'] = fraud_stats['year_month'].astype(str)  # Convert to string for plotting

    # Find the index of the maximum amount of fraud transactions
    month_max_amt = fraud_stats['fraud_amount'].idxmax()
    month_max_cnt = fraud_stats['fraud_count'].idxmax()

    # Retrieve the month for count and amount using the index
    month_amt = fraud_stats.loc[month_max_amt, 'year_month']
    month_cnt = fraud_stats.loc[month_max_cnt, 'year_month']

    # Group by city and state, and count fraud occurrences
    #fraud_counts_loc = fraud_df.groupby(['city', 'state']).size().reset_index(name='fraud_count')   
    fraud_counts_loc = fraud_df.groupby(['city', 'state', 'lat', 'long']).size().reset_index(name='fraud_count')

    # Group by city to calculate sum of amounts and count of frauds
    fraud_stats_loc = fraud_df.groupby(['city', 'state', 'lat', 'long']).agg({
        'amt': 'sum',  # Sum of amounts for fraud transactions
        'is_fraud': 'count'  # Count of fraud transactions
    }).rename(columns={'is_fraud': 'fraud_count'}).reset_index()

    # Find the index of the maximum amount of fraud transactions
    idx_max_amt = fraud_stats_loc['amt'].idxmax()

    # Retrieve the city name and amount using the index
    max_fraud_city = fraud_stats_loc.loc[idx_max_amt, 'city']
    max_fraud_amt = fraud_stats_loc.loc[idx_max_amt, 'amt']

    # Top 3 Categories by fraud
    top_three_fraud_categories = fraud_counts_tree[['category', 'Fraud count']].sort_values(by='Fraud count', ascending=False).head(3)


    explore_data  = st.toggle("Perform EDA")

    if explore_data:
        st.write("Voila!")

        top_left, top_right = st.columns((2,1.5))

        with top_left:
            plot_treemap(fraud_counts_tree)
        
        with top_right:
            plot_pie(dataset)

        plot_combined(fraud_stats)

        #plot_geo_map(fraud_counts_loc)

        plot_geo_map2(fraud_stats_loc)
    
    
        st.header("Important Findings")
        st.subheader('City with highest amount of Fraudulent transactions')
        st.markdown("{} - ${}".format(max_fraud_city, max_fraud_amt))
        
        st.subheader('Top 3 categories by number of fraudulent transactions')
        #st.write(top_three_fraud_categories, index=False)
        st.dataframe(top_three_fraud_categories.set_index('category'))


        st.subheader('When did the highest number of fraudlent transctions happen?')
        st.markdown("{} : {} transactions".format(month_cnt, fraud_stats['fraud_count'].max()))
        st.markdown("{} : ${}".format(month_amt, fraud_stats['fraud_amount'].max().round(2)))


    st.write_stream(stream_imbalance_caution)

    smote_on  = st.checkbox("Use SMOTE")
    st.session_state.isSMOTE = True
    #isSMOTE = True
    if smote_on:
        st.write("Good Choice!")
        print('isSMOTE: ', st.session_state.isSMOTE)
    else:
        st.session_state.isSMOTE = False

    st.session_state.X = dataset.iloc[:, :-1]
    st.session_state.y = dataset.iloc[:, -1]

    model_option = st.radio("Select the model to train your dataset with:", ("Logistic Regression", "Decision Tree", "Random Forest Classifier", "Ada Boost",
                    "XGBoost", "Naive Bayes"), help = "Logistic Regression: Linear model for classification and regression." +
                          "\n\n Decision Tree: Tree-based model that makes decisions based on feature values." + 
                          "\n\n Random Forest Classifier: Ensemble of decision trees for classification." +
                            "\n\n Ada Boost: Ensemble model that combines weak learners to create a strong learner." + 
                            "\n\n XGBoost: Optimized gradient boosting framework for improved model performance." + 
                            "\n\n Naive Bayes: Probabilistic model based on Bayes theorem for classification.")
    
    if st.button("Train Model!"):

        if model_option == "Logistic Regression":
            
            st.write(f"<p style='color:#0FF900'><strong>Training with Logistic Regression!</strong></p>", unsafe_allow_html=True)
            
            model, X, y, X_train, X_test, y_train, y_test = logisticRegression(st.session_state.X, st.session_state.y, st.session_state.isSMOTE)
            visualize_lr(model, X, y, X_train, X_test, y_train, y_test)


        elif model_option == "Decision Tree":

            st.write(f"<p style='color:#0FF900'><strong>Training with Decision Tree!</strong></p>", unsafe_allow_html=True)
            
            decisionTree(st.session_state.X, st.session_state.y, st.session_state.isSMOTE)
        
        elif model_option == "Random Forest Classifier":
            
            st.write(f"<p style='color:#0FF900'><strong>Training with Random Forest Classifier!</strong></p>", unsafe_allow_html=True)
        
            randomForestClassifier(st.session_state.X, st.session_state.y, st.session_state.isSMOTE)

        elif model_option == "Ada Boost":
            
            st.write(f"<p style='color:#0FF900'><strong>Training with AdaBoost!</strong></p>", unsafe_allow_html=True)
            
            adaBoost(st.session_state.X, st.session_state.y, st.session_state.isSMOTE)

        elif model_option == "XGBoost":

            st.write(f"<p style='color:#0FF900'><strong>Training with XGBoost!</strong></p>", unsafe_allow_html=True)
            
            XGBoost(st.session_state.X, st.session_state.y, st.session_state.isSMOTE)

        
        elif model_option == "Naive Bayes":
            
            st.write(f"<p style='color:#0FF900'><strong>Training with Naive Bayes!</strong></p>", unsafe_allow_html=True)
            
            naiveBayes(st.session_state.X, st.session_state.y, st.session_state.isSMOTE)




def app():


    data_option = st.selectbox("Select an option to make prediction", ["--Select--", "Default Dataset", "Upload Dataset - CSV"])

    if data_option == "Default Dataset":
        #dataset = pd.read_csv('./data/credit_card_transactions/fraudTrain.csv')
        dataset = pd.read_csv('./data/credit_card_transactions/fraudTest_mini.csv')

        st.header('About the dataset')

        about_data = """This is a simulated credit card transaction dataset containing legitimate and fraud transactions from the duration 1st Jan 2019 - 31st Dec 2020.
          It covers credit cards of 1000 customers doing transactions with a pool of 800 merchants."""
        st.markdown(about_data)

        st.link_button("Go to Data", "https://www.kaggle.com/datasets/kartik2112/fraud-detection/data")

        st.header('Sample Raw Data')
        st.write(dataset.head(10))
        #st.write(dataset.columns)
        
        basicEDA2(dataset)


    elif data_option == "Upload Dataset":
        st.markdown(" Feature releasing soon! Please choose from other options!")
        '''st.markdown("<h6 style='text-align: center;'>Upload your dataset here!</h6>", unsafe_allow_html=True)

        dataset = st.file_uploader("Choose your dataset - CSV file", type="csv")

        X = None
        y = None

        if dataset is not None:
            try:
                data = pd.read_csv(dataset)

                st.write(f"<p style='color:#0FF900'><strong>All required columns are present!</strong></p>", unsafe_allow_html=True)

                st.markdown("<h6 style='text-align: center;'>Displaying first 5 rows of the dataset</h6>", unsafe_allow_html=True)

                st.write(data.head())
            
                basicEDA(data)

                tooltip_text = {
                    'Clean Automatically': 'This is the first option.',
                    'Clean Manually': 'This is the second option.'
                }

                clean_option = st.radio("Select data cleaning option", ["Clean Automatically", "Already Cleaned"], help='Clean Automatically: Select this option to automatically clean the data using predefined cleaning rules. \n\nAlready Cleaned: Select this option if the data is already cleaned and ready for modeling.')

                #tooltip = tooltip_text.get(clean_option, '')
                #st.markdown(f'<span title="{tooltip}">{clean_option}</span>', unsafe_allow_html=True)
                if st.button("Clean!"):
                    if clean_option == "Clean Automatically":
                        st.session_state.X, st.session_state.y = clean_auto(data)
                        #logisticRegression(X, y)
                        st.write(f"<p style='color:#0FF900'><strong>Automatic cleaning successful!</strong></p>", unsafe_allow_html=True)

                model_option = st.radio("Select the model to train your dataset with:", ("Logistic Regression", "Decision Tree", "Random Forest Classifier", "Ada Boost", "XGBoost", "Naive Bayes"), help = 'Logistic Regression: Linear model for classification and regression. \n\n Decision Tree: Tree-based model that makes decisions based on feature values. \n\n Random Forest Classifier: Ensemble of decision trees for classification. \n\n Ada Boost: Ensemble model that combines weak learners to create a strong learner. \n\n XGBoost: Optimized gradient boosting framework for improved model performance. \n\n Naive Bayes: Probabilistic model based on Bayes theorem for classification.')

                if st.button("Train Model!"):
                    #with st.spinner(f"Training with {model_option}..."):
                    #st.write("Clicked Train Model!")
                    #if st.session_state.X is None:
                        #st.write(f"<p style='color:red'><strong>Please clean dataset before training!</strong></p>", unsafe_allow_html=True)
                    #else:
                    if model_option == "Logistic Regression":
                        
                        st.write(f"<p style='color:#0FF900'><strong>Training with Logistic Regression!</strong></p>", unsafe_allow_html=True)
                        #X, y = clean_auto(data)
                        logisticRegression(st.session_state.X, st.session_state.y)

                    elif model_option == "Decision Tree":

                        st.write(f"<p style='color:#0FF900'><strong>Training with Decision Tree!</strong></p>", unsafe_allow_html=True)
                        #X, y = clean_auto(data)
                        decisionTree(st.session_state.X, st.session_state.y)
                    
                    elif model_option == "Random Forest Classifier":
                        
                        st.write(f"<p style='color:#0FF900'><strong>Training with Random Forest Classifier!</strong></p>", unsafe_allow_html=True)
                    # X, y = clean_auto(data)
                        randomForestClassifier(st.session_state.X, st.session_state.y)

                    elif model_option == "Ada Boost":
                        
                        st.write(f"<p style='color:#0FF900'><strong>Training with AdaBoost!</strong></p>", unsafe_allow_html=True)
                        #X, y = clean_auto(data)
                        adaBoost(st.session_state.X, st.session_state.y)

                    elif model_option == "XGBoost":

                        st.write(f"<p style='color:#0FF900'><strong>Training with XGBoost!</strong></p>", unsafe_allow_html=True)
                        #X, y = clean_auto(data)
                        XGBoost(st.session_state.X, st.session_state.y)

                    
                    elif model_option == "Naive Bayes":
                        
                        st.write(f"<p style='color:#0FF900'><strong>Training with Naive Bayes!</strong></p>", unsafe_allow_html=True)
                        #X, y = clean_auto(data)
                        naiveBayes(st.session_state.X, st.session_state.y)

            except Exception as e:
                st.write(f"Error in reading the CSV file: {e}")'''

    