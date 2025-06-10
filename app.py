import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.title("My first data analyst app")
# Dataset selection
dataset_options = ['titanic', 'iris', 'diamonds', 'tips', 'Upload your own file']
selected_dataset = st.sidebar.selectbox("Select a dataset", dataset_options)

df = None

if selected_dataset == 'titanic':
    df = sns.load_dataset('titanic')
elif selected_dataset == 'iris':
    df = sns.load_dataset('iris')
elif selected_dataset == 'diamonds':
    df = sns.load_dataset('diamonds')
elif selected_dataset == 'tips':
    df = sns.load_dataset('tips')
elif selected_dataset == 'Upload your own file':
    uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=['csv', 'xlsx', 'xls'])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

if df is not None:
    st.write("Preview of selected dataset:")
    st.write(df)
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")
    #display column names and data types
    st.write("Column names and data types:", df.dtypes)
    #print the null values
    if df.isnull().sum().sum()>0:
        st.write("Null values in each column:", df.isnull().sum().sort_values(ascending=False))
    else:
        st.write("No null values found in the dataset.")
    
    # Display basic statistics
    st.write("Basic statistics of the dataset:",df.describe())
    #select a specific column for x and y axis from the dataset and plot a graph
    x_col = st.sidebar.selectbox("Select a column for X-axis", df.columns)
    y_col = st.sidebar.selectbox("Select a column for Y-axis", df.columns)
    plot_type = st.sidebar.selectbox("Select a plot type", ['scatter', 'line', 'bar', 'histogram','pairplot'])
    if plot_type == 'scatter':
        st.scatter_chart(df[[x_col, y_col]])
    
    elif plot_type == 'line':
        st.line_chart(df[[x_col, y_col]])
    elif plot_type == 'bar':
        st.bar_chart(df[[x_col, y_col]])
    elif plot_type == 'histogram':
        if df[x_col].dtype in [np.int64, np.float64]:
            st.write(f"Histogram of {x_col}")
            plt.figure(figsize=(10, 5))
            sns.histplot(df[x_col], bins=30, kde=True)
            st.pyplot(plt)
        else:
            st.write(f"{x_col} is not a numeric column, cannot plot histogram.")
    elif plot_type == 'pairplot':
        st.write("Pairplot of the dataset")
        hue_col = st.sidebar.selectbox("Select a column for hue", df.columns)
        st.pyplot(sns.pairplot(df,hue=hue_col))
    #create heat map of the dataset
    if st.sidebar.checkbox("Show heatmap"):
       st.write("Heatmap of the dataset")
       numeric_cols = df.select_dtypes(include=np.number).columns
       corr_matrix = df[numeric_cols].corr()
       fig, ax = plt.subplots()
       sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
       st.pyplot(fig)
    
            
        

    


 