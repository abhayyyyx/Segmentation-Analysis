import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import silhouette_score
warnings.filterwarnings('ignore')

# Sidebar for user input
with st.sidebar:
    st.markdown("Customer Segmentation")
    user_input = st.selectbox('Please Select', ('Visualization', 'Model'))

# Load the data
try:
    df = pd.read_excel('C:\\Users\\SUMAN\\Desktop\\RITU\\data_cleaned.xlsx')
except FileNotFoundError:
    st.error("data_cleaned.xlsx not found. Please make sure the file is in the correct directory.")
    st.stop()

classif_data = df[['Education', 'Marital_Status', 'Income', 'MntWines', 'MntFruits',
                   'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
                   'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
                   'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
                   'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
                   'AcceptedCmp2', 'Age', 'total_children', 'Clusters']]

# Encode categorical features
le_education = LabelEncoder()
le_marital_status = LabelEncoder()

classif_data['Education'] = le_education.fit_transform(classif_data['Education'])
classif_data['Marital_Status'] = le_marital_status.fit_transform(classif_data['Marital_Status'])

# Split the data into features and target
X = classif_data.drop('Clusters', axis=1)
y = classif_data['Clusters']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Random Forest model
rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)
rf.fit(X_train, y_train)

# Define cluster labels (adjust these labels based on your clustering results)
cluster_labels = {
    0: 'Cluster A',
    1: 'Cluster B',
    2: 'Cluster C',
    }

if user_input == 'Visualization':
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Visualization of Clustering")

    st.subheader('Elbow Method')
    # Clustering using KMeans
    WCSS = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(classif_data)
        WCSS.append(kmeans.inertia_)

    plt.plot(range(1, 11), WCSS, marker='o', linestyle='--')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for Optimal k')
    st.pyplot()

    # Silhouette Score method to find optimal k
    st.subheader('Silhouette Method')
    silhouette_scores = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(classif_data)
        silhouette_scores.append(silhouette_score(classif_data, labels))

    plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='--')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method for Optimal k')
    st.pyplot()

else:
    with st.form("prediction_form"):
        Education = st.selectbox("Education:", options=['Undergraduate', 'Graduate', 'Postgraduate'])
        Marital_Status = st.selectbox("Marital Status:", options=['Married', 'Single'])
        Income = st.number_input("Income:", min_value=0)
        MntWines = st.number_input("MntWines:", min_value=0)
        MntFruits = st.number_input("MntFruits:", min_value=0)
        MntMeatProducts = st.number_input("MntMeatProducts:", min_value=0)
        MntFishProducts = st.number_input("MntFishProducts:", min_value=0)
        MntSweetProducts = st.number_input("MntSweetProducts:", min_value=0)
        MntGoldProds = st.number_input("MntGoldProds:", min_value=0)
        NumDealsPurchases = st.number_input("NumDealsPurchases:", min_value=0)
        NumWebPurchases = st.number_input("NumWebPurchases:", min_value=0)
        NumCatalogPurchases = st.number_input("NumCatalogPurchases:", min_value=0)
        NumStorePurchases = st.number_input("NumStorePurchases:", min_value=0)
        NumWebVisitsMonth = st.number_input("NumWebVisitsMonth:", min_value=0)
        AcceptedCmp3 = st.number_input("AcceptedCmp3:", min_value=0, max_value=1)
        AcceptedCmp4 = st.number_input("AcceptedCmp4:", min_value=0, max_value=1)
        AcceptedCmp5 = st.number_input("AcceptedCmp5:", min_value=0, max_value=1)
        AcceptedCmp1 = st.number_input("AcceptedCmp1:", min_value=0, max_value=1)
        AcceptedCmp2 = st.number_input("AcceptedCmp2:", min_value=0, max_value=1)
        Age = st.number_input("Age:", min_value=0)
        TotalChildren = st.number_input("TotalChildren:", min_value=0)

        submitted = st.form_submit_button("Predict")

        if submitted:
            # Prepare the input data
            input_data = [[Education, Marital_Status,
                           Income, MntWines, MntFruits, MntMeatProducts, MntFishProducts,
                           MntSweetProducts, MntGoldProds, NumDealsPurchases, NumWebPurchases,
                           NumCatalogPurchases, NumStorePurchases, NumWebVisitsMonth,
                           AcceptedCmp3, AcceptedCmp4, AcceptedCmp5, AcceptedCmp1, AcceptedCmp2,
                           Age, TotalChildren]]

            # Create a DataFrame
            input_df = pd.DataFrame(input_data, columns=X.columns)

            # Encode categorical variables using the fitted encoders
            input_df['Education'] = le_education.transform([input_df['Education'][0]])[0]
            input_df['Marital_Status'] = le_marital_status.transform([input_df['Marital_Status'][0]])[0]

            try:
                # Making prediction
                prediction = rf.predict(input_df)
                cluster_label = cluster_labels.get(prediction[0], 'Unknown Cluster')
                st.write(f"The predicted customer cluster is: *{cluster_label}*")
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
	# 1.	Adjust the cluster_labels dictionary to match your actual cluster labels.
	# 2.	The model now returns a cluster label instead of just a cluster number.
	# 3.	Ensure that the education and marital_status columns are encoded consistently during both training and prediction.