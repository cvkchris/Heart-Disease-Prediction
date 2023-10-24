
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knn

df1 = pd.read_csv("heart(1).csv")
df2 = pd.read_csv('heart.csv')
df = pd.concat([df1, df2], ignore_index=True)

del df['exang']
del df['oldpeak']

features = df.columns[0:11].values.tolist()

x = df[features]
y = df['target']

X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.1, random_state=100)

model  = knn(n_neighbors=3)

model = model.fit(X_train.values, Y_train.values)


def predict():
    # Collect input features from the user
    age = int(st.slider('Age', 25, 70))
    sex = st.radio('Sex', ["***MALE***","***FEMALE***"])
    cp = int(st.slider('Chest Pain',0, 4))
    trestbps = int(st.slider('Resting Systolic Blood Pressure (during admission in hospital) in mm/Hg', 95, 200))
    chol = int(st.slider("Cholesterol level in mg/dl",125,560))
    fbs = st.radio("Is Fasting Blood Sugar > 120 mg/dl", ["YES","NO"])
    restecg = st.radio("Resting ECG Result", [0,1,2,3])
    thalach = int(st.slider("Maximum Heart Rate Achieved", 70,200))
    slope = int(st.radio("Slope of Peak Exercise ST Segment", [0,1,2]))
    ca = int(st.radio("Number of Major Vessels colored by Flourosopy", [0,1,2,3,4]))
    thal = st.radio("Thalassemia", ['Normal', 'Fixed Defect', 'Reversible Defect'])

    sex_num = 0
    fbs_num = 0
    thal_num = 1

    if sex == "MALE":
      sex_num = 1
    else:
      sex_num = 0

    if fbs == "YES":
      fbs_num = 1
    else:
      fbs_num = 0

    if thal == "Normal":
      thal_num = 1
    elif thal == 'Fixed Defect':
      thal_num = 2
    else:
      thal_num = 3

    # Create a feature array with the user's input
    features = np.array([[int(age),int(sex_num),int(cp),int(trestbps),int(chol),int(fbs_num),int(restecg), int(thalach), int(slope), int(ca), int(thal_num)]])
    # Make predictions using the kNN model
    prediction = model.predict(features)

    if prediction == 1:
      prediction_txt = "Heart Disease"
      return prediction_txt
    else:
      prediction_txt = "NO Heart Disease"
      return prediction_txt


def thalach_count():
  df4 = df[['thalach', 'target']].copy()

  df4['target'] = df4['target'].map({1:'Disease', 0:'No Disease'})

  fig = px.histogram(df4, x='thalach', color='target', facet_col='target',
              barmode='group',
              title='Thalach of patients with/without Heart Disease',
              labels={'thalach': 'thalach Count', 'target': 'Target'},
              color_discrete_map={'No Disease': 'blue', 'Disease': 'red'})

  fig.update_layout(title_x=0.5)
  return fig

def age_wise():
  df1 = df[['age', 'target']]
  df1 = df1[df1['target']==1]
  df1 = pd.DataFrame(df1.groupby('age')['target'].count())

  fig = px.bar(df1, x=df1.index, y="target")
  fig.update_layout(
    xaxis_title="Age",
    yaxis_title="Number of Heart Patients",
    title="Number of Heart Patients VS Age"
  )
  fig.update_layout(title_x=0.5)
  return fig

st.markdown(
        f"""
        <style>
        .stApp {{ 
            background-image: url("https://raw.githubusercontent.com/cvkchris/Heart-Disease-Prediction/main/Heart_background.png");
            background-attachment: fixed;
            background-size: cover
        }}
        .sidebar {{
            background-image: linear-gradient(#2e7bcf,#2e7bcf);
        }}    
        </style>
        """,
        unsafe_allow_html=True
)

st.title('Heart Disease Prediction using kNN')
st.sidebar.header("Options")
st.sidebar.divider()
about = st.sidebar.button("About")
heart_disease = st.sidebar.button("Predict Heart Disease")
age_wise_plot = st.sidebar.button('Number of Heart Patients Age-wise')
thalach_count_plot = st.sidebar.button("Thalach Plot of Patients")

if about:
  #About
  st.subheader("About")
  st.write("Welcome to our Heart Disease Prediction website! We are dedicated to utilizing the power of machine learning, specifically the K-Nearest Neighbors (KNN) algorithm, to help you make informed decisions about your heart health. This application is designed to predict whether a patient has any heart disease or not using KNN.")
  #Aim
  st.subheader("Our Aim:")
  st.write("At the core our aim is the well-being of your heart. Heart disease is a prevalent and life-altering health concern, and early detection and proactive management are key to leading a healthy life. Our website is designed to provide you with a reliable and user-friendly tool for assessing your risk of heart disease, based on your personal health data.")
  
  #Privacy
  st.subheader("Your Privacy is Our Priority")
  st.write("We understand that health data is sensitive. Rest assured, your data's privacy and security are paramount to us. The data that is being entered is not collected by us and thus will be erased as soon as you close the app.")

  # Creator's Name
  st.header("Creator Info")
  st.write("This Heart Disease Prediction App was created by : ")
  st.write("- Chris Vinod Kurian")
  st.write("- Gaurav Prakash")

  # Disclaimer
  st.warning("This application is intended for educational and informational purposes. It is not a substitute for professional medical advice. Consult a medical professional for accurate diagnosis and treatment.")

  #Background Image Credits
  st.caption("Image by kjpargeter on Freepik")

if heart_disease:
  try:
    st.subheader("Predict Heart Disease")
    prediction = predict()

    st.subheader("Result")
    st.info(f"The Patient Has {prediction}")
    
  except Exception as e:
      st.text(f"An error occurred: {e}")

if age_wise_plot:
    st.subheader("Number of Heart Patients Age-wise")
    fig = age_wise()
    st.plotly_chart(fig)

if thalach_count_plot:
    st.subheader("Number of Heart Patients W.R.T Thalach")
    fig = thalach_count()
    st.plotly_chart(fig)  