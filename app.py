
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


df1 = pd.read_csv("heart(1).csv")
df2 = pd.read_csv('heart.csv')
df3 = pd.read_csv("heart(2).csv")

df = pd.concat([df1, df2, df3], ignore_index=True)

df = df.drop_duplicates()

del df['ca']
del df['thal']
del df['exang']
del df['oldpeak']

features = df.columns[0:9].values.tolist()

x = df[features].to_numpy()
y = df['target'].to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.1, random_state=100)

knn_model  = knn(n_neighbors=141)
knn_model = knn_model.fit(X_train, Y_train)

gnb = GaussianNB()
gnb.fit(X_train, Y_train)

lr = LogisticRegression(max_iter = 1000)
lr.fit(X_train, Y_train)

def predict(model):
    
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

    # Create a feature array with the user's input
    features = np.array([[age,sex_num,cp,trestbps,chol,fbs_num,restecg,thalach,slope]])

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
              labels={'thalach': 'thalach Count', 'target': 'Target'},
              color_discrete_map={'No Disease': 'blue', 'Disease': 'red'})

  return fig

def age_wise():
  df5 = df[['age', 'target']]
  df5 = df5[df5['target']==1]
  df5 = pd.DataFrame(df5.groupby('age')['target'].count())

  fig = px.bar(df5, x=df5.index, y="target")
  fig.update_layout(
    xaxis_title="Age",
    yaxis_title="Number of Heart Patients",
    title="Number of Heart Patients VS Age"
  )
  return fig

def gender_distribution():
  df6 = df[['sex', 'target']].copy()
  df6['sex'] = df6['sex'].map({1:'Male', 0:'Female'})
  df6['target'] = df6['target'].map({1:'Diseased', 0:'Normal'})

  # Count the gender distribution
  gender_distribution = df6.value_counts().reset_index()
  gender_distribution.columns = ['Gender','Target','Count']

  gender_distribution['Patients'] = gender_distribution['Target']+ " " + gender_distribution['Gender']

  fig = px.pie(gender_distribution, values='Count', names='Patients',
              title='Distribution of Patients by Gender')

  return fig

st.markdown(
        f"""
        <style>
        .stApp {{ 
            background-image: url("https://raw.githubusercontent.com/cvkchris/Heart-Disease-Prediction/main/Heart_background.png");
            background-attachment: fixed;
            background-size: cover
        }}
        div.stSidebarUserContent {{
            background-image: linear-gradient(#2e7bcf,#2e7bcf);
        }}

        button.st-emotion-cache-hc3laj{{
            width : 190px;
        }}    
        </style>
        """,
        unsafe_allow_html=True
)


st.title('Heart Disease Prediction')
st.sidebar.header("Options")
st.sidebar.divider()
about = st.sidebar.button("About")
knn_button = st.sidebar.button('KNN')
nb_button = st.sidebar.button('Naive Bayes')
lr_button = st.sidebar.button('Logistic Regression')
age_wise_plot = st.sidebar.button('Number of Heart Patients Age-wise')
thalach_count_plot = st.sidebar.button("Thalach Plot of Patients")
gender_distribution_plot = st.sidebar.button("Gender Distribution")

if "knn" not in st.session_state:
  st.session_state.knn = False
  st.session_state.nb = False
  st.session_state.lr = False

if about:
  st.session_state.knn = False
  st.session_state.nb = False
  st.session_state.lr = False

if knn_button:
  st.session_state.knn = True
  st.session_state.nb = False
  st.session_state.lr = False

  st.header("Predict Heart Disease") 
  st.subheader("KNN Model") 
  prediction = predict(knn_model)
  st.subheader("Result")
  st.info(f"The Patient Has {prediction}")

elif nb_button:
  st.session_state.nb = True
  st.session_state.lr = False
  st.session_state.knn = False

  st.header("Predict Heart Disease") 
  st.subheader("Naive Bayes Model") 
  prediction = predict(gnb)
  st.subheader("Result")
  st.info(f"The Patient Has {prediction}")

elif lr_button:
  st.session_state.lr = True
  st.session_state.nb = False
  st.session_state.knn = False

  st.header("Predict Heart Disease") 
  st.subheader("Logistic Regrssion Model") 
  prediction = predict(lr)
  st.subheader("Result")
  st.info(f"The Patient Has {prediction}")

elif age_wise_plot:
  st.session_state.knn = False
  st.session_state.nb = False
  st.session_state.lr = False
  st.subheader("Number of Heart Patients Age-wise")
  fig = age_wise()
  st.plotly_chart(fig)

elif thalach_count_plot:
  st.session_state.knn = False
  st.session_state.nb = False
  st.session_state.lr = False
  st.subheader("Number of Heart Patients W.R.T Thalach")
  fig = thalach_count()
  st.plotly_chart(fig) 

elif gender_distribution_plot:
  st.session_state.knn = False
  st.session_state.nb = False
  st.session_state.lr = False
  st.subheader("Gender Distribution of Heart Diseased Patients")
  fig = gender_distribution()
  st.plotly_chart(fig)   

elif st.session_state.knn == True:
  st.header("Predict Heart Disease") 
  st.subheader("KNN Model") 
  prediction = predict(knn_model)
  st.subheader("Result")
  st.info(f"The Patient Has {prediction}")

elif st.session_state.nb == True:
  st.header("Predict Heart Disease") 
  st.subheader("Naive Bayes Model") 
  prediction = predict(gnb)
  st.subheader("Result")
  st.info(f"The Patient Has {prediction}")

elif st.session_state.lr == True:
  st.header("Predict Heart Disease") 
  st.subheader("Logistic Regression Model") 
  prediction = predict(lr)
  st.subheader("Result")
  st.info(f"The Patient Has {prediction}")

else:
  st.session_state.knn = False
  st.session_state.nb = False
  st.session_state.lr = False
  #About
  st.subheader("About")
  st.write("Welcome to our Heart Disease Prediction website! We are dedicated to utilizing the power of machine learning, specifically using:")
  st.write("- K-Nearest Neighbors (KNN) model")
  st.write("- Naive Bayes model")
  st.write("- Logistic Regression model") 
  st.write("to help you make informed decisions about your heart health. This application is designed to predict whether a patient has any heart disease or not from the input parameters.")
  #Aim
  st.subheader("Our Aim:")
  st.write("At the core our aim is the well-being of your heart. Heart disease is a prevalent and life-altering health concern, and early detection and proactive management are key to leading a healthy life. Our website is designed to provide you with a reliable and user-friendly tool for assessing your risk of heart disease, based on your personal health data.")
  
  #Privacy
  st.subheader("Your Privacy is Our Priority")
  st.write("We understand that health data is sensitive. Rest assured, your data's privacy and security are paramount to us. The data that is being entered is not collected by us and thus will be erased as soon as you close the app.")

  # Creator's Name
  st.subheader("Creator Info")
  st.write("This Heart Disease Prediction App was created by : ")
  st.write("- Chris Vinod Kurian")
  st.write("- Gaurav Prakash")

  # Disclaimer
  st.warning("This application is intended for educational and informational purposes. It is not a substitute for professional medical advice. Consult a medical professional for accurate diagnosis and treatment.")

  #Background Image Credits
  st.caption("Image by kjpargeter on Freepik")    