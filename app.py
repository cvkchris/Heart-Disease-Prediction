
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knn

df1 = pd.read_csv("heart(1).csv")
df2 = pd.read_csv('heart.csv')
df = pd.concat([df1, df2], ignore_index=True)

del df['exang']
del df['oldpeak']

features = df.columns[0:12].values.tolist()
x = df[features]
y = df['target']

X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.1, random_state=100)

model  = knn(n_neighbors=3)

model = model.fit(X_train, Y_train)

# y_pred = model.predict(X_test)

def main():
    st.title('Heart Disease Prediction using kNN')
    st.write('This app uses kNN to predict whether a patient has heart disease or not')

    # Collect input features from the user
    age = st.slider('Age', 25, 70)
    sex = st.radio('Sex', ["***MALE***","***FEMALE***"])
    cp = st.slider('Chest Pain',0, 4)
    trestbps = st.slider('Resting Systolic Blood Pressure (during admission in hospital) in mm/Hg', 95, 200)
    chol = st.slider("Cholesterol level in mg/dl",126,564)
    fbs = st.radio("Is Fasting Blood Sugar > 120 mg/dl", ["YES","NO"])
    restecg = st.radio("Resting ECG Result", ['0','1','2','3'])
    thalach = st.slider("Maximum Heart Rate Achieved", 70,200)
    slope = st.radio("Slope of Peak Exercise ST Segment", ['0','1','2'])
    ca = st.radio("Number of Major Vessels colored by Flourosopy", ['0','1','2','3','4'])
    thal = st.radio("Thalassemia", ['Normal', 'Fixed Defect', 'Reversible Defect'])

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
    features = [[age,sex_num,cp,trestbps,chol,fbs_num,restecg, thalach, slope, ca, thal_num]]

    # Make predictions using the kNN model
    prediction = model.predict(features)

    predict_button = st.button("Predict")
    
    if predict_button: 
        if prediction == 1:
            prediction_txt = "Heart Disease"
        else:
            prediction_txt = "NO Heart Disease"

        # Display the prediction
        st.write(f'The Patient has {prediction_txt}')
    

if __name__ == '__main__':
    main()