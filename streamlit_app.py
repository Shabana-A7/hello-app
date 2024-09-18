import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load and cache the data
@st.cache_data
def load_data():
    #  path to your CSV file
    return pd.read_csv(r'C:/Users/Administrator/Documents/Python Scripts/streamlit1/predictive_maintenance.csv')
data = load_data()

# Inspect the data
st.subheader('Data Preview')
st.write(data.head(10))
st.write('Column names:', data.columns)

# Preprocess the data
def preprocess_data(df):
    # Encode the target variable
    le = LabelEncoder()
    df['Failure Type'] = le.fit_transform(df['Failure Type'])
    
    # Select features and target
    features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    target = 'Failure Type'
    
    X = df[features]
    y = df[target]
    
    return X, y, le

X, y, label_encoder = preprocess_data(data)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Display model performance
st.subheader('Model Performance')
st.write(f"Accuracy: {accuracy:.2f}")
st.text(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Streamlit UI
st.title('Predictive Maintenance Dashboard')

st.sidebar.header('User Input Features')

def user_input_features():
    air_temp = st.sidebar.slider('Air Temperature [K]', min_value=float(data['Air temperature [K]'].min()), max_value=float(data['Air temperature [K]'].max()), value=float(data['Air temperature [K]'].mean()))
    proc_temp = st.sidebar.slider('Process Temperature [K]', min_value=float(data['Process temperature [K]'].min()), max_value=float(data['Process temperature [K]'].max()), value=float(data['Process temperature [K]'].mean()))
    rpm = st.sidebar.slider('Rotational Speed [rpm]', min_value=float(data['Rotational speed [rpm]'].min()), max_value=float(data['Rotational speed [rpm]'].max()), value=float(data['Rotational speed [rpm]'].mean()))
    torque = st.sidebar.slider('Torque [Nm]', min_value=float(data['Torque [Nm]'].min()), max_value=float(data['Torque [Nm]'].max()), value=float(data['Torque [Nm]'].mean()))
    tool_wear = st.sidebar.slider('Tool Wear [min]', min_value=float(data['Tool wear [min]'].min()), max_value=float(data['Tool wear [min]'].max()), value=float(data['Tool wear [min]'].mean()))
    
    return air_temp, proc_temp, rpm, torque, tool_wear

air_temp, proc_temp, rpm, torque, tool_wear = user_input_features()

# Predict for user input
input_features = [[air_temp, proc_temp, rpm, torque, tool_wear]]
prediction = model.predict(input_features)
prediction_proba = model.predict_proba(input_features)[0]

st.subheader('Prediction')
prediction_label = label_encoder.inverse_transform([prediction[0]])[0]
st.write(f'Prediction: {prediction_label}')

# Show prediction probabilities
st.write('Prediction probabilities:')
for i, label in enumerate(label_encoder.classes_):
    st.write(f"Probability of {label}: {prediction_proba[i]:.2f}")

# Visualize feature importance
st.subheader('Feature Importance')
importances = model.feature_importances_
feature_names = ['Air Temperature [K]', 'Process Temperature [K]', 'Rotational Speed [rpm]', 'Torque [Nm]', 'Tool Wear [min]']
indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)

fig, ax = plt.subplots()
ax.bar(range(len(importances)), [importances[i] for i in indices], align='center')
ax.set_xticks(range(len(importances)))
ax.set_xticklabels([feature_names[i] for i in indices])
ax.set_xlabel('Features')
ax.set_ylabel('Importance')
ax.set_title('Feature Importance')
st.pyplot(fig)
