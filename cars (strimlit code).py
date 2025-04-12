import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# App Title and Credit
st.title("🚗 Car Evaluation Classifier using Random Forest & Streamlit")
st.write("Created by Aftab | Predict the car condition based on various features using Machine Learning.")

# File uploader (optional if user wants to try different data)
uploaded_file = st.file_uploader(" Upload your car.csv file", type=['csv'])

# Load default dataset from UCI if no file uploaded
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    return pd.read_csv(url, names=columns)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = load_data()

# Encode categorical columns
df_encoded = df.apply(lambda col: pd.factorize(col)[0])

# Split data
X = df_encoded.iloc[:, :-1]
y = df_encoded.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Display Accuracy
accuracy = model.score(X_test, y_test)
st.success(f"Model Accuracy: {accuracy*100:.2f}%")

# Prediction UI
st.subheader("Predict Car Condition")

input_data = []
for column in df.columns[:-1]:
    value = st.selectbox(f"{column}", df[column].unique())
    input_data.append(value)

# Encode user inputs
input_encoded = [pd.Series(df[column].unique()).tolist().index(val) for column, val in zip(df.columns[:-1], input_data)]

# Predict and decode
prediction = model.predict([input_encoded])[0]
decoded_label = pd.Series(df[df.columns[-1]].unique())[prediction]

st.success(f"✅ Predicted Condition: {decoded_label}")

