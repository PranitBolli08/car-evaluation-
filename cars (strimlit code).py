import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# App Title
st.title("🚗 Car Evaluation using KNN Regression & Streamlit")
st.write("Predict the car condition using K-Nearest Neighbors Regression based on various features.")


# File uploader (optional, not used directly here)
uploaded_file = st.file_uploader("📁 Upload your car.csv file", type=['csv'])

@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    return pd.read_csv(url, names=columns)

df = load_data()

# Encoding categorical columns
df_encoded = df.apply(lambda col: pd.factorize(col)[0])

# Splitting data
X = df_encoded.iloc[:, :-1]
y = df_encoded.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model: KNN Regressor
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train, y_train)

# Accuracy (R² Score)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
st.success(f"📈 R² Score (Model Accuracy): {r2*100:.2f}%")

# Prediction UI
st.subheader("🧪 Predict Car Condition")

input_data = []
for column in df.columns[:-1]:
    value = st.selectbox(f"{column}", df[column].unique())
    input_data.append(value)

# Convert input to encoded form
input_encoded = [pd.Series(df[column].unique()).tolist().index(val) for column, val in zip(df.columns[:-1], input_data)]

# Predict using KNN Regressor
predicted_value = model.predict([input_encoded])[0]
rounded_prediction = round(predicted_value)

# Decode prediction back to class label
class_labels = pd.Series(df['class'].unique())
decoded_label = class_labels.iloc[rounded_prediction] if rounded_prediction < len(class_labels) else "Unknown"

st.success(f"✅ Predicted Condition: {decoded_label}")

