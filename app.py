import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------- Page Config ----------------
st.set_page_config(page_title="ML Classification App", layout="wide")

st.title("🤖 Machine Learning Classification App")

# ---------------- Sample Dataset ----------------
@st.cache_data
def load_data():
    data = {
        "Feature1": [5, 7, 8, 2, 3, 9, 4, 6],
        "Feature2": [1, 2, 3, 6, 7, 4, 5, 8],
        "Target":   [0, 0, 1, 1, 1, 0, 1, 0]
    }
    return pd.DataFrame(data)

df = load_data()

st.subheader("📊 Dataset")
st.dataframe(df)

# ---------------- Split Data ----------------
X = df[["Feature1", "Feature2"]]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------- Model ----------------
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.subheader("📈 Model Performance")
st.write(f"Accuracy: **{accuracy:.2f}**")

# ---------------- Confusion Matrix ----------------
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
ax.matshow(cm)
for (i, j), val in np.ndenumerate(cm):
    ax.text(j, i, f"{val}", ha='center', va='center')
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

# ---------------- User Input ----------------
st.sidebar.header("🔎 Make Prediction")

feature1 = st.sidebar.number_input("Feature1", min_value=0)
feature2 = st.sidebar.number_input("Feature2", min_value=0)

if st.sidebar.button("Predict"):
    input_data = np.array([[feature1, feature2]])
    prediction = model.predict(input_data)[0]
    st.sidebar.success(f"Prediction: {prediction}")
