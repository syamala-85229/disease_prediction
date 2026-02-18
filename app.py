# ==========================================================
# M.TECH MULTI-DISEASE DIAGNOSTIC SYSTEM - STREAMLIT APP
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# ----------------------------------------------------------
# MULTI-USER LOGIN DATA
# ----------------------------------------------------------

logins = [
    {"username": "admin", "password": "admin@123"},
    {"username": "doctor1", "password": "doctor1@123"},
    {"username": "doctor2", "password": "doctor2@123"},
    {"username": "researcher1", "password": "research@123"},
]


# ----------------------------------------------------------
# LOGIN PAGE
# ----------------------------------------------------------

def medical_login():

    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(to right, #e6f2ff, #f9fcff);
        }
        .login-card {
            background-color: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
            width: 400px;
            margin: auto;
            margin-top: 100px;
        }
        .title {
            text-align: center;
            font-size: 26px;
            font-weight: bold;
            color: #0b6fa4;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    st.markdown('<div class="title">🩺 AI Medical Diagnostic System</div>', unsafe_allow_html=True)

    username = st.text_input("User ID")
    password = st.text_input("Password", type="password")

    if st.button("🔐 Secure Login"):

        user_found = False

        for user in logins:
            if username == user["username"] and password == user["password"]:
                st.session_state["authenticated"] = True
                st.session_state["user"] = username
                user_found = True
                break

        if not user_found:
            st.error("Invalid Credentials")

    st.markdown('</div>', unsafe_allow_html=True)


# ----------------------------------------------------------
# SESSION HANDLING
# ----------------------------------------------------------

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if "user" not in st.session_state:
    st.session_state["user"] = ""

if not st.session_state["authenticated"]:
    medical_login()
    st.stop()

# Sidebar after login
st.sidebar.success(f"Logged in as: {st.session_state['user']}")

if st.sidebar.button("Logout"):
    st.session_state["authenticated"] = False
    st.session_state["user"] = ""
    st.experimental_rerun()


# ----------------------------------------------------------
# MAIN APP
# ----------------------------------------------------------

st.title("🩺 Multi-Disease Diagnostic System (B.Tech Project)")
st.markdown("---")


# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("mtech_disease_dataset_5000.csv")

df = load_data()

X = df.drop("Disease", axis=1)
y = df["Disease"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)


# ----------------------------------------------------------
# INITIALIZE MODELS
# ----------------------------------------------------------

models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=200),
    "SVM": SVC(kernel='linear', probability=True)
}

for model in models.values():
    model.fit(X_train, y_train)


# ----------------------------------------------------------
# SYMPTOM INPUT SECTION
# ----------------------------------------------------------

st.header("📝 Enter Patient Symptoms")

symptom_list = sorted(list(X.columns))

col1, col2 = st.columns(2)

with col1:
    symptom1 = st.selectbox("Symptom 1", [""] + symptom_list)
    symptom2 = st.selectbox("Symptom 2", [""] + symptom_list)
    symptom3 = st.selectbox("Symptom 3", [""] + symptom_list)

with col2:
    symptom4 = st.selectbox("Symptom 4", [""] + symptom_list)
    symptom5 = st.selectbox("Symptom 5", [""] + symptom_list)


# ----------------------------------------------------------
# PREDICTION
# ----------------------------------------------------------

if st.button("🔍 Predict Disease"):

    selected_symptoms = [
        s for s in [symptom1, symptom2, symptom3, symptom4, symptom5] if s != ""
    ]

    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom.")
        st.stop()

    input_vector = [0] * len(X.columns)

    for symptom in selected_symptoms:
        index = list(X.columns).index(symptom)
        input_vector[index] = 1

    input_array = np.array([input_vector])

    st.subheader("🧾 Prediction Results")

    for name, model in models.items():
        pred = model.predict(input_array)
        disease_name = le.inverse_transform(pred)
        st.success(f"{name} predicts: {disease_name[0]}")


    # ------------------------------------------------------
    # MODEL PERFORMANCE
    # ------------------------------------------------------

    st.markdown("---")
    st.subheader("📊 Model Performance Evaluation")

    metrics_data = []
    accuracy_results = {}

    for name, model in models.items():

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cv = cross_val_score(model, X_train, y_train, cv=5).mean()
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        accuracy_results[name] = acc

        metrics_data.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Cross Validation": round(cv, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4)
        })

    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True)


    # Accuracy Graph
    fig1, ax1 = plt.subplots()
    ax1.bar(accuracy_results.keys(), accuracy_results.values())
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Model Accuracy Comparison")
    st.pyplot(fig1)


    # Confusion Matrix
    st.subheader("📌 Confusion Matrix (Random Forest)")

    best_model = models["Random Forest"]
    y_pred_best = best_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred_best)

    fig2, ax2 = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, cmap="Blues", ax=ax2)
    ax2.set_title("Confusion Matrix")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    st.pyplot(fig2)


    # ROC Curve
    st.subheader("📈 Multi-Class ROC Curve (Random Forest)")

    y_test_bin = label_binarize(y_test, classes=np.unique(y_encoded))
    y_score = best_model.predict_proba(X_test)

    fig3, ax3 = plt.subplots()

    for i in range(y_test_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax3.plot(fpr, tpr, label=f"Class {i} (AUC={roc_auc:.2f})")

    ax3.plot([0,1],[0,1],'--')
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.set_title("ROC Curve")
    ax3.legend(fontsize=6)

    st.pyplot(fig3)
