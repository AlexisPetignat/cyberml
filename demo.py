import streamlit as st
import pandas as pd
import numpy as np
import xgboost
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC, OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, average_precision_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
import shap
import matplotlib.pyplot as plt

# Configuration
DATA_FOLDER = './data/'
ATTACK_FOLDER = 'attack_data/'
BENIGN_FOLDER = 'benign_data/'

st.set_page_config(page_title="CYBERML Demo", layout="wide")

st.title("🛡️ CYBERML - IoT Attack Detection Demo")
st.markdown("""
**Project:** CYBERML 2025-2026  
**Dataset:** CIC IoT-DIAD 2024
""")

# --- Utils ---

@st.cache_data
def load_and_merge_data(duration_sec):
    attack_file = f"attack_samples_{duration_sec}sec.csv"
    benign_file = f"benign_samples_{duration_sec}sec.csv"
    
    try:
        attack_data = pd.read_csv(DATA_FOLDER + ATTACK_FOLDER + attack_file)
        benign_data = pd.read_csv(DATA_FOLDER + BENIGN_FOLDER + benign_file)
    except FileNotFoundError:
        st.error(f"Files for {duration_sec}s not found in {DATA_FOLDER}")
        return None

    combined_data = pd.concat([attack_data, benign_data], ignore_index=True)
    combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)
    return combined_data

@st.cache_data
def preprocess_data(combined_data):
    combined_data.fillna(0, inplace=True)
    encoded_data = combined_data.copy()
    encoded_values = {}

    label_columns = ['label1', 'label2', 'label3', 'label4']
    for col in label_columns:
        if col in encoded_data.columns:
            encoded_data[col], uniques = pd.factorize(encoded_data[col])
            encoded_values[col] = dict(enumerate(uniques))

    encoded_data = encoded_data.select_dtypes(include=[np.number])
    
    if 'label1' not in encoded_data.columns:
        st.error("label1 column missing/dropped")
        return None, None, None, None, None, None, None

    X = encoded_data.drop(['label1', 'label2', 'label3', 'label4'], axis=1, errors='ignore')
    y = encoded_data['label1']
    y_type = encoded_data['label2'] if 'label2' in encoded_data.columns else None
    
    X_train, X_test, y_train, y_test, y_type_train, y_type_test = train_test_split(
        X, y, y_type, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, y_type_train, y_type_test, encoded_values

def plot_shap(model, X_train):
    with st.spinner("Calculating SHAP values..."):
        # Sample if too large
        if len(X_train) > 1000:
            X_sample = X_train.sample(1000, random_state=42)
        else:
            X_sample = X_train

        try:
            explainer = shap.TreeExplainer(model)
            explanation = explainer(X_sample)
            
            fig, ax = plt.subplots()
            shap.plots.beeswarm(explanation, show=False)
            st.pyplot(plt.gcf())
            plt.clf()
        except Exception as e:
            st.warning(f"Could not plot SHAP: {e}")

# --- Sidebar ---
st.sidebar.header("Configuration")
duration = st.sidebar.selectbox("Select Time Window (seconds)", [1, 3, 5, 7], index=2)

# --- Load Data ---
data = load_and_merge_data(duration)

if data is not None:
    X_train, X_test, y_train, y_test, y_type_train, y_type_test, encoded_values = preprocess_data(data)
    
    # Identify mapping for Label 1 (Binary)
    # encoded_values['label1'] returns {0: 'benign', 1: 'attack'} or similar
    lbl1_map = encoded_values['label1']
    # Invert to find code for 'attack'
    code_attack = [k for k, v in lbl1_map.items() if 'attack' in v.lower()]
    code_benign = [k for k, v in lbl1_map.items() if 'benign' in v.lower()]
    
    attack_label_code = code_attack[0] if code_attack else 1
    
    st.sidebar.success(f"Loaded {len(data)} samples for {duration}s window")

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Supervised Learning", "Unsupervised Learning", "Attack Analysis", "Single Sample Prediction"])

    with tab1:
        st.header("Dataset Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Samples", len(data))
            st.metric("Features", X_train.shape[1])
        with col2:
            class_counts = data['label1'].value_counts()
            st.bar_chart(class_counts)
        
        st.subheader("Sample Data")
        st.dataframe(data.head())

    with tab2:
        st.header("Supervised Classification (Binary)")
        model_name = st.selectbox("Choose Model", ["XGBoost", "Logistic Regression", "Linear SVC"])
        
        if st.button("Train & Evaluate", key="btn_supervised"):
            with st.spinner(f"Training {model_name}..."):
                if model_name == "XGBoost":
                    model = xgboost.XGBClassifier(eval_metric='logloss')
                    model.fit(X_train, y_train)
                elif model_name == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_train, y_train)
                else:
                    model = LinearSVC(dual="auto")
                    model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                
                # Metrics
                acc = balanced_accuracy_score(y_test, y_pred)
                mcc = matthews_corrcoef(y_test, y_pred)
                
                c1, c2 = st.columns(2)
                c1.metric("Balanced Accuracy", f"{acc:.4f}")
                c2.metric("MCC", f"{mcc:.4f}")
                
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                st.write(cm)
                
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
                
                if model_name == "XGBoost":
                    st.subheader("Feature Importance (SHAP)")
                    plot_shap(model, X_train)

    with tab3:
        st.header("Unsupervised Anomaly Detection")
        algo = st.selectbox("Algorithm", ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"])
        contamination = st.slider("Contamination (Expected Outlier Fraction)", 0.01, 0.5, 0.39)
        
        if st.button("Run Anomaly Detection", key="btn_unsupervised"):
            with st.spinner(f"Running {algo}..."):
                # Logic to map -1 (anomaly) to Attack Code, 1 (normal) to Benign Code
                # Attack is usually the anomaly
                
                if algo == "Isolation Forest":
                    model = IsolationForest(contamination=contamination, random_state=42)
                    model.fit(X_train)
                    y_pred_raw = model.predict(X_test) # -1 for outliers
                elif algo == "One-Class SVM":
                    # Subsample for SVM speed
                    if len(X_train) > 10000:
                        X_sub = X_train.sample(10000, random_state=42)
                    else:
                        X_sub = X_train
                    model = OneClassSVM(kernel="linear")
                    model.fit(X_sub)
                    y_pred_raw = model.predict(X_test)
                else: # LOF
                    if len(X_train) > 10000:
                        X_sub = X_train.sample(10000, random_state=42)
                    else:
                        X_sub = X_train
                    model = LocalOutlierFactor(novelty=True, contamination=contamination)
                    model.fit(X_sub)
                    y_pred_raw = model.predict(X_test)

                # Map predictions
                # -1 (Anomaly) -> Attack
                # 1 (Normal) -> Benign
                
                # We need to know which code corresponds to 'attack' in y_test
                # code_attack was calculated earlier
                
                # prediction vector init with benign code
                y_pred = np.full(y_pred_raw.shape, code_benign[0] if code_benign else 0)
                # set attack code where -1
                y_pred[y_pred_raw == -1] = attack_label_code
                
                # Metrics
                acc = balanced_accuracy_score(y_test, y_pred)
                mcc = matthews_corrcoef(y_test, y_pred)
                
                c1, c2 = st.columns(2)
                c1.metric("Balanced Accuracy", f"{acc:.4f}")
                c2.metric("MCC", f"{mcc:.4f}")
                
                st.write("Note: Unsupervised methods treat the minority/anomalous class as the target.")

    with tab4:
        st.header("Multiclass Analysis (Attack Types)")
        if y_type_train is not None:
             if st.button("Train Multiclass XGBoost", key="btn_multi"):
                 with st.spinner("Training Multiclass XGBoost..."):
                    # Map codes back to names for display
                    lbl2_map = encoded_values['label2']
                    names = [lbl2_map[i] for i in sorted(lbl2_map.keys())]
                    
                    model_multi = xgboost.XGBClassifier(objective="multi:softmax", num_class=len(names))
                    model_multi.fit(X_train, y_type_train)
                    y_pred_multi = model_multi.predict(X_test)
                    
                    st.subheader("Classification Report")
                    report = classification_report(y_type_test, y_pred_multi, target_names=names, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose())
                    
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_type_test, y_pred_multi)
                    st.dataframe(pd.DataFrame(cm, index=names, columns=names))
                    
                    st.subheader("Feature Importance (Multiclass)")
                    plot_shap(model_multi, X_train)
        else:
            st.warning("Label2 (Attack Type) not found in dataset.")

    with tab5:
        st.header("Single Sample Prediction")
        
        if len(X_test) > 0:
            # Select sample
            sample_idx = st.number_input("Select Sample Index from Test Set", min_value=0, max_value=len(X_test)-1, value=0, step=1)
            
            # Get sample data
            sample_data = X_test.iloc[[sample_idx]]
            try:
                # y_test might be a Series or numpy array, handle safely
                true_label = y_test.iloc[sample_idx] if hasattr(y_test, 'iloc') else y_test[sample_idx]
                true_label_str = encoded_values['label1'][true_label]
            except:
                true_label_str = "Unknown"
                true_label = -1
            
            st.write(f"**True Label:** {true_label_str} (Code: {true_label})")
            
            with st.expander("View Features"):
                st.dataframe(sample_data)
            
            st.subheader("Prediction")
            model_choice_single = st.selectbox("Select Model for Prediction", ["XGBoost", "Logistic Regression", "Linear SVC"], key="single_model_select")
            
            if st.button("Predict Single Sample"):
                with st.spinner("Processing..."):
                     # Simple caching using session state
                     if "trained_models" not in st.session_state:
                         st.session_state["trained_models"] = {}
                     
                     # Key depends on duration and model type
                     key = f"{model_choice_single}_{duration}"
                     
                     if key not in st.session_state["trained_models"]:
                         st.info(f"Training {model_choice_single} model first...")
                         if model_choice_single == "XGBoost":
                            model = xgboost.XGBClassifier(eval_metric='logloss')
                         elif model_choice_single == "Logistic Regression":
                            model = LogisticRegression(max_iter=1000)
                         else:
                            model = LinearSVC(dual="auto")
                         
                         model.fit(X_train, y_train)
                         st.session_state["trained_models"][key] = model
                     
                     model = st.session_state["trained_models"][key]
                     
                     prediction = model.predict(sample_data)[0]
                     pred_str = encoded_values['label1'][prediction]
                     
                     c1, c2 = st.columns(2)
                     is_correct = (prediction == true_label)
                     c1.metric("Predicted Label", f"{pred_str}", 
                               delta="Correct" if is_correct else "Incorrect", 
                               delta_color="normal" if is_correct else "inverse")
                     
                     if hasattr(model, "predict_proba"):
                         proba = model.predict_proba(sample_data)
                         # st.write("Probabilities:", proba)
                         confidence = np.max(proba)
                         c2.metric("Confidence", f"{confidence:.2%}")
                     else:
                        c2.metric("Confidence", "N/A (SVM/Linear)")
        else:
            st.warning("Test set is empty.")

else:
    st.warning("Please ensure data is unpacked in `./data/` folder.")
