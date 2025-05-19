import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image

# At the TOP of app.py (after imports but before Streamlit code)
def feature_importance_analysis(model, feature_names, model_name):
    """Generate and save feature importance plot"""
    try:
        if not hasattr(model, 'feature_importances_'):
            return None
            
        plt.figure(figsize=(12, 8))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]  # Top 20 features
        
        plt.title(f"Feature Importances - {model_name}")
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        
        os.makedirs('models', exist_ok=True)
        plot_path = f'models/{model_name.lower().replace(" ", "_")}_feature_importances.png'
        plt.savefig(plot_path)
        plt.close()
        return plot_path
        
    except Exception as e:
        print(f"Error generating feature importance: {str(e)}")
        return None
    

# At the top of app.py, add this function
def check_required_files():
    required_files = [
        'models/processed_data.joblib',
        'models/best_model.joblib',
        'models/scaler.joblib'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        st.error(f"Missing required files: {', '.join(missing_files)}")
        st.write("Please run the setup script first:")
        st.code("python setup.py")
        return False
    return True

# Modify the load_data_and_models function
@st.cache_resource
def load_data_and_models():
    """Load all required data and models with error handling"""
    try:
        # Load processed data
        if not os.path.exists('models/processed_data.joblib'):
            raise FileNotFoundError("Processed data not found. Run setup.py first")
            
        X_train, X_test, y_train, y_test, feature_names = joblib.load('models/processed_data.joblib')
        
        # Load best model
        if not os.path.exists('models/best_model.joblib'):
            raise FileNotFoundError("Model not found. Run setup.py first")
            
        model = joblib.load('models/best_model.joblib')
        
        # Load metrics if exists
        metrics_df = None
        if os.path.exists('models/model_metrics.csv'):
            metrics_df = pd.read_csv('models/model_metrics.csv')
        else:
            st.warning("Model metrics file not found")
        
        # Load original data for EDA
        if not os.path.exists('data/bank-additional-full.csv'):
            raise FileNotFoundError("Original data not found in data/ folder")
            
        original_data = pd.read_csv('data/bank-additional-full.csv', sep=';')
        
        return {
            'model': model,
            'metrics': metrics_df,
            'original_data': original_data,
            'feature_names': feature_names,
            'X_train': X_train,
            'y_train': y_train
        }
        
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        st.stop()

# At the beginning of your Streamlit app, add this check
if not check_required_files():
    st.stop()


# Set page config
st.set_page_config(
    page_title="Term Deposit Subscription Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data and models
@st.cache_resource
def load_data_and_models():
    # Load processed data
    X_train, X_test, y_train, y_test, feature_names = joblib.load('models/processed_data.joblib')
    
    # Load best model
    model = joblib.load('models/best_model.joblib')
    
    # Load metrics
    metrics_df = pd.read_csv('models/model_metrics.csv')
    
    # Load original data for EDA
    original_data = pd.read_csv('data/bank-additional-full.csv', sep=';')
    
    return {
        'model': model,
        'metrics': metrics_df,
        'original_data': original_data,
        'feature_names': feature_names,
        'X_train': X_train,
        'y_train': y_train
    }

data_models = load_data_and_models()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA", "Prediction", "Model Info"])

# Home Page
if page == "Home":
    st.title("Term Deposit Subscription Predictor")
    st.image("https://miro.medium.com/v2/resize:fit:1400/1*_NVBTVdmjt3Qvq3CZOy-SQ.jpeg", width=700)
    
    st.markdown("""
    ## 🏦 Banking Campaign Optimization
        
    This application helps predict whether a bank client will subscribe to a term deposit based on their 
    demographic and campaign information. The model can be used to optimize marketing campaigns by 
    targeting clients who are more likely to subscribe.
    
    ### 📊 Problem Statement
    A Portuguese banking institution conducted multiple direct marketing campaigns via phone calls to 
    promote term deposit products. The challenge is to predict which clients are likely to subscribe 
    to help optimize marketing efforts.
    
    ### 🎯 Business Objectives
    - Improve campaign efficiency by targeting high-potential clients
    - Reduce operational costs by minimizing unnecessary calls
    - Enhance customer experience with relevant offers
    - Gain insights into factors influencing subscription
    
    ### 🔍 Dataset Information
    The dataset contains information about:
    - Client demographics (age, job, education, etc.)
    - Financial information (balance, loans, defaults)
    - Campaign details (contact type, duration, previous contacts)
    - Outcome (whether the client subscribed to a term deposit)
    """)

# EDA Page
elif page == "EDA":
    st.title("Exploratory Data Analysis")
    st.write("Explore the dataset and insights about term deposit subscriptions")
    
    # Original data
    if st.checkbox("Show raw data"):
        st.write(data_models['original_data'].head())
    
    # Target distribution
    st.subheader("Target Variable Distribution")
    fig = px.pie(data_models['original_data'], names='y', title='Subscription Rate')
    st.plotly_chart(fig, use_container_width=True)
    
    # Age distribution
    st.subheader("Age Distribution by Subscription")
    fig = px.histogram(data_models['original_data'], x='age', color='y', 
                       nbins=30, barmode='overlay',
                       title='Age Distribution by Subscription Status')
    st.plotly_chart(fig, use_container_width=True)
    
    # Job distribution
    st.subheader("Subscription Rate by Job Type")
    job_df = data_models['original_data'].groupby(['job', 'y']).size().unstack().fillna(0)
    job_df['subscription_rate'] = job_df['yes'] / (job_df['yes'] + job_df['no']) * 100
    job_df = job_df.sort_values('subscription_rate', ascending=False)
    fig = px.bar(job_df, x=job_df.index, y='subscription_rate', 
                 title='Subscription Rate by Job Type (%)')
    st.plotly_chart(fig, use_container_width=True)
    
    # Duration analysis
    st.subheader("Call Duration vs Subscription")
    fig = px.box(data_models['original_data'], x='y', y='duration', 
                 title='Call Duration Distribution by Subscription Status')
    st.plotly_chart(fig, use_container_width=True)
    
    # Previous campaign outcome
    st.subheader("Previous Campaign Outcome Impact")
    poutcome_df = data_models['original_data'].groupby(['poutcome', 'y']).size().unstack().fillna(0)
    poutcome_df['subscription_rate'] = poutcome_df['yes'] / (poutcome_df['yes'] + poutcome_df['no']) * 100
    fig = px.bar(poutcome_df, x=poutcome_df.index, y='subscription_rate',
                 title='Subscription Rate by Previous Campaign Outcome (%)')
    st.plotly_chart(fig, use_container_width=True)

# Prediction Page
elif page == "Prediction":
    st.title("Term Deposit Subscription Prediction")
    st.write("Fill in the client details to predict the likelihood of subscription")
    
    # Create form for input
    with st.form("client_details"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 18, 100, 30)
            job = st.selectbox("Job Type", [
                "admin.", "blue-collar", "entrepreneur", "housemaid",
                "management", "retired", "self-employed", "services",
                "student", "technician", "unemployed"
            ])
            marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
            education = st.selectbox("Education Level", ["primary", "secondary", "tertiary"])
            default = st.radio("Has Credit in Default?", ["no", "yes"])
            balance = st.number_input("Average Yearly Balance (€)", -10000, 100000, 0)
        
        with col2:
            housing = st.radio("Has Housing Loan?", ["no", "yes"])
            loan = st.radio("Has Personal Loan?", ["no", "yes"])
            contact = st.selectbox("Contact Type", ["cellular", "telephone"])
            month = st.selectbox("Last Contact Month", [
                "jan", "feb", "mar", "apr", "may", "jun",
                "jul", "aug", "sep", "oct", "nov", "dec"
            ])
            day = st.slider("Last Contact Day of Month", 1, 31, 15)
            duration = st.number_input("Last Contact Duration (seconds)", 0, 5000, 180)
        
        campaign = st.slider("Number of Contacts This Campaign", 1, 50, 1)
        pdays = st.number_input("Days Since Last Contact (-1 if not contacted)", -1, 1000, -1)
        previous = st.slider("Number of Previous Contacts", 0, 50, 0)
        poutcome = st.selectbox("Previous Campaign Outcome", ["failure", "other", "success", "unknown"])
        
        submitted = st.form_submit_button("Predict")
    
    if submitted:
        # Create input dataframe
        input_data = {
            'age': age,
            'job': job,
            'marital': marital,
            'education': education,
            'default': default,
            'balance': balance,
            'housing': housing,
            'loan': loan,
            'contact': contact,
            'day': day,
            'month': month,
            'duration': duration,
            'campaign': campaign,
            'pdays': pdays,
            'previous': previous,
            'poutcome': poutcome
        }
        
        # Convert to dataframe
        input_df = pd.DataFrame([input_data])
        
        # Preprocess the input data (similar to training preprocessing)
        # Age groups
        bins = [0, 25, 45, 65, 100]
        labels = ['young', 'adult', 'middle-aged', 'senior']
        input_df['age_group'] = pd.cut(input_df['age'], bins=bins, labels=labels)
        
        # Balance categories
        input_df['balance_category'] = pd.cut(input_df['balance'], 
                                           bins=[-float('inf'), 0, 1000, 5000, float('inf')],
                                           labels=['negative', 'low', 'medium', 'high'])
        
        # Create duration in minutes
        input_df['duration_min'] = input_df['duration'] / 60
        
        # Create interaction terms
        input_df['has_loan'] = np.where((input_df['housing'] == 'yes') | (input_df['loan'] == 'yes'), 1, 0)
        
        # Create campaign success rate
        input_df['prev_success_rate'] = np.where(
            input_df['previous'] > 0,
            np.where(input_df['poutcome'] == 'success', 1, 0) / input_df['previous'],
            0
        )
        
        # Binary encoding
        binary_cols = ['default', 'housing', 'loan']
        for col in binary_cols:
            input_df[col] = input_df[col].map({'yes': 1, 'no': 0})
        
        # One-hot encoding
        categorical_cols = ['job', 'marital', 'education', 'contact', 
                           'month', 'poutcome', 'age_group', 'balance_category']
        input_df = pd.get_dummies(input_df, columns=categorical_cols)
        
        # Ensure all training columns are present
        # Add missing columns with 0
        for col in data_models['feature_names']:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns to match training data
        input_df = input_df[data_models['feature_names']]
        
        # Scale numerical features
        scaler = joblib.load('models/scaler.joblib')
        numerical_cols = ['age', 'balance', 'duration', 'campaign', 
                         'pdays', 'previous', 'duration_min']
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
        
        # Make prediction
        prediction = data_models['model'].predict(input_df)
        prediction_proba = data_models['model'].predict_proba(input_df)
        
        # Display results
        st.subheader("Prediction Results")
        
        if prediction[0] == 1:
            st.success(f"✅ The client is likely to subscribe to a term deposit (probability: {prediction_proba[0][1]:.2%})")
        else:
            st.error(f"❌ The client is not likely to subscribe to a term deposit (probability: {prediction_proba[0][1]:.2%})")
        
        # Show probability breakdown
        st.write("Prediction Probability Breakdown:")
        prob_df = pd.DataFrame({
            'Class': ['No', 'Yes'],
            'Probability': [prediction_proba[0][0], prediction_proba[0][1]]
        })
        st.bar_chart(prob_df.set_index('Class'))
        
        # Show feature importance (if available)
        if hasattr(data_models['model'], 'feature_importances_'):
            st.subheader("Top Influencing Factors")
            feature_imp = pd.DataFrame({
                'Feature': data_models['feature_names'],
                'Importance': data_models['model'].feature_importances_
            })
            top_features = feature_imp.sort_values('Importance', ascending=False).head(10)
            
            fig = px.bar(top_features, x='Importance', y='Feature', orientation='h',
                         title='Top 10 Most Important Features')
            st.plotly_chart(fig, use_container_width=True)

# Model Info Page
elif page == "Model Info":
    st.title("Model Information")
    
    # Load model info
    try:
        model_name = data_models['metrics'].loc[data_models['metrics']['F1 Score'].idxmax(), 'Model']
        model = data_models['model']
        feature_names = data_models['feature_names']
        
        st.subheader("Feature Importance")
        
        # Try to load existing plot
        plot_path = f'models/{model_name.lower().replace(" ", "_")}_feature_importances.png'
        
        if os.path.exists(plot_path):
            st.image(plot_path, caption=f'{model_name} Feature Importances')
        else:
            st.warning("Feature importance visualization not available")
            
            if st.button("Generate Feature Importance Plot"):
                with st.spinner("Generating visualization..."):
                    generated_path = feature_importance_analysis(model, feature_names, model_name)
                    if generated_path:
                        st.success("Generated successfully!")
                        st.experimental_rerun()
                    else:
                        st.error("This model type doesn't support feature importance analysis")
    
    except Exception as e:
        st.error(f"Error loading model information: {str(e)}")