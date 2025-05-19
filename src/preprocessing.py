import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

def load_data(filepath):
    """Load the dataset from CSV"""
    df = pd.read_csv(filepath, sep=';')
    print(f"Original data shape: {df.shape}")
    return df

def clean_data(df):
    """Handle missing values and data cleaning"""
    
    # Replace 'unknown' with NaN for proper handling
    categorical_cols = ['job', 'marital', 'education', 'contact', 'poutcome']
    for col in categorical_cols:
        df[col] = df[col].replace('unknown', np.nan)
    
    # For default column, replace 'unknown' with 'no' (assuming if unknown, likely no default)
    df['default'] = df['default'].replace('unknown', 'no')
    
    # Drop rows with NaN in important categorical features
    df = df.dropna(subset=['job', 'marital', 'education'])
    
    # For poutcome, create a new category 'not_contacted' for clients not previously contacted
    df['poutcome'] = df['poutcome'].fillna('not_contacted')
    
    print(f"Data shape after cleaning: {df.shape}")
    return df

def feature_engineering(df):
    """Create new features and transform existing ones"""
    
    # Create age groups
    bins = [0, 25, 45, 65, 100]
    labels = ['young', 'adult', 'middle-aged', 'senior']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
    
    # Create balance categories
    df['balance_category'] = pd.cut(df['balance'], 
                                   bins=[-float('inf'), 0, 1000, 5000, float('inf')],
                                   labels=['negative', 'low', 'medium', 'high'])
    
    # Create contact duration in minutes
    df['duration_min'] = df['duration'] / 60
    
    # Create interaction terms
    df['has_loan'] = np.where((df['housing'] == 'yes') | (df['loan'] == 'yes'), 1, 0)
    
    # Create campaign success rate (previous success / previous contacts)
    df['prev_success_rate'] = np.where(
        df['previous'] > 0,
        np.where(df['poutcome'] == 'success', 1, 0) / df['previous'],
        0
    )
    
    return df

def encode_features(df):
    """Encode categorical variables"""
    
    # Binary encoding for yes/no features
    binary_cols = ['default', 'housing', 'loan', 'y']
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])
    
    # One-hot encoding for other categorical features
    categorical_cols = ['job', 'marital', 'education', 'contact', 
                       'month', 'poutcome', 'age_group', 'balance_category']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df

def scale_features(df):
    """Scale numerical features"""
    numerical_cols = ['age', 'balance', 'duration', 'campaign', 
                     'pdays', 'previous', 'duration_min']
    
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Save the scaler for later use
    joblib.dump(scaler, 'models/scaler.joblib')
    
    return df

def handle_imbalance(X, y):
    """Handle class imbalance using SMOTE"""
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

def preprocess_pipeline(filepath):
    """Complete preprocessing pipeline"""
    df = load_data(filepath)
    df = clean_data(df)
    df = feature_engineering(df)
    df = encode_features(df)
    df = scale_features(df)
    
    # Separate features and target
    X = df.drop('y', axis=1)
    y = df['y']
    
    # Handle class imbalance
    X_res, y_res = handle_imbalance(X, y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, X.columns

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names = preprocess_pipeline('../data/bank-additional-full.csv')
    # Save processed data
    joblib.dump((X_train, X_test, y_train, y_test, feature_names), 'models/processed_data.joblib')