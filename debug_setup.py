import os
import sys
import pandas as pd
import joblib
from src.preprocessing import preprocess_pipeline
from src.train import train_models, evaluate_models, save_best_model

def debug_setup():
    print("=== DEBUGGING SETUP ===")
    
    # 1. Check directories
    print("\n1. Checking directories...")
    os.makedirs('models', exist_ok=True)
    print(f"models/ directory exists: {os.path.exists('models')}")
    
    # 2. Check dataset
    print("\n2. Checking dataset...")
    data_path = 'data/bank-additional-full.csv'
    print(f"Dataset exists: {os.path.exists(data_path)}")
    if os.path.exists(data_path):
        print(f"File size: {os.path.getsize(data_path)} bytes")
    
    # 3. Test preprocessing
    print("\n3. Testing preprocessing...")
    try:
        X_train, X_test, y_train, y_test, features = preprocess_pipeline(data_path)
        print("Preprocessing completed successfully!")
        print(f"X_train shape: {X_train.shape}")
        
        # Save processed data
        joblib.dump((X_train, X_test, y_train, y_test, features), 
                   'models/processed_data.joblib')
        print("Saved processed_data.joblib")
    except Exception as e:
        print(f"Preprocessing failed: {str(e)}")
        return
    
    # 4. Test model training
    print("\n4. Testing model training...")
    try:
        models = train_models(X_train, y_train)
        print(f"Trained {len(models)} models")
    except Exception as e:
        print(f"Training failed: {str(e)}")
        return
    
    # 5. Test evaluation
    print("\n5. Testing evaluation...")
    try:
        metrics_df = evaluate_models(models, X_test, y_test)
        metrics_df.to_csv('models/model_metrics.csv', index=False)
        print("Saved model_metrics.csv")
        print(metrics_df)
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        return
    
    # 6. Test saving best model
    print("\n6. Testing model saving...")
    try:
        save_best_model(models, metrics_df)
        print("Saved best_model.joblib and scaler.joblib")
    except Exception as e:
        print(f"Saving failed: {str(e)}")
        return
    
    print("\n=== ALL CHECKS COMPLETED ===")
    print("You can now run: streamlit run app.py")

if __name__ == "__main__":
    debug_setup()