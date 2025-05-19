import os
import subprocess
from src.preprocessing import preprocess_pipeline
from src.train import train_models, evaluate_models, save_best_model

def main():
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('notebooks', exist_ok=True)
    
    print("Starting the Term Deposit Prediction Pipeline...")
    
    # Step 1: Data Preprocessing
    print("\nStep 1: Data Preprocessing")
    X_train, X_test, y_train, y_test, feature_names = preprocess_pipeline('data/bank-additional-full.csv')
    
    # Step 2: Model Training
    print("\nStep 2: Model Training")
    models = train_models(X_train, y_train)
    
    # Step 3: Model Evaluation
    print("\nStep 3: Model Evaluation")
    metrics_df = evaluate_models(models, X_test, y_test)
    print("\nModel Performance Metrics:")
    print(metrics_df)
    
    # Step 4: Save Best Model
    print("\nStep 4: Saving Best Model")
    best_model_name = save_best_model(models, metrics_df)
    print(f"\nBest model is: {best_model_name}")
    
    # Step 5: Launch Streamlit App
    print("\nStep 5: Launching Streamlit App")
    print("To run the app, execute: streamlit run app.py")
    
    # Optionally launch the app automatically
    # subprocess.run(["streamlit", "run", "app.py"])

if __name__ == "__main__":
    main()