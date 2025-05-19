import os
import sys
import pandas as pd  # Add this import
import joblib  # Add this import
from src.preprocessing import preprocess_pipeline
from src.train import train_models, evaluate_models, save_best_model

def setup_project():
    try:
        print("Creating directories...")
        os.makedirs('models', exist_ok=True)
        
        print("Checking for dataset...")
        if not os.path.exists('data/bank-additional-full.csv'):
            raise FileNotFoundError("Dataset not found at data/bank-additional-full.csv")
        
        print("Running preprocessing...")
        X_train, X_test, y_train, y_test, feature_names = preprocess_pipeline('data/bank-additional-full.csv')
        
        print("Training models...")
        models = train_models(X_train, y_train)
        
        print("Evaluating models...")
        metrics_df = evaluate_models(models, X_test, y_test)
        
        print("Saving best model...")
        save_best_model(models, metrics_df)
        
        print("\n✅ Setup completed successfully!")
        print("You can now run: streamlit run app.py")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    setup_project()