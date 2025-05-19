import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def load_processed_data():
    """Load processed data from file"""
    X_train, X_test, y_train, y_test, feature_names = joblib.load('models/processed_data.joblib')
    return X_train, X_test, y_train, y_test, feature_names

def train_models(X_train, y_train):
    """Train multiple classification models"""
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'LightGBM': LGBMClassifier(random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Evaluate models and return metrics"""
    metrics = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_prob)
        })
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'models/{name.lower().replace(" ", "_")}_cm.png')
        plt.close()
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_prob):.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(f'models/{name.lower().replace(" ", "_")}_roc.png')
        plt.close()
    
    metrics_df = pd.DataFrame(metrics)
    return metrics_df

def feature_importance_analysis(model, feature_names, model_name):
    """
    Analyze and plot feature importance and save the plot
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        model_name: Name of the model for plot title
    
    Returns:
        str: Path to saved plot or None if not applicable
    """
    try:
        # Check if model supports feature importance
        if not hasattr(model, 'feature_importances_'):
            print(f"Model {model_name} doesn't support feature importance analysis")
            return None

        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Create figure with improved styling
        plt.figure(figsize=(14, 10))
        ax = plt.gca()
        
        # Create bars with color gradient
        bars = ax.bar(range(20), importances[indices][:20], 
                     align='center',
                     color=plt.cm.viridis(np.linspace(0.2, 1, 20)))
        
        # Customize ticks and labels
        ax.set_xticks(range(20))
        ax.set_xticklabels([feature_names[i] for i in indices[:20]], 
                          rotation=45, ha='right', fontsize=10)
        ax.tick_params(axis='y', labelsize=10)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)
        
        # Add grid and styling
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_title(f'Top 20 Feature Importances - {model_name}', 
                    fontsize=14, pad=20)
        ax.set_ylabel('Importance Score', fontsize=12)
        ax.set_xlabel('Features', fontsize=12)
        
        plt.tight_layout()
        
        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)
        
        # Save plot in multiple formats
        plot_base = f'models/{model_name.lower().replace(" ", "_")}_feature_importances'
        plot_path_png = f'{plot_base}.png'
        plot_path_svg = f'{plot_base}.svg'
        
        plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(plot_path_svg, format='svg', bbox_inches='tight')
        plt.close()
        
        print(f"Saved feature importance plots to {plot_base}.*")
        return plot_path_png
        
    except Exception as e:
        print(f"Error generating feature importance plot: {str(e)}")
        return None

def save_best_model(models, metrics_df):
    """Save the best performing model based on F1 score"""
    best_model_name = metrics_df.loc[metrics_df['F1 Score'].idxmax(), 'Model']
    best_model = models[best_model_name]
    
    joblib.dump(best_model, 'models/best_model.joblib')
    print(f"Best model saved: {best_model_name}")
    
    return best_model_name

if __name__ == "__main__":
    # Load processed data
    X_train, X_test, y_train, y_test, feature_names = load_processed_data()
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Evaluate models
    metrics_df = evaluate_models(models, X_test, y_test)
    print("\nModel Performance Metrics:")
    print(metrics_df.to_string(index=False))
    
    # Save metrics
    metrics_df.to_csv('models/model_metrics.csv', index=False)
    
    # Feature importance analysis for tree-based models
    for name, model in models.items():
        if name in ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM']:
            feature_imp = feature_importance_analysis(model, feature_names)
            if feature_imp is not None:
                feature_imp.to_csv(f'models/{name.lower().replace(" ", "_")}_feature_importance.csv', index=False)
    
    # Save best model
    best_model_name = save_best_model(models, metrics_df)