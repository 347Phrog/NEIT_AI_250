#!/usr/bin/env python3
"""
AI 250 - Disease Prediction Model Training and Visualization

This script demonstrates:
1. Loading and exploring health data
2. Training multiple ML models (Decision Tree, Random Forest, Gradient Boosting)
3. Comparing model accuracy
4. Visualizing confusion matrices
5. Analyzing feature importance (which symptoms matter most)

In-class demonstration of predictive analytics in healthcare.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib
#matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Load data and show basic statistics."""
    print("=" * 70)
    print("STEP 1: LOADING AND EXPLORING THE DATA")
    print("=" * 70)
    
    # Load datasets
    train_data = pd.read_csv("training_data.csv")
    test_data = pd.read_csv("test_data.csv")
    
    # Clean up any unnamed columns from trailing commas
    train_data = train_data.loc[:, ~train_data.columns.str.contains('^Unnamed')]
    test_data = test_data.loc[:, ~test_data.columns.str.contains('^Unnamed')]
    
    print(f"\nTraining data: {train_data.shape[0]} samples, {train_data.shape[1]} columns")
    print(f"Test data: {test_data.shape[0]} samples, {test_data.shape[1]} columns")
    
    # Extract features and labels
    X_train = train_data.drop('prognosis', axis=1)
    y_train = train_data['prognosis']
    X_test = test_data.drop('prognosis', axis=1)
    y_test = test_data['prognosis']
    
    print(f"\nNumber of symptoms (features): {len(X_train.columns)}")
    print(f"Number of diseases (classes): {len(y_train.unique())}")
    
    # Show some sample symptoms
    print(f"\nSample symptoms:")
    symptoms = list(X_train.columns)
    for i, symptom in enumerate(symptoms[:10]):
        print(f"   {i+1}. {symptom}")
    print(f"   ... and {len(symptoms) - 10} more")
    
    # Show disease distribution
    print(f"\nDisease distribution (top 10):")
    disease_counts = y_train.value_counts().head(10)
    for disease, count in disease_counts.items():
        print(f"   • {disease}: {count} samples")
    
    return X_train, X_test, y_train, y_test, symptoms


def train_and_compare_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare their performance."""
    print("\n" + "=" * 70)
    print("STEP 2: TRAINING AND COMPARING MODELS")
    print("=" * 70)
    
    # Initialize models
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "predictions": y_pred,
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }
        
        print(f"   {name} Accuracy: {accuracy * 100:.2f}%")
    
    # Show comparison
    print("\n" + "-" * 70)
    print("MODEL COMPARISON SUMMARY:")
    print("-" * 70)
    
    for name, result in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        bar_length = int(result['accuracy'] * 50)
        bar = "=" * bar_length + "-" * (50 - bar_length)
        print(f"{name:20} [{bar}] {result['accuracy']*100:.2f}%")
    
    return results


def visualize_model_comparison(results):
    """Create bar chart comparing model accuracies."""
    print("\n" + "=" * 70)
    print("STEP 3: VISUALIZING MODEL COMPARISON")
    print("=" * 70)
    
    names = list(results.keys())
    accuracies = [results[name]['accuracy'] * 100 for name in names]
    
    plt.figure(figsize=(10, 6))
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars = plt.bar(names, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.title('Model Accuracy Comparison\nDisease Prediction from Symptoms', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150)
    print("   Saved: model_comparison.png")
    # plt.show()  # Comment out for headless - uncomment in class


def visualize_confusion_matrix(results, y_test):
    """Show confusion matrix for the best model."""
    print("\n" + "=" * 70)
    print("STEP 4: CONFUSION MATRIX (Best Model)")
    print("=" * 70)
    
    # Get best model
    best_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_result = results[best_name]
    
    print(f"\n   Showing confusion matrix for: {best_name}")
    print(f"   (Too many classes to show full matrix - showing summary)")
    
    # For 41 classes, full confusion matrix is hard to read
    # Instead, show top misclassifications
    y_pred = best_result['predictions']
    
    # Create a simplified confusion analysis
    correct = sum(y_test == y_pred)
    incorrect = sum(y_test != y_pred)
    
    print(f"\n   Correct predictions: {correct}")
    print(f"   Incorrect predictions: {incorrect}")
    print(f"   Accuracy: {correct / len(y_test) * 100:.2f}%")
    
    # Show which diseases are most often confused
    if incorrect > 0:
        misclassified = pd.DataFrame({
            'Actual': y_test[y_test != y_pred],
            'Predicted': y_pred[y_test != y_pred]
        })
        print(f"\n   Sample misclassifications:")
        for i, row in misclassified.head(5).iterrows():
            print(f"      Actual: {row['Actual'][:30]:30} → Predicted: {row['Predicted'][:30]}")


def visualize_feature_importance(results, symptoms):
    """Show which symptoms are most important for prediction."""
    print("\n" + "=" * 70)
    print("STEP 5: FEATURE IMPORTANCE (Which Symptoms Matter Most)")
    print("=" * 70)
    
    # Get Random Forest model (good for feature importance)
    rf_model = results["Random Forest"]["model"]
    importances = rf_model.feature_importances_
    
    # Get top 20 most important symptoms
    indices = np.argsort(importances)[-20:][::-1]
    top_symptoms = [symptoms[i] for i in indices]
    top_importances = [importances[i] for i in indices]
    
    print("\n   Top 20 Most Predictive Symptoms:")
    print("   " + "-" * 50)
    for i, (symptom, importance) in enumerate(zip(top_symptoms[:10], top_importances[:10]), 1):
        bar = "█" * int(importance * 100)
        print(f"   {i:2d}. {symptom:35} {importance:.4f}")
    
    # Visualize
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_symptoms)))
    bars = plt.barh(range(len(top_symptoms)), top_importances, color=colors)
    plt.yticks(range(len(top_symptoms)), [s.replace('_', ' ').title() for s in top_symptoms])
    plt.xlabel('Importance Score', fontsize=12)
    plt.title('Top 20 Most Predictive Symptoms\n(Random Forest Feature Importance)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    print("\n   Saved: feature_importance.png")
    # plt.show()  # Comment out for headless - uncomment in class


def interactive_prediction(results, symptoms):
    """Allow interactive symptom-based prediction."""
    print("\n" + "=" * 70)
    print("🩺 BONUS: INTERACTIVE PREDICTION")
    print("=" * 70)
    
    rf_model = results["Random Forest"]["model"]
    
    # Demo with sample symptoms
    demo_symptoms = ['high_fever', 'headache', 'fatigue', 'sweating', 'chills']
    
    print(f"\n   Demo: Predicting disease for symptoms:")
    for s in demo_symptoms:
        print(f"      • {s}")
    
    # Create symptom vector
    symptom_vector = np.zeros(len(symptoms))
    for s in demo_symptoms:
        if s in symptoms:
            symptom_vector[symptoms.index(s)] = 1
    
    # Predict
    probs = rf_model.predict_proba(symptom_vector.reshape(1, -1))[0]
    top_indices = np.argsort(probs)[-5:][::-1]
    
    print(f"\n   Top 5 Predicted Conditions:")
    print("   " + "-" * 50)
    for i, idx in enumerate(top_indices, 1):
        disease = rf_model.classes_[idx]
        prob = probs[idx] * 100
        bar = "█" * int(prob / 2)
        print(f"   {i}. {disease:35} {prob:5.1f}%")


def main():
    """Run the complete demonstration."""
    print("\n" + "=" * 70)
    print("AI 250 - DISEASE PREDICTION DEMO")
    print("   Predictive Analytics in Healthcare")
    print("=" * 70)
    
    # Step 1: Load and explore data
    X_train, X_test, y_train, y_test, symptoms = load_and_explore_data()
    
    input("\n[Press Enter to continue to model training...]\n")
    
    # Step 2: Train and compare models
    results = train_and_compare_models(X_train, X_test, y_train, y_test)
    
    input("\n[Press Enter to see visualizations...]\n")
    
    # Step 3: Visualize model comparison
    visualize_model_comparison(results)
    
    # Step 4: Confusion matrix
    visualize_confusion_matrix(results, y_test)
    
    input("\n[Press Enter to see feature importance...]\n")
    
    # Step 5: Feature importance
    visualize_feature_importance(results, symptoms)
    
    # Bonus: Interactive prediction demo
    interactive_prediction(results, symptoms)
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print("\nFiles saved:")
    print("   • model_comparison.png")
    print("   • feature_importance.png")
    print("\n DISCLAIMER: This is for educational purposes only.")
    print("Always consult healthcare professionals for medical advice.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
