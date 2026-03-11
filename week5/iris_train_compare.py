from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

DATA_COLUMNS = [
    'sepal_length_cm',
    'sepal_width_cm',
    'petal_length_cm',
    'petal_width_cm',
    'species',
]

BASE_DIR = Path(__file__).resolve().parent
IRIS_PATH = BASE_DIR / 'iris_info' / 'bezdekIris.data'


def load_and_explore_data(path: Path = IRIS_PATH):
    iris = pd.read_csv(path, header=None, names=DATA_COLUMNS).dropna()

    print('=' * 70)
    print('STEP 1: LOADING AND EXPLORING IRIS DATA')
    print('=' * 70)
    print(f'Loaded file: {path}')
    print(f'Total samples: {len(iris)}')
    print(f'Features: {len(DATA_COLUMNS) - 1}')
    print(f'Classes: {iris["species"].nunique()}')

    X = iris.drop('species', axis=1)
    y = iris['species']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return iris, X_train, X_test, y_train, y_test


def train_and_compare_models(X_train, X_test, y_train, y_test):
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    print('\n' + '=' * 70)
    print('STEP 2: TRAINING AND COMPARING MODELS')
    print('=' * 70)

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': predictions,
            'confusion_matrix': confusion_matrix(y_test, predictions, labels=sorted(y_test.unique())),
        }
        print(f'{name:20} {accuracy * 100:6.2f}%')

    return results


def visualize_model_comparison(results, output_path: Path = BASE_DIR / 'iris_model_comparison.png'):
    names = list(results.keys())
    accuracies = [results[name]['accuracy'] * 100 for name in names]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(names, accuracies, color=['#3498db', '#2ecc71', '#e67e22'], edgecolor='black')
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{acc:.1f}%', ha='center')
    plt.title('Iris Model Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Saved: {output_path.name}')


def visualize_confusion_matrix(results, y_test, output_path: Path = BASE_DIR / 'iris_confusion_matrix.png'):
    best_name = max(results, key=lambda name: results[name]['accuracy'])
    best = results[best_name]
    labels = sorted(y_test.unique())

    disp = ConfusionMatrixDisplay(best['confusion_matrix'], display_labels=labels)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix: {best_name}')
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f'Best model: {best_name}')
    print(classification_report(y_test, best['predictions']))
    print(f'Saved: {output_path.name}')


def visualize_feature_importance(results, output_path: Path = BASE_DIR / 'iris_feature_importance.png'):
    rf_model = results['Random Forest']['model']
    importances = rf_model.feature_importances_
    ranking = np.argsort(importances)[::-1]

    ordered_features = [DATA_COLUMNS[i] for i in ranking]
    ordered_importances = importances[ranking]

    plt.figure(figsize=(9, 5))
    plt.barh(ordered_features, ordered_importances, color='#9b59b6', edgecolor='black')
    plt.gca().invert_yaxis()
    plt.xlabel('Importance Score')
    plt.title('Iris Feature Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Saved: {output_path.name}')


def visualize_species_scatter(iris_df, output_path: Path = BASE_DIR / 'iris_scatter.png'):
    plt.figure(figsize=(8, 6))
    for species, group in iris_df.groupby('species'):
        plt.scatter(group['petal_length_cm'], group['petal_width_cm'], label=species, alpha=0.75, s=40)
    plt.title('Iris Species by Petal Measurements')
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Saved: {output_path.name}')


def main():
    iris_df, X_train, X_test, y_train, y_test = load_and_explore_data()
    results = train_and_compare_models(X_train, X_test, y_train, y_test)

    print('\n' + '=' * 70)
    print('STEP 3: GENERATING VISUALIZATIONS')
    print('=' * 70)
    visualize_model_comparison(results)
    visualize_confusion_matrix(results, y_test)
    visualize_feature_importance(results)
    visualize_species_scatter(iris_df)

    print('\nRun this file with one of:')
    print('  - py week5/iris_train_compare.py   (Windows)')
    print('  - python week5/iris_train_compare.py')
    print('  - python3 week5/iris_train_compare.py')


if __name__ == '__main__':
    main()