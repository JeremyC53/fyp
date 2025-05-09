import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, learning_curve

def extract_features(file_path):
    """
    Reads a CSV file and computes the mean and standard deviation of
    the quaternion (qw, qx, qy, qz) and acceleration (ax, ay, az) columns.
    Returns a 14-element feature vector.
    """
    df = pd.read_csv(file_path)
    features = []
    # Use only quaternion and acceleration columns.
    for col in ['qw', 'qx', 'qy', 'qz', 'ax', 'ay', 'az']:
        features.append(df[col].mean())
        features.append(df[col].std())
    return np.array(features)

def load_data():
    """
    Loads CSV files for the three motions from their directories and assigns
    a discrete label: 0 for bicep curls, 1 for hammer curls, and 2 for lateral raises.
    """
    motions = ['bicep', 'hammer', 'lat']
    label_map = {'bicep': 0, 'hammer': 1, 'lat': 2}
    
    X = []  # Feature vectors
    y = []  # Class labels
    
    for motion in motions:
        for rep in range(1, 21):  # rep_id from 1 to 20
            file_path = os.path.join('motion_class', motion, f'rep{rep}.csv')
            if os.path.exists(file_path):
                features = extract_features(file_path)
                X.append(features)
                y.append(label_map[motion])
            else:
                print(f"Warning: {file_path} does not exist.")
    
    return np.array(X), np.array(y)

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, filename='confusion_matrix.png'):
    """
    Plots and saves a confusion matrix.
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(filename)
    plt.close()

def plot_learning_curve(estimator, X, y, title='Learning Curve (Accuracy)', filename='accuracy_graph.png'):
    """
    Generates and saves a learning curve plot using cross-validation accuracy.
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring='accuracy', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10))
    
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', label="Training Accuracy")
    plt.plot(train_sizes, test_scores_mean, 'o-', label="CV Accuracy")
    plt.title(title)
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def main():
    # Load the dataset
    X, y = load_data()
    
    # Split into training (60%) and validation (40%) sets with stratification.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y)
    
    # Create and train the Random Forest classifier.
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model on the validation set.
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print("Validation Accuracy:", accuracy)
    
    # Compute and plot the confusion matrix.
    cm = confusion_matrix(y_val, y_pred)
    class_names = ['bicep', 'hammer', 'lat']
    plot_confusion_matrix(cm, classes=class_names)
    
    # Plot the learning curve.
    plot_learning_curve(model, X, y)
    
    # Save the trained model to a pickle file.
    with open('rf_motion_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved to 'rf_motion_model.pkl'.")
    print("Confusion matrix saved as 'confusion_matrix.png'.")
    print("Learning curve saved as 'accuracy_graph.png'.")

if __name__ == '__main__':
    main()
