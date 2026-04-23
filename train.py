import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model():
    print("Loading dataset...")
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    print("Training model...")
    # Train a simple Random Forest model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Basic model versioning
    model_version = "v1"
    model_filename = f"model_{model_version}.pkl"
    
    print(f"Saving model to {model_filename}...")
    joblib.dump(clf, model_filename)
    
    print("Training complete! Model saved successfully.")

if __name__ == "__main__":
    train_model()
