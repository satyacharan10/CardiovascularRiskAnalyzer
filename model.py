import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Load your dataset
data = pd.read_csv('C:/Users/satya/Downloads/Project-2/Project-2/heart.csv')

# Feature and target separation
X = data.drop('target', axis=1)
y = data['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Define and fit the scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Define and fit the classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=0)
classifier.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(classifier, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

def load_model_and_scaler():
    global classifier, scaler
    classifier = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
