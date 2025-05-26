from palmerpenguins import load_penguins
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

def main():
    # Load the penguins dataset
    penguins = load_penguins()
    print("Penguins dataset loaded successfully!")
    print(penguins.head())

    # Person A - Clean and split the dataset
    penguins = penguins.dropna()
    label_encoder = LabelEncoder()
    penguins['species'] = label_encoder.fit_transform(penguins['species'])
    penguins['island'] = label_encoder.fit_transform(penguins['island'])
    penguins['sex'] = label_encoder.fit_transform(penguins['sex'])

    X = penguins.drop('species', axis=1)
    y = penguins['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Dataset split complete.")

    # Person B - Create the XGBoost model
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    print("XGBoost model created.")

    # Person C - Fit the model
    model.fit(X_train, y_train)
    print("Model training complete.")

if __name__ == "__main__":
    main()
