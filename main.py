from palmerpenguins import load_penguins
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

def load_data():
    penguins = load_penguins().dropna()
    print("Penguins dataset loaded successfully!")
    print(penguins.head())
    return penguins

def preprocess_data(df):
    # Encode target
    label_encoder = LabelEncoder()
    df['species'] = label_encoder.fit_transform(df['species'])

    # Define features and target
    X = df.drop(columns=['species'])
    y = df['species']

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
