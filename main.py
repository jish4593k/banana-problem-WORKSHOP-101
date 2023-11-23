import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np

def create_dataset(num_samples=1000):
    np.random.seed(42)
    money = np.random.uniform(0, 50, num_samples)
    price = np.random.uniform(0, 10, num_samples)
    price_p = np.random.uniform(0, 10, num_samples)
    bananas = money / price

    return money, price, price_p, bananas

def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(3,)),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs=50):
    model.fit(X_train, y_train, epochs=epochs, verbose=0)

def main():
    money, price, price_p, bananas = create_dataset()
    X = np.column_stack((money, price, price_p))
    y = bananas

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model()
    train_model(model, X_train, y_train)

    test_data = np.array([[5, 1, 5], [15, 1, 3], [20, 3, 5]])
    predictions = model.predict(test_data)

    for i, prediction in enumerate(predictions):
        print(f"Predicted Max Bananas for Test {i+1}: {prediction[0]}")

if __name__ == "__main__":
    main()
