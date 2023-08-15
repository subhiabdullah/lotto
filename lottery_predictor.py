import random
import numpy as np
from collections import Counter
from statistics import mean, median, mode
from keras.models import Sequential
from keras.layers import Dense
import logging
import unittest
import matplotlib.pyplot as plt
from keras.models import load_model
import os
from draws_data import draws
from draws_data import draws
import draws_data
draws = draws_data.draws

# Überprüfen Sie die Daten direkt nach dem Import
print("Checking data format...")
for idx, draw in enumerate(draws):
    if not isinstance(draw, list) or len(draw) != 7:  # Angenommen, jede Ziehung hat 7 Zahlen
        print(f"Error in draw {idx}: {draw}")

# ... [Rest des Codes, z.B. die Definition der LotteryPredictor-Klasse]



class LotteryPredictor:

    def analyze_draw(self, draw):
        return {
            "mean": mean(draw),
            "median": median(draw),
            "mode": self.safe_mode(draw),
            "frequencies": self.number_frequencies(draw)
        }

    def safe_mode(self, data):
        try:
            return mode(data)
        except:
            return "No unique mode"

    def number_frequencies(self, draw):
        return dict(Counter(draw))

    def most_common(self):
        common_numbers = self.counter.most_common(6)
        prediction = [num for num, _ in common_numbers]
        explanation = "Most common numbers"
        return prediction, explanation

    def least_common(self):
        common_numbers = self.counter.most_common()[:-7:-1]
        prediction = [num for num, _ in common_numbers]
        explanation = "Least common numbers"
        return prediction, explanation
    
    
    
    
    def normalize_data(self, data):
        max_val = max(self.get_all_numbers())  # Änderung hier
        min_val = min(self.get_all_numbers())  # Änderung hier
        return [(x - min_val) / (max_val - min_val) for x in data]

    def neural_network(self):
        X = np.array(self.draws)[:, :5]
        y = np.array(self.draws)[:, 5:]
        
        # Normalize data
        X = self.normalize_data(X)
        y = self.normalize_data(y)
        
        model = Sequential()
        model.add(Dense(10, input_dim=5, activation='relu'))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        
        try:
            model.fit(X, y, epochs=100, verbose=0)
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            return [], "Error during prediction"
        
        last_draw = np.array(self.draws[-1])[:5]
        normalized_last_draw = self.normalize_data(last_draw)
        
        try:
            prediction = model.predict(np.array([normalized_last_draw]))[0].astype(int).tolist()
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return [], "Error during prediction"
        
        explanation = "Neural network prediction"
        return prediction, explanation
    
    def get_all_numbers(self):
        """Returns a flattened list of all numbers from the draws."""
        return [num for draw in self.draws for num in draw]
    


    # ... [rest of the methods]

    def evaluate_model(self, test_draws):
        """Evaluates the neural network model using test data."""
        X_train = np.array(self.draws)[:, :5]
        y_train = np.array(self.draws)[:, 5:]
        X_test = np.array(test_draws)[:, :5]
        y_test = np.array(test_draws)[:, 5:]

        # Normalize data
        X_train = self.normalize_data(X_train)
        y_train = self.normalize_data(y_train)
        X_test = self.normalize_data(X_test)
        y_test = self.normalize_data(y_test)

        model = Sequential()
        model.add(Dense(10, input_dim=5, activation='relu'))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, y_train, epochs=self.nn_epochs, verbose=0)

        loss = model.evaluate(X_test, y_test, verbose=0)
        return loss

    def plot_number_frequencies(self):
        """Plots the frequency of each number in the draws."""
        frequencies = self.number_frequencies(self.get_all_numbers())  # Änderung hier
        plt.bar(frequencies.keys(), frequencies.values())
        plt.xlabel('Number')
        plt.ylabel('Frequency')
        plt.title('Number Frequencies in Draws')
        plt.show()
        
    def __init__(self, draws, nn_epochs=100, model_path="lottery_model.h5"):
        self.draws = draws
        self.nn_epochs = nn_epochs
        self.model_path = model_path
        # ... [rest of the initialization]

    # ... [rest of the methods]

    def save_model(self):
        """Saves the trained neural network model to a file."""
        model = self._train_neural_network()
        model.save(self.model_path)

    def load_model(self):
        """Loads the neural network model from a file."""
        if os.path.exists(self.model_path):
            return load_model(self.model_path)
        else:
            print(f"No model found at {self.model_path}. Training a new one...")
            return self._train_neural_network()

    def _train_neural_network(self):
        """Private method to train the neural network."""
        X = np.array(self.draws)[:, :5]
        y = np.array(self.draws)[:, 5:]

        # Normalize data
        X = self.normalize_data(X)
        y = self.normalize_data(y)

        model = Sequential()
        model.add(Dense(10, input_dim=5, activation='relu'))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X, y, epochs=self.nn_epochs, verbose=0)

        return model
    
    
    def future_predictions(self, num_predictions):
        vorhersagen = []
        for _ in range(num_predictions):
            letzte_ziehung = self.draws[-1][:5]
            vorhergesagte_zahlen, _ = self.neural_network()
            vorhersagen.append(letzte_ziehung + vorhergesagte_zahlen)  # Kombinieren Sie die letzten 5 Zahlen mit den 2 vorhergesagten Zahlen
            self.draws.append(letzte_ziehung + vorhergesagte_zahlen)  # Aktualisieren Sie die Ziehungen mit der neuen Vorhersage
        return vorhersagen


if __name__ == "__main__":
    predictor = LotteryPredictor(draws)  # Erstellen Sie eine Instanz von LotteryPredictor

    action = input("What would you like to do? (analyze/predict/plot/exit): ")

    while action != "exit":
        if action == "analyze":
            for draw in draws:
                analysis = predictor.analyze_draw(draw)
                print(f"Draw: {draw}")
                for key, value in analysis.items():
                    print(f"{key.capitalize()}: {value}")
                print("-" * 40)
        elif action == "predict":
            future_draws = predictor.future_predictions(10)
            for idx, prediction in enumerate(future_draws, start=1):
                print(f"Prediction {idx}: {prediction}")
            print("-" * 40)
        elif action == "plot":
            predictor.plot_number_frequencies()
        else:
            print("Invalid action. Please choose from (analyze/predict/plot/exit).")

        action = input("What would you like to do next? (analyze/predict/plot/exit): ")

    print("Goodbye!")

    
    