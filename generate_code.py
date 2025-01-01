from sklearn.neural_network import MLPRegressor
import numpy as np

class CodeGenerator:
    def __init__(self):
        self.model = MLPRegressor()
        
    def fit(self, X, Y):
        self.model.fit(X, Y)
        
    def generate(self, prompt):
        X = np.array([[prompt]])  
        y = self.model.predict(X)
        return y[0]

if __name__ == "__main__":
    # Example training data
    X_train = np.array([[1], [2], [3]])  # Shape (3, 1)
    y_train = np.array([2, 4, 6])  # Shape (3,) or reshape to (3, 1) if necessary

    # Reshape y_train to be a 2D array with shape (3, 1)
    y_train = y_train.reshape(-1, 1)

    code_generator = CodeGenerator()
    code_generator.fit(X_train, y_train)
    
    # Generate a python function to implement a simple algorithm
    prompt = 2  # The input for the model needs to be numeric and match training data features
    code = code_generator.generate(prompt)
    
    # Print the generated code
    print(code)
