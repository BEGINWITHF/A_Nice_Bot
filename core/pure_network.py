"""
Truly Pure Neural Network - No External Libraries
Only Python built-ins - the purest form of AI
"""

import math
import random
import json
import os
from datetime import datetime

class PureNeuralNetwork:
    """
    A neural network built with only Python's math module.
    No numpy, no scipy, no external dependencies.
    True tabula rasa.
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize with RANDOM weights - knows nothing
        self.weights_input_hidden = [[random.gauss(0, 0.1) for _ in range(hidden_size)] for _ in range(input_size)]
        self.weights_hidden_output = [[random.gauss(0, 0.1) for _ in range(output_size)] for _ in range(hidden_size)]
        
        # Biases
        self.bias_hidden = [0.0] * hidden_size
        self.bias_output = [0.0] * output_size
        
        # Training history
        self.training_history = []
        self.knowledge_level = 0.0
        
    def sigmoid(self, x):
        """Sigmoid activation function"""
        try:
            return 1 / (1 + math.exp(-max(-500, min(500, x))))
        except OverflowError:
            return 0.0 if x < 0 else 1.0
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid"""
        return x * (1 - x)
    
    def forward(self, X):
        """Forward pass - process input through network"""
        # Input to hidden layer
        self.hidden_input = []
        self.hidden_output = []
        for j in range(self.hidden_size):
            total = self.bias_hidden[j]
            for i in range(self.input_size):
                total += X[i] * self.weights_input_hidden[i][j]
            self.hidden_input.append(total)
            self.hidden_output.append(self.sigmoid(total))
        
        # Hidden to output layer
        self.final_input = []
        self.final_output = []
        for k in range(self.output_size):
            total = self.bias_output[k]
            for j in range(self.hidden_size):
                total += self.hidden_output[j] * self.weights_hidden_output[j][k]
            self.final_input.append(total)
            self.final_output.append(self.sigmoid(total))
        
        return self.final_output
    
    def backward(self, X, y, output):
        """Backward pass - learn from mistakes"""
        # Calculate error
        error = [y[i] - output[i] for i in range(self.output_size)]
        
        # Output layer gradient
        output_gradient = [error[i] * self.sigmoid_derivative(output[i]) for i in range(self.output_size)]
        
        # Hidden layer error
        hidden_error = [0.0] * self.hidden_size
        for j in range(self.hidden_size):
            for k in range(self.output_size):
                hidden_error[j] += output_gradient[k] * self.weights_hidden_output[j][k]
        
        hidden_gradient = [hidden_error[j] * self.sigmoid_derivative(self.hidden_output[j]) for j in range(self.hidden_size)]
        
        # Update weights - THIS IS THE LEARNING
        for j in range(self.hidden_size):
            for k in range(self.output_size):
                self.weights_hidden_output[j][k] += self.hidden_output[j] * output_gradient[k] * self.learning_rate
        
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.weights_input_hidden[i][j] += X[i] * hidden_gradient[j] * self.learning_rate
        
        # Update biases
        for k in range(self.output_size):
            self.bias_output[k] += output_gradient[k] * self.learning_rate
        
        for j in range(self.hidden_size):
            self.bias_hidden[j] += hidden_gradient[j] * self.learning_rate
        
        return sum(abs(e) for e in error) / len(error)
    
    def train(self, X, y, epochs=100):
        """Train the network"""
        for epoch in range(epochs):
            output = self.forward(X)
            error = self.backward(X, y, output)
            
            self.training_history.append({
                "epoch": epoch,
                "error": error,
                "timestamp": datetime.now().isoformat()
            })
        
        # Update knowledge level
        if self.training_history:
            initial_error = self.training_history[0]["error"]
            final_error = self.training_history[-1]["error"]
            self.knowledge_level = max(0, min(1, 1 - (final_error / max(initial_error, 0.001))))
    
    def predict(self, X):
        """Make a prediction"""
        return self.forward(X)
    
    def learn_from_interaction(self, input_data, expected_output):
        """Learn from a single interaction"""
        output = self.forward(input_data)
        error = self.backward(input_data, expected_output, output)
        
        return {
            "input": input_data,
            "output": output,
            "expected": expected_output,
            "error": error,
            "knowledge_level": self.knowledge_level
        }
    
    def get_stats(self):
        """Get statistics about the network"""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "knowledge_level": self.knowledge_level,
            "training_epochs": len(self.training_history)
        }
    
    def save(self, filepath):
        """Save the network to disk"""
        data = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "learning_rate": self.learning_rate,
            "weights_input_hidden": self.weights_input_hidden,
            "weights_hidden_output": self.weights_hidden_output,
            "bias_hidden": self.bias_hidden,
            "bias_output": self.bias_output,
            "knowledge_level": self.knowledge_level,
            "training_history": self.training_history[-100:],
            "saved_at": datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath):
        """Load the network from disk"""
        if not os.path.exists(filepath):
            return False
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.input_size = data["input_size"]
        self.hidden_size = data["hidden_size"]
        self.output_size = data["output_size"]
        self.learning_rate = data["learning_rate"]
        self.weights_input_hidden = data["weights_input_hidden"]
        self.weights_hidden_output = data["weights_hidden_output"]
        self.bias_hidden = data["bias_hidden"]
        self.bias_output = data["bias_output"]
        self.knowledge_level = data["knowledge_level"]
        self.training_history = data.get("training_history", [])
        
        return True
