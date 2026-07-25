# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY

A company has collected a dataset containing various input features and corresponding numerical output values related to a specific problem (such as sales, price, or demand prediction). The relationship between the input variables and output is complex and cannot be accurately modeled using simple statistical methods.

To solve this problem, the company wants to develop a neural network-based regression model that can learn patterns from the existing data. The model should be trained using historical data so that it can understand how input features influence the output values.

Once trained, the model will be used to predict continuous numerical outputs for new, unseen data points. This will help the company make better decisions based on accurate predictions.

The goal is to minimize prediction error and improve the model’s performance using appropriate training techniques such as backpropagation and optimization algorithms

## Neural Network Model

<img width="974" height="557" alt="image" src="https://github.com/user-attachments/assets/6230d1bb-97d0-40bb-8ddb-a8e455796a24" />

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: ROHITH S

### Register Number: 212225240122

```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        #Include your code here
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv("D:\\Exp-1 DL.csv")
X = df.iloc[:, 0:1].values
Y = df.iloc[:, 1:2].values
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X = x_scaler.fit_transform(X)
Y = y_scaler.fit_transform(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train = torch.FloatTensor(X_train)
Y_train = torch.FloatTensor(Y_train)
X_test = torch.FloatTensor(X_test)
Y_test = torch.FloatTensor(Y_test)


# Initialize the Model, Loss Function, and Optimizer

class RegressionNN(nn.Module):
    def __init__(self):
        super(RegressionNN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),

            nn.Linear(16, 8),
            nn.ReLU(),

            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.network(x)

model = RegressionNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    #Include your code here
epochs = 1000

loss_history = []

for epoch in range(epochs):

    prediction = model(X_train)

    loss = criterion(prediction, Y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if (epoch+1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] Loss = {loss.item():.6f}")

with torch.no_grad():
    predicted = model(X_test)

test_loss = criterion(predicted, Y_test)

print("\nTest Loss:", test_loss.item())

predicted_original = y_scaler.inverse_transform(predicted.numpy())
actual_original = y_scaler.inverse_transform(Y_test.numpy())

print("\nActual vs Predicted")

for actual, pred in zip(actual_original, predicted_original):
    print(f"Actual = {actual[0]:.2f}    Predicted = {pred[0]:.2f}")

plt.figure(figsize=(8,5))
plt.plot(loss_history)
plt.title("Loss during")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

```

### Dataset Information

<img width="134" height="306" alt="image" src="https://github.com/user-attachments/assets/b15d36dc-5e2e-487d-912d-06a14789f074" />

### OUTPUT

<img width="460" height="397" alt="image" src="https://github.com/user-attachments/assets/51ac339c-520f-459d-8c6d-d15a1d7d1439" />

<img width="328" height="115" alt="image" src="https://github.com/user-attachments/assets/5d5b3bd4-b921-486d-9dcf-a4831084df15" />

<img width="319" height="35" alt="image" src="https://github.com/user-attachments/assets/fdd3dd6b-4245-4ce3-b540-91b572b7601e" />

### Training Loss Vs Iteration Plot

<img width="785" height="560" alt="image" src="https://github.com/user-attachments/assets/3a9f3fa6-0239-4102-8a5f-4c265f58d66e" />

### New Sample Data Prediction

<img width="708" height="90" alt="image" src="https://github.com/user-attachments/assets/c22d0286-bbdb-4233-b412-3ab588dc5474" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
