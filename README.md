# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset


## Design Steps

### Step 1:
Import the required libraries such as PyTorch, NumPy, Pandas, and Matplotlib.

### Step 2:
Load the stock price dataset and preprocess it by normalizing and creating sequences.

### Step 3:
Split the dataset into training and testing sets.

### Step 4:
Define the RNN model architecture with input layer, hidden layers, and output layer.

### Step 5:
Initialize the model, define the loss function (MSELoss), and select an optimizer (Adam/RMSProp).

### Step 6:
Train the model using forward pass, loss calculation, backpropagation, and parameter updates.

### Step 7:
Evaluate the model performance on the test dataset.

### Step 8:
Plot True Stock Price vs Predicted Stock Price for comparison.

### Step 9:
Generate numerical predictions on test data.

### Step 10:
Summarize the results and analyze the prediction accuracy of the RNN model.




## Program
#### Name: SUNIL KUMAR T
#### Register Number: 212223240164
```Python 
## Step 2: Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x,h0)
        out = self.fc(out[:, -1, :])
        return out

model = RNNModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the Model
num_epochs = 100
train_losses = [] # Initialize an empty list to store training losses
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    train_losses.append(loss.item()) # Append the loss after each epoch

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')






```

## Output

### True Stock Price, Predicted Stock Price vs time

<img width="888" height="646" alt="Screenshot 2025-09-30 102525" src="https://github.com/user-attachments/assets/8ea2d9aa-d02c-4d23-8243-f2c1cc93c0fd" />

### Predictions 


<img width="289" height="58" alt="image" src="https://github.com/user-attachments/assets/d21acb67-9eee-4bc5-be0c-bafd36b1a072" />

## Result

The Recurrent Neural Network model was successfully implemented for stock price prediction. The model captured temporal dependencies in stock data and produced reliable predictions on test samples, showing the effectiveness of RNNs for time series forecasting.
