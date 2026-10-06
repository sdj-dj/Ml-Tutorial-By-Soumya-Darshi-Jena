# Complete AI/ML Pipeline with Python

A complete beginner-friendly AI/ML project demonstrating the end-to-end machine learning and deep learning workflow using **Python, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Joblib, and PyTorch**.

The project starts with raw data and goes through data analysis, preprocessing, classical machine learning, neural network training, evaluation, model saving/loading, and prediction.

---

## 🚀 Project Overview

This project predicts a student's **final exam score** based on:

- Study hours
- Attendance percentage
- Previous exam score

Two different approaches are demonstrated:

1. **Classical Machine Learning using Scikit-learn**
2. **Deep Learning using PyTorch**

The project helps understand how a real AI/ML pipeline is constructed from beginning to end.

---

## 🔄 Complete Pipeline

```text
Raw Data
   ↓
Data Insertion
   ↓
Pandas
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Duplicate Removal
   ↓
Data Visualization
   ↓
Correlation Analysis
   ↓
Feature / Target Separation
   ↓
Train / Test Split
   ↓
Feature Scaling
   ↓
Scikit-learn
   ↓
Linear Regression
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Save Model
   ↓
Load Model
   ↓
New Prediction
   ↓
PyTorch
   ↓
Neural Network
   ↓
Loss Function
   ↓
Optimizer
   ↓
Forward Pass
   ↓
Backpropagation
   ↓
Weight Updates
   ↓
Training
   ↓
Testing
   ↓
Evaluation
   ↓
Save PyTorch Model
   ↓
Load PyTorch Model
   ↓
New Prediction
```

---

# 📌 Technologies Used

## Python

The primary programming language used for the entire project.

---

## NumPy

Used for:

- Numerical operations
- Array operations
- Mathematical calculations
- RMSE calculation
- Converting and processing numerical data

Example:

```python
rmse = np.sqrt(mse)
```

---

## Pandas

Used for:

- Creating DataFrames
- Loading and manipulating datasets
- Data inspection
- Missing-value checking
- Duplicate removal
- Feature and target preparation

Example:

```python
df = pd.DataFrame(data)
```

---

## Matplotlib

Used for data visualization.

Example:

```python
plt.title("Study Hours vs Final Score")
plt.xlabel("Study Hours")
plt.ylabel("Final Score")
plt.show()
```

---

## Seaborn

Used for statistical visualization and understanding relationships between features.

Example:

```python
sns.scatterplot(
    data=df,
    x="study_hours",
    y="final_score"
)
```

Correlation visualization:

```python
sns.heatmap(
    df.corr(),
    annot=True
)
```

---

## Scikit-learn

Used for classical machine learning.

The project uses Scikit-learn for:

- Train/test splitting
- Feature scaling
- Linear Regression
- Model training
- Prediction
- Evaluation metrics

Main components:

```text
train_test_split
StandardScaler
LinearRegression
mean_absolute_error
mean_squared_error
r2_score
```

---

## Joblib

Used to save and load the trained Scikit-learn model.

```python
joblib.dump(model, "student_model.pkl")
```

Load it later:

```python
model = joblib.load("student_model.pkl")
```

This allows us to make predictions without retraining the model.

---

## PyTorch

PyTorch is used for the deep learning portion of the project.

The project demonstrates:

- Neural networks
- Layers
- Activation functions
- Loss functions
- Optimizers
- Forward propagation
- Backpropagation
- Weight updates
- Training loops
- Model evaluation
- Model saving/loading

---

# 🧠 Dataset

The example dataset contains four columns:

| Feature | Description |
|---|---|
| `study_hours` | Number of hours a student studies |
| `attendance` | Percentage of classes attended |
| `previous_score` | Previous exam score |
| `final_score` | Final exam score |

The first three columns are the **input features**.

```text
study_hours
attendance
previous_score
```

The target variable is:

```text
final_score
```

---

# 1️⃣ Data Insertion

The dataset is created using a Python dictionary.

```python
data = {
    "study_hours": [...],
    "attendance": [...],
    "previous_score": [...],
    "final_score": [...]
}
```

It is then converted into a Pandas DataFrame:

```python
df = pd.DataFrame(data)
```

---

# 2️⃣ Data Understanding

Before training a model, we need to understand the dataset.

The project uses:

```python
df.info()
```

and:

```python
df.describe()
```

This helps identify:

- Number of rows
- Number of columns
- Data types
- Statistical properties
- Minimum values
- Maximum values
- Mean
- Standard deviation

---

# 3️⃣ Missing Value Detection

Missing values are checked using:

```python
df.isnull().sum()
```

If missing values exist, they can be handled using techniques such as:

```python
df["column"] = df["column"].fillna(
    df["column"].mean()
)
```

The appropriate technique depends on the dataset.

---

# 4️⃣ Duplicate Removal

Duplicate records can negatively affect model training.

The project removes duplicates using:

```python
df = df.drop_duplicates()
```

---

# 5️⃣ Data Visualization

Visualization helps us understand patterns in the data.

For example:

```python
sns.scatterplot(
    data=df,
    x="study_hours",
    y="final_score"
)
```

This helps us visually investigate the relationship between study hours and final scores.

---

# 6️⃣ Correlation Analysis

Correlation measures how strongly numerical variables are related.

The project calculates correlation using:

```python
df.corr()
```

A heatmap is then used for visualization:

```python
sns.heatmap(
    df.corr(),
    annot=True
)
```

Correlation values generally range from:

```text
-1 → Strong negative relationship
 0 → No linear relationship
+1 → Strong positive relationship
```

Correlation does **not** by itself prove causation.

---

# 7️⃣ Features and Target

Machine learning separates the data into:

### Features

The inputs used by the model:

```python
X = df[
    [
        "study_hours",
        "attendance",
        "previous_score"
    ]
]
```

### Target

The value the model attempts to predict:

```python
y = df["final_score"]
```

Conceptually:

```text
X = Input
 ↓
Machine Learning Model
 ↓
y = Prediction
```

---

# 8️⃣ Train/Test Split

The dataset is divided into training and testing data.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The training data is used to learn patterns.

The test data is used to evaluate the model on unseen examples.

```text
Dataset
   ↓
 ┌───────────────┐
 ↓               ↓
Training        Testing
 80%              20%
```

---

# 9️⃣ Feature Scaling

The project uses `StandardScaler`:

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

Scaling puts features onto a comparable numerical scale.

An important rule is:

```text
Training data:
fit_transform()

Test data:
transform()
```

We should not fit the scaler independently on the test set because that can introduce information leakage.

---

# 🔟 Classical Machine Learning with Scikit-learn

The first model is:

```python
LinearRegression()
```

The model is created:

```python
model = LinearRegression()
```

Then trained:

```python
model.fit(
    X_train_scaled,
    y_train
)
```

Conceptually:

```text
Training Data
     ↓
Linear Regression
     ↓
Learned Parameters
     ↓
Trained Model
```

---

# 1️⃣1️⃣ Prediction

After training, the model predicts the test data:

```python
predictions = model.predict(
    X_test_scaled
)
```

The model takes the input features and produces predicted final scores.

---

# 1️⃣2️⃣ Model Evaluation

The project uses four regression metrics.

## MAE

Mean Absolute Error:

```python
mean_absolute_error(
    y_test,
    predictions
)
```

MAE represents the average absolute prediction error.

---

## MSE

Mean Squared Error:

```python
mean_squared_error(
    y_test,
    predictions
)
```

MSE squares the prediction errors before averaging them.

---

## RMSE

Root Mean Squared Error:

```python
rmse = np.sqrt(mse)
```

RMSE is in the same units as the target variable.

---

## R²

R-squared:

```python
r2_score(
    y_test,
    predictions
)
```

R² measures how well the model explains variation in the target.

---

# 1️⃣3️⃣ Saving the Scikit-learn Model

The trained model is saved using Joblib:

```python
joblib.dump(
    model,
    "student_model.pkl"
)
```

The scaler is also saved:

```python
joblib.dump(
    scaler,
    "student_scaler.pkl"
)
```

Saving both is important because future input data must be transformed using the same preprocessing procedure.

---

# 1️⃣4️⃣ Loading the Model

The model can later be loaded:

```python
loaded_model = joblib.load(
    "student_model.pkl"
)
```

The scaler is also loaded:

```python
loaded_scaler = joblib.load(
    "student_scaler.pkl"
)
```

No retraining is required.

---

# 1️⃣5️⃣ New Student Prediction

Suppose a new student has:

```text
Study Hours     = 8
Attendance      = 90%
Previous Score  = 80
```

The input is created:

```python
new_student = pd.DataFrame(
    [[8, 90, 80]],
    columns=[
        "study_hours",
        "attendance",
        "previous_score"
    ]
)
```

The same scaler used during training is applied:

```python
new_student_scaled = loaded_scaler.transform(
    new_student
)
```

Then the trained model makes the prediction:

```python
prediction = loaded_model.predict(
    new_student_scaled
)
```

---

# 🧠 Deep Learning Section

After classical machine learning, the project moves to **PyTorch**.

The purpose is to demonstrate how the same type of prediction problem can be solved using a neural network.

---

# 1️⃣6️⃣ Convert Data to PyTorch Tensors

PyTorch works with tensors.

```python
X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32
)
```

The target is also converted:

```python
y_train_tensor = torch.tensor(
    y_train.values,
    dtype=torch.float32
).reshape(-1, 1)
```

Conceptually:

```text
NumPy / Pandas
      ↓
PyTorch Tensor
      ↓
Neural Network
```

---

# 1️⃣7️⃣ Create a Neural Network

The project defines:

```python
class StudentNeuralNetwork(nn.Module):
```

The architecture is:

```text
Input Layer
3 Features
    ↓
Linear Layer
3 → 32
    ↓
ReLU
    ↓
Linear Layer
32 → 16
    ↓
ReLU
    ↓
Output Layer
16 → 1
```

The neural network learns parameters that map the three input features to the final score.

---

# 1️⃣8️⃣ Loss Function

The project uses:

```python
nn.MSELoss()
```

The loss function measures the difference between:

```text
Actual Score
     vs
Predicted Score
```

The objective of training is to reduce the loss.

---

# 1️⃣9️⃣ Optimizer

The project uses Adam:

```python
optimizer = optim.Adam(
    nn_model.parameters(),
    lr=0.001
)
```

The optimizer updates the neural network's parameters using the gradients calculated during backpropagation.

---

# 2️⃣0️⃣ Neural Network Training

The training process follows:

```text
Input
 ↓
Forward Pass
 ↓
Prediction
 ↓
Loss Calculation
 ↓
Gradient Calculation
 ↓
Backpropagation
 ↓
Optimizer
 ↓
Weight Update
 ↓
Repeat
```

The code:

```python
predictions = nn_model(
    X_train_tensor
)

loss = loss_function(
    predictions,
    y_train_tensor
)

optimizer.zero_grad()

loss.backward()

optimizer.step()
```

This process is repeated for multiple epochs.

---

# 2️⃣1️⃣ PyTorch Testing

After training:

```python
nn_model.eval()
```

Predictions are generated without calculating gradients:

```python
with torch.no_grad():

    test_predictions = nn_model(
        X_test_tensor
    )
```

This is used during inference/evaluation.

---

# 2️⃣2️⃣ PyTorch Evaluation

The PyTorch predictions are evaluated using the same regression metrics:

```text
MAE
MSE
RMSE
R²
```

This allows us to compare the classical ML model and neural network.

---

# 2️⃣3️⃣ Save PyTorch Model

The neural network weights are saved:

```python
torch.save(
    nn_model.state_dict(),
    "student_neural_network.pth"
)
```

---

# 2️⃣4️⃣ Load PyTorch Model

The model architecture is recreated:

```python
loaded_nn_model = StudentNeuralNetwork()
```

Then the saved weights are loaded:

```python
loaded_nn_model.load_state_dict(
    torch.load(
        "student_neural_network.pth",
        weights_only=True
    )
)
```

---

# 2️⃣5️⃣ Final Prediction

A new student can now be passed to the loaded PyTorch model.

```text
New Student
     ↓
Preprocessing
     ↓
PyTorch Tensor
     ↓
Trained Neural Network
     ↓
Predicted Final Score
```

---

# 📊 Complete Architecture

```text
                    DATA
                      │
                      ↓
               ┌─────────────┐
               │   Pandas    │
               └──────┬──────┘
                      ↓
              Data Cleaning
                      ↓
             Data Visualization
                      ↓
             Correlation Analysis
                      ↓
              Feature / Target
                      ↓
              Train/Test Split
                      ↓
              StandardScaler
                      ↓
           ┌──────────┴──────────┐
           ↓                     ↓
     Scikit-learn             PyTorch
           ↓                     ↓
   Linear Regression       Neural Network
           ↓                     ↓
       Prediction            Training
           ↓                     ↓
      Evaluation             Loss
                                 ↓
                          Backpropagation
                                 ↓
                             Optimizer
                                 ↓
                              Weights
                                 ↓
                              Testing
           └──────────┬──────────┘
                      ↓
                  Evaluation
                      ↓
                 Save Model
                      ↓
                 Load Model
                      ↓
                New Prediction
```

---

# 🔥 Next-Level AI Roadmap

This project represents the beginning of a much larger AI/ML pipeline.

After mastering this project, the next progression is:

```text
Python
   ↓
NumPy
   ↓
Pandas
   ↓
Matplotlib
   ↓
Seaborn
   ↓
Scikit-learn
   ↓
Classical Machine Learning
   ↓
PyTorch
   ↓
Deep Learning
   ↓
CNNs
   ↓
RNNs
   ↓
Attention
   ↓
Transformers
   ↓
Hugging Face
   ↓
Large Language Models
   ↓
Dataset Preparation
   ↓
Supervised Fine-Tuning
   ↓
LoRA
   ↓
QLoRA
   ↓
LLM Fine-Tuning
   ↓
Evaluation
   ↓
RAG
   ↓
Tool Calling
   ↓
AI Agents
   ↓
Multi-Agent Systems
   ↓
Agentic AI
```

---

# ⚠️ Important Note About LLM Fine-Tuning

The PyTorch neural network in this project demonstrates the **fundamentals of deep learning**, but it is **not an LLM fine-tuning implementation**.

Modern LLM fine-tuning typically uses an ecosystem such as:

```text
PyTorch
   +
Hugging Face Transformers
   +
Hugging Face Datasets
   +
PEFT
   +
LoRA / QLoRA
   +
TRL
```

A typical LLM fine-tuning workflow is:

```text
Dataset
   ↓
Data Cleaning
   ↓
Formatting
   ↓
Tokenizer
   ↓
Base LLM
   ↓
LoRA / QLoRA
   ↓
Supervised Fine-Tuning
   ↓
Evaluation
   ↓
Model Adapter
   ↓
Inference
```

---

# 🎯 Learning Objectives

After completing this project, you should understand:

- How data enters an ML pipeline
- How Pandas is used for data handling
- How missing values and duplicates are handled
- How visualization helps understand data
- What correlation means
- Difference between features and target
- Why train/test splitting is necessary
- Why feature scaling is used
- How Scikit-learn models are trained
- How predictions are generated
- How regression models are evaluated
- How models are saved and loaded
- What PyTorch tensors are
- How a neural network is constructed
- What a loss function does
- What an optimizer does
- How forward propagation works
- How backpropagation works
- How neural networks learn
- How PyTorch models are saved and loaded
- How classical ML differs from deep learning
- How this foundation leads toward LLMs and fine-tuning

---

# 📁 Suggested Project Structure

```text
complete-ai-ml-pipeline/
│
├── main.py
├── README.md
│
├── student_model.pkl
├── student_scaler.pkl
├── student_neural_network.pth
│
└── requirements.txt
```

---

# 📦 Requirements

Create a `requirements.txt` file:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
torch
```

Install everything using:

```bash
pip install -r requirements.txt
```

---

# ▶️ How to Run

Clone/download the project and install the dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python main.py
```

The program will:

1. Create the dataset
2. Analyze the data
3. Check missing values
4. Remove duplicates
5. Visualize the data
6. Analyze correlations
7. Split the dataset
8. Scale the features
9. Train a Scikit-learn model
10. Evaluate the model
11. Save the model
12. Load the model
13. Predict a new student's score
14. Build a PyTorch neural network
15. Train the neural network
16. Evaluate the neural network
17. Save the PyTorch model
18. Load the PyTorch model
19. Predict a new student's score

---

# 🏁 Conclusion

This project demonstrates a complete introductory **AI/ML workflow**, starting from raw data and progressing through classical machine learning and deep learning.

The most important conceptual pipeline is:

```text
DATA
 ↓
PREPROCESSING
 ↓
TRAINING
 ↓
EVALUATION
 ↓
MODEL SAVING
 ↓
INFERENCE
```

The project then provides the foundation for moving toward:

```text
Deep Learning
      ↓
Transformers
      ↓
LLMs
      ↓
Fine-Tuning
      ↓
LoRA / QLoRA
      ↓
RAG
      ↓
Agents
      ↓
Multi-Agent AI
      ↓
Production AI
```

This project is intended as a learning foundation for building more advanced AI systems.
#EXAMPLE:
```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib

import torch
import torch.nn as nn
import torch.optim as optim

data = {
    "study_hours": [
        1, 2, 2, 3, 3,
        4, 4, 5, 5, 6,
        6, 7, 7, 8, 8,
        9, 9, 10, 10, 11
    ],

    "attendance": [
        50, 55, 60, 62, 65,
        68, 70, 72, 75, 78,
        80, 82, 84, 86, 88,
        90, 92, 94, 96, 98
    ],

    "previous_score": [
        40, 45, 48, 50, 53,
        55, 58, 60, 63, 65,
        68, 70, 72, 75, 78,
        80, 83, 86, 90, 92
    ],

    "final_score": [
        42, 47, 50, 53, 55,
        58, 61, 63, 66, 69,
        71, 74, 77, 80, 82,
        85, 87, 90, 94, 96
    ]
}

df = pd.DataFrame(data)

print("\n==============================")
print("RAW DATA")
print("==============================")

print(df)

print("\n==============================")
print("DATA INFORMATION")
print("==============================")

print(df.info())

print("\nSTATISTICS:")
print(df.describe())

print("\n==============================")
print("MISSING VALUES")
print("==============================")

print(df.isnull().sum())

df = df.drop_duplicates()

sns.scatterplot(
    data=df,
    x="study_hours",
    y="final_score"
)

plt.title("Study Hours vs Final Score")
plt.xlabel("Study Hours")
plt.ylabel("Final Score")

plt.show()

sns.histplot(data=df, x="attendance", kde=True)

plt.title("Attendance Distribution")
plt.xlabel("Attendance")
plt.ylabel("Frequency")

plt.show()
print("\n==============================")
print("CORRELATION")
print("==============================")

print(df.corr())


sns.heatmap(
    df.corr(),
    annot=True
)

plt.title("Feature Correlation")

plt.show()

X = df[
    [
        "study_hours",
        "attendance",
        "previous_score"
    ]
]

y = df["final_score"]


print("\nFEATURES:")
print(X)

print("\nTARGET:")
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("\n==============================")
print("DATA SPLIT")
print("==============================")

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n==============================")
print("SCIKIT-LEARN MODEL")
print("==============================")


model = LinearRegression()
model.fit(X_train_scaled, y_train)
sklearn_predictions = model.predict(X_test_scaled)

print("\nPredictions:")
print(sklearn_predictions)

mae = mean_absolute_error(y_test, sklearn_predictions)
mse = mean_squared_error(y_test, sklearn_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,sklearn_predictions)

print("\n==============================")
print("SCIKIT-LEARN EVALUATION")
print("==============================")

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2  :", r2)

joblib.dump(model,"student_model.pkl")
joblib.dump(scaler,"student_scaler.pkl")

print("\nScikit-learn model saved.")

loaded_model = joblib.load("student_model.pkl")
loaded_scaler = joblib.load("student_scaler.pkl")

new_student = pd.DataFrame([[8, 90, 80]], columns=["study_hours","attendance", "previous_score"])
new_student_scaled = loaded_scaler.transform(new_student)
prediction = loaded_model.predict(new_student_scaled)

print("\n==============================")
print("NEW STUDENT PREDICTION")
print("==============================")

print("Predicted final score:",prediction[0])

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

class StudentNeuralNetwork(nn.Module):

    def __init__(self):
      super().__init__()
      self.network = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
      )

    def forward(self, x):
      return self.network(x)

nn_model = StudentNeuralNetwork()
print("\n==============================")
print("PYTORCH MODEL")
print("==============================")
print(nn_model)
loss_function = nn.MSELoss()
optimizer = optim.Adam(
    nn_model.parameters(),
    lr=0.001
)
epochs = 1000
print("\n==============================")
print("PYTORCH TRAINING")
print("==============================")
for epoch in range(epochs):
    predictions = nn_model(
        X_train_tensor
    )
    loss = loss_function(
        predictions,
        y_train_tensor
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Loss: {loss.item():.4f}"
        )
nn_model.eval()
with torch.no_grad():

    test_predictions = nn_model(
        X_test_tensor
    )
test_predictions_numpy = (
    test_predictions.numpy()
)
pytorch_mae = mean_absolute_error(
    y_test,
    test_predictions_numpy
)
pytorch_mse = mean_squared_error(
    y_test,
    test_predictions_numpy
)
pytorch_rmse = np.sqrt(
    pytorch_mse
)
pytorch_r2 = r2_score(
    y_test,
    test_predictions_numpy
)
print("\n==============================")
print("PYTORCH EVALUATION")
print("==============================")
print("MAE :", pytorch_mae)
print("MSE :", pytorch_mse)
print("RMSE:", pytorch_rmse)
print("R2  :", pytorch_r2)
torch.save(
    nn_model.state_dict(),
    "student_neural_network.pth"
)
print("\nPyTorch model saved.")
loaded_nn_model = StudentNeuralNetwork()
loaded_nn_model.load_state_dict(
    torch.load(
        "student_neural_network.pth",
        weights_only=True
    )
)
loaded_nn_model.eval()
new_student_tensor = torch.tensor(
    new_student_scaled,
    dtype=torch.float32
)
with torch.no_grad():

    nn_prediction = loaded_nn_model(
        new_student_tensor
    )
print("\n==============================")
print("PYTORCH NEW PREDICTION")
print("==============================")
print(
    "Predicted score:",
    nn_prediction.item()
)

```

#OUTPUT FOR THIS PROGRAM:

```text
<>:433: SyntaxWarning: invalid escape sequence '\/'
<>:433: SyntaxWarning: invalid escape sequence '\/'
/tmp/ipykernel_1462/2948059240.py:433: SyntaxWarning: invalid escape sequence '\/'
  \/

==============================
RAW DATA
==============================
    study_hours  attendance  previous_score  final_score
0             1          50              40           42
1             2          55              45           47
2             2          60              48           50
3             3          62              50           53
4             3          65              53           55
5             4          68              55           58
6             4          70              58           61
7             5          72              60           63
8             5          75              63           66
9             6          78              65           69
10            6          80              68           71
11            7          82              70           74
12            7          84              72           77
13            8          86              75           80
14            8          88              78           82
15            9          90              80           85
16            9          92              83           87
17           10          94              86           90
18           10          96              90           94
19           11          98              92           96

==============================
DATA INFORMATION
==============================
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 20 entries, 0 to 19
Data columns (total 4 columns):
 #   Column          Non-Null Count  Dtype
---  ------          --------------  -----
 0   study_hours     20 non-null     int64
 1   attendance      20 non-null     int64
 2   previous_score  20 non-null     int64
 3   final_score     20 non-null     int64
dtypes: int64(4)
memory usage: 772.0 bytes
None

STATISTICS:
       study_hours  attendance  previous_score  final_score
count    20.000000    20.00000       20.000000    20.000000
mean      6.000000    77.25000       66.550000    70.000000
std       2.991215    14.14167       15.404972    16.212406
min       1.000000    50.00000       40.000000    42.000000
25%       3.750000    67.25000       54.500000    57.250000
50%       6.000000    79.00000       66.500000    70.000000
75%       8.250000    88.50000       78.500000    82.750000
max      11.000000    98.00000       92.000000    96.000000

==============================
MISSING VALUES
==============================
study_hours       0
attendance        0
previous_score    0
final_score       0
dtype: int64
```

<img width="562" height="455" alt="image" src="https://github.com/user-attachments/assets/3248c9ca-b1b7-48b0-9014-45e19c96df5f" />

<img width="563" height="455" alt="image" src="https://github.com/user-attachments/assets/ad83b4c8-4744-4915-b336-b9a6c26093bc" />


```text
==============================
CORRELATION
==============================
                study_hours  attendance  previous_score  final_score
study_hours        1.000000    0.990401        0.994846     0.996309
attendance         0.990401    1.000000        0.993733     0.995377
previous_score     0.994846    0.993733        1.000000     0.999308
final_score        0.996309    0.995377        0.999308     1.000000
```
<img width="533" height="435" alt="image" src="https://github.com/user-attachments/assets/8db2072a-2d23-4c1f-9c2e-c13527075d87" />

```text
FEATURES:
    study_hours  attendance  previous_score
0             1          50              40
1             2          55              45
2             2          60              48
3             3          62              50
4             3          65              53
5             4          68              55
6             4          70              58
7             5          72              60
8             5          75              63
9             6          78              65
10            6          80              68
11            7          82              70
12            7          84              72
13            8          86              75
14            8          88              78
15            9          90              80
16            9          92              83
17           10          94              86
18           10          96              90
19           11          98              92

TARGET:
0     42
1     47
2     50
3     53
4     55
5     58
6     61
7     63
8     66
9     69
10    71
11    74
12    77
13    80
14    82
15    85
16    87
17    90
18    94
19    96
Name: final_score, dtype: int64

==============================
DATA SPLIT
==============================
Training samples: 16
Testing samples : 4

==============================
SCIKIT-LEARN MODEL
==============================

Predictions:
[41.14503911 90.50451452 84.64674024 46.58275175]

==============================
SCIKIT-LEARN EVALUATION
==============================
MAE : 0.5324958562330995
MSE : 0.3210953969682804
RMSE: 0.5666528010768855
R2  : 0.999314630956311

Scikit-learn model saved.

==============================
NEW STUDENT PREDICTION
==============================
Predicted final score: 83.8336895518916

==============================
PYTORCH MODEL
==============================
StudentNeuralNetwork(
  (network): Sequential(
    (0): Linear(in_features=3, out_features=32, bias=True)
    (1): ReLU()
    (2): Linear(in_features=32, out_features=16, bias=True)
    (3): ReLU()
    (4): Linear(in_features=16, out_features=1, bias=True)
  )
)

==============================
PYTORCH TRAINING
==============================
Epoch 100/1000 Loss: 4624.6587
Epoch 200/1000 Loss: 2401.4790
Epoch 300/1000 Loss: 772.5297
Epoch 400/1000 Loss: 468.2254
Epoch 500/1000 Loss: 294.3404
Epoch 600/1000 Loss: 163.0529
Epoch 700/1000 Loss: 86.6309
Epoch 800/1000 Loss: 46.3422
Epoch 900/1000 Loss: 26.5891
Epoch 1000/1000 Loss: 16.2126

==============================
PYTORCH EVALUATION
==============================
MAE : 9.932214736938477
MSE : 171.16571044921875
RMSE: 13.08303139372595
R2  : 0.6346516609191895

PyTorch model saved.

==============================
PYTORCH NEW PREDICTION
==============================
Predicted score: 83.590576171875
```
