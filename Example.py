# ============================================================
# COMPLETE AI/ML PIPELINE IN ONE PYTHON FILE
# ============================================================
#
# Pipeline:
#
# Raw Data
#    ↓
# Pandas
#    ↓
# Cleaning
#    ↓
# Visualization
#    ↓
# Scikit-learn
#    ↓
# Train/Test Split
#    ↓
# Classical ML Model
#    ↓
# Evaluation
#    ↓
# PyTorch Neural Network
#    ↓
# Training
#    ↓
# Evaluation
#    ↓
# Save Model
#
# NOTE:
# Real LLM fine-tuning normally uses Hugging Face
# Transformers + PEFT/LoRA/QLoRA rather than this
# small PyTorch example.
# ============================================================


# ============================================================
"""1. IMPORT LIBRARIES:

    - NumPy: For numerical operations
    - Pandas: For data manipulation and analysis
    - Matplotlib: For data visualization
    - Seaborn: For statistical data visualization
    - Scikit-learn: For machine learning algorithms and tools
    - Joblib: For saving and loading models
    - PyTorch: For building and training neural networks"""
# ============================================================

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


# ============================================================
""" 2. CREATE / INSERT DATA:

    - study_hours: Number of hours a student studies
    - attendance: Percentage of classes attended by the student
    - previous_score: The student's score in the previous exam
    - final_score: The student's score in the final exam (target variable)"""
# ============================================================

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


# Convert dictionary into DataFrame

df = pd.DataFrame(data)


print("\n==============================")
print("RAW DATA")
print("==============================")

print(df)


# ============================================================
"""3. UNDERSTAND DATA:
        Understanding the data is crucial for building an effective machine learning model.
        This involves exploring the dataset, identifying patterns, and gaining insights into the relationships between variables."""
# ============================================================

print("\n==============================")
print("DATA INFORMATION")
print("==============================")

print(df.info())

print("\nSTATISTICS:")
print(df.describe())


# ============================================================
"""4. CHECK MISSING VALUES:

        Missing values can lead to biased or inaccurate models.
        It's important to identify and handle missing values appropriately, either by removing them or imputing them with suitable values."""
# ============================================================

print("\n==============================")
print("MISSING VALUES")
print("==============================")

print(df.isnull().sum())


# If missing values existed, we could do:

# df["study_hours"] = df["study_hours"].fillna(
#     df["study_hours"].mean()
# )


# ============================================================
"""5. REMOVE DUPLICATES:

        Duplicate records can skew the results of a machine learning model.
        It's important to identify and remove duplicate entries to ensure the integrity of the dataset."""
# ============================================================

df = df.drop_duplicates()


# ============================================================
"""6. DATA VISUALIZATION:

        Visualizing the data helps in understanding the relationships between variables and identifying patterns or trends.
        Scatter plots, histograms, and heatmaps are common visualization techniques used in data analysis."""
# ============================================================

sns.scatterplot(
    data=df,
    x="study_hours",
    y="final_score"
)

plt.title("Study Hours vs Final Score")
plt.xlabel("Study Hours")
plt.ylabel("Final Score")

plt.show()


# ============================================================
""" 7. CORRELATION:
                    how strongly two features in a dataset are related to each other.

Correlation is a statistical measure that describes the degree to which two variables move in relation to each other.
It can be positive (both variables increase together), negative (one variable increases while the other decreases), or zero (no relationship).
Correlation is often quantified using a correlation coefficient, such as Pearson's correlation coefficient, which ranges from -1 to 1."""
# ============================================================

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


# ============================================================
"""8. SEPARATE FEATURES AND TARGET:
        In machine learning, we often separate the dataset into features (input variables) and the target (output variable) to train a model.
        Features are the independent variables that the model uses to make predictions, while the target is the dependent variable that we want to predict."""
# ============================================================

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


# ============================================================
"""9. TRAIN / TEST SPLIT:
        
        Splitting the dataset into training and testing sets is a common practice in machine learning.
        The training set is used to train the model, while the testing set is used to evaluate its performance on unseen data. 
        This helps in assessing the model's generalization ability and prevents overfitting"""
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\n==============================")
print("DATA SPLIT")
print("==============================")

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
"""10. FEATURE SCALING: 
        
        Feature scaling is a technique used to standardize the range of independent variables or features of data.
        It is important in machine learning to ensure that all features contribute equally to the model and to prevent features with larger scales from dominating those with smaller scales."""
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ============================================================
"""11. CLASSICAL MACHINE LEARNING
        USING SCIKIT-LEARN:
        Scikit-learn is a popular Python library for machine learning that provides simple and efficient tools for data analysis and modeling.
        It includes a wide range of algorithms for classification, regression, clustering, and more, along with utilities for model evaluation and selection."""
# ============================================================

print("\n==============================")
print("SCIKIT-LEARN MODEL")
print("==============================")


model = LinearRegression()

model.fit(
    X_train_scaled,
    y_train
)


# ============================================================
"""12. PREDICTION:

        After training a machine learning model, we can use it to make predictions on new or unseen data.
        The model takes input features and outputs predicted values based on the patterns it learned during training."""
# ============================================================

sklearn_predictions = model.predict(
    X_test_scaled
)


print("\nPredictions:")
print(sklearn_predictions)


# ============================================================
"""13. EVALUATION:

        Evaluating the performance of a machine learning model is crucial to understand how well it generalizes to new data.
        Common evaluation metrics for regression models include Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2).
        These metrics help in assessing the accuracy and reliability of the model's predictions."""
# ============================================================

mae = mean_absolute_error(
    y_test,
    sklearn_predictions
)

mse = mean_squared_error(
    y_test,
    sklearn_predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    sklearn_predictions
)


print("\n==============================")
print("SCIKIT-LEARN EVALUATION")
print("==============================")

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2  :", r2)


# ============================================================
"""14. SAVE SCIKIT-LEARN MODEL:
        
        Saving a trained machine learning model allows us to reuse it later without retraining.
        This is especially useful for deploying models in production environments or sharing them with others."""
# ============================================================

joblib.dump(
    model,
    "student_model.pkl"
)

joblib.dump(
    scaler,
    "student_scaler.pkl"
)

print("\nScikit-learn model saved.")


# ============================================================
"""15. LOAD MODEL:

        Loading a saved machine learning model allows us to use it for making predictions on new data without retraining.
        This is useful for deploying models in production or for further analysis."""
# ============================================================

loaded_model = joblib.load(
    "student_model.pkl"
)

loaded_scaler = joblib.load(
    "student_scaler.pkl"
)


# ============================================================
"""16. NEW STUDENT PREDICTION:

        After loading a saved machine learning model, we can use it to make predictions for new data points.
        In this case, we will predict the final score of a new student based on their study hours, attendance, and previous score using the loaded model and scaler."""
# ============================================================

new_student = pd.DataFrame(
    [[8, 90, 80]],
    columns=[
        "study_hours",
        "attendance",
        "previous_score"
    ]
)


new_student_scaled = loaded_scaler.transform(
    new_student
)


prediction = loaded_model.predict(
    new_student_scaled
)


print("\n==============================")
print("NEW STUDENT PREDICTION")
print("==============================")

print(
    "Predicted final score:",
    prediction[0]
)


# ============================================================
"""17. NOW MOVE TO DEEP LEARNING:
                 ||
                 ||
                 ||
                 ||
              \\    //
               \\  //
                \\//
                 \/
"""
# ============================================================
#
# Scikit-learn = Classical ML
#
# PyTorch = Neural Networks / Deep Learning
#
# ============================================================


# Convert Pandas/NumPy data into PyTorch tensors

X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train.values,
    dtype=torch.float32
).reshape(-1, 1)


X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test.values,
    dtype=torch.float32
).reshape(-1, 1)


# ============================================================
# 18. CREATE NEURAL NETWORK
# ============================================================

class StudentNeuralNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            # Input = 3 features
            nn.Linear(3, 32),

            # Activation function
            nn.ReLU(),

            # Hidden layer
            nn.Linear(32, 16),

            nn.ReLU(),

            # Output layer
            nn.Linear(16, 1)
        )


    def forward(self, x):

        return self.network(x)


# Create model

nn_model = StudentNeuralNetwork()


print("\n==============================")
print("PYTORCH MODEL")
print("==============================")

print(nn_model)


# ============================================================
""" 19. LOSS FUNCTION:

        The loss function is a crucial component in training neural networks.
        It measures the difference between the predicted output and the actual target values.
        The goal of training is to minimize this loss, allowing the model to make more accurate predictions.
"""
# ============================================================

loss_function = nn.MSELoss()


# ============================================================
""" 20. OPTIMIZER:

        The optimizer is responsible for updating the model's parameters during training.
        It uses the gradients computed by backpropagation to adjust the weights and biases of the neural network.
"""
# ============================================================

optimizer = optim.Adam(
    nn_model.parameters(),
    lr=0.001
)


# ============================================================
""" 21. TRAINING LOOP:

        The training loop is the core of the neural network training process.
        It involves multiple epochs where the model makes predictions, calculates the loss, performs backpropagation, and updates the weights using the optimizer.
        This iterative process continues until the model converges to an optimal set of parameters that minimize the loss function.
"""
# ============================================================

epochs = 1000


print("\n==============================")
print("PYTORCH TRAINING")
print("==============================")


for epoch in range(epochs):

    # -----------------------------
    # Forward Pass
    # -----------------------------

    predictions = nn_model(
        X_train_tensor
    )


    # -----------------------------
    # Calculate Loss
    # -----------------------------

    loss = loss_function(
        predictions,
        y_train_tensor
    )


    # -----------------------------
    # Clear old gradients
    # -----------------------------

    optimizer.zero_grad()


    # -----------------------------
    # Backpropagation
    # -----------------------------

    loss.backward()


    # -----------------------------
    # Update weights
    # -----------------------------

    optimizer.step()


    # Print progress

    if (epoch + 1) % 100 == 0:

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Loss: {loss.item():.4f}"
        )


# ============================================================
""" 22. PYTORCH TESTING:

        After training the neural network, we evaluate its performance on the test dataset.
        This involves making predictions on the test data and calculating evaluation metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2) to assess the model's accuracy and generalization ability.
"""
# ============================================================

nn_model.eval()


with torch.no_grad():

    test_predictions = nn_model(
        X_test_tensor
    )


# Convert tensor → NumPy

test_predictions_numpy = (
    test_predictions.numpy()
)


# ============================================================
""" 23. PYTORCH EVALUATION:
+
+        After testing the neural network, we evaluate its performance using various metrics.
+        These metrics provide insights into the model's accuracy and generalization ability.
+"""
# ============================================================

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


# ============================================================
"""24. SAVE PYTORCH MODEL:

        Saving a trained PyTorch model allows us to reuse it later without retraining.
        This is especially useful for deploying models in production environments or sharing them with others.
"""
# ============================================================

torch.save(
    nn_model.state_dict(),
    "student_neural_network.pth"
)

print("\nPyTorch model saved.")


# ============================================================
"""25. LOAD PYTORCH MODEL:
+
+        Loading a saved PyTorch model allows us to use it for making predictions or further training.
+        This is useful for deploying models in production environments or sharing them with others.
"""
# ============================================================

loaded_nn_model = StudentNeuralNetwork()

loaded_nn_model.load_state_dict(
    torch.load(
        "student_neural_network.pth",
        weights_only=True
    )
)

loaded_nn_model.eval()


# ============================================================
"""26. PREDICT NEW STUDENT WITH PYTORCH:
        
        After loading a saved PyTorch model, we can use it to make predictions for new data points.
        In this case, we will predict the final score of a new student based on their study hours, attendance, and previous score using the loaded model.
"""
# ============================================================

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


# ============================================================
"""27. FINAL PIPELINE:
+
+        The final pipeline integrates all the steps from data preprocessing to model evaluation.
+        It provides a complete workflow for training and deploying a PyTorch neural network.
+"""
# ============================================================

print("""

============================================================
COMPLETE PIPELINE
============================================================

RAW DATA
   ↓
Pandas
   ↓
Data Cleaning
   ↓
NumPy
   ↓
Matplotlib / Seaborn
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
StandardScaler
   ↓
Scikit-learn
   ↓
Classical ML
   ↓
Evaluation
   ↓
PyTorch
   ↓
Neural Network
   ↓
Loss Function
   ↓
Backpropagation
   ↓
Optimizer
   ↓
Training
   ↓
Evaluation
   ↓
Save Model
   ↓
Load Model
   ↓
Prediction

============================================================
NEXT LEVEL
============================================================

PyTorch
   ↓
Deep Learning
   ↓
Transformers
   ↓
Hugging Face
   ↓
LLMs
   ↓
SFT
   ↓
LoRA
   ↓
QLoRA
   ↓
Fine-Tuning
   ↓
RAG
   ↓
Agents
   ↓
Multi-Agent AI

============================================================
""")
