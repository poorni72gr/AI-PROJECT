What is AI?

AI (Artificial Intelligence) means making a computer think and learn like humans.

Examples:

Chatbots

Face recognition

Recommendation systems

Spam detection

Python is the most popular language for AI because it is easy and has many libraries.

AI vs ML vs DL (Simple)

AI → Big concept (machines acting smart)

Machine Learning (ML) → Machine learns from data

Deep Learning (DL) → Uses neural networks

Python Libraries for AI

Common libraries:

numpy → math operations

pandas → data handling

scikit-learn → machine learning

matplotlib → graphs

tensorflow / pytorch → deep learning

Example 1: Simple AI Logic (No Library)
age = 20

if age >= 18:
    print("Eligible to vote")
else:
    print("Not eligible")


👉 This is basic AI logic (rule-based system).

Example 2: Machine Learning (Prediction Example)
Step 1: Import libraries
from sklearn.linear_model import LinearRegression
import numpy as np

Step 2: Training data
# Study hours
X = np.array([[1], [2], [3], [4], [5]])

# Marks
y = np.array([35, 40, 50, 60, 70])

Step 3: Train the model
model = LinearRegression()
model.fit(X, y)

Step 4: Prediction
prediction = model.predict([[6]])
print("Predicted marks:", prediction[0])


👉 The model learns from data and predicts marks.

Example 3: AI Chatbot (Very Simple)
while True:
    msg = input("You: ").lower()

    if msg == "hi":
        print("Bot: Hello!")
    elif msg == "bye":
        print("Bot: Goodbye!")
        break
    else:
        print("Bot: I don't understand")
