import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the dataset
breast_cancer = pd.read_csv("breast_cancer.csv")

# Data preprocessing
breast_cancer.drop(columns="Unnamed: 32", axis=1, inplace=True)
label_encoder = LabelEncoder()
breast_cancer["target"] = label_encoder.fit_transform(breast_cancer["diagnosis"])
breast_cancer.drop(columns="diagnosis", axis=1, inplace=True)

# Exploratory data analysis
print(breast_cancer.head())
print(breast_cancer.describe())
print(breast_cancer["target"].value_counts())
print(breast_cancer.groupby("target").mean())

# Data visualization
sns.set()
sns.countplot(x="target", data=breast_cancer)
# plt.show()

for column in breast_cancer:
    sns.displot(data=breast_cancer, x=column)
    # plt.show()

# Model training and evaluation
x = breast_cancer.drop(columns="target", axis=1)
y = breast_cancer["target"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

logistic_model = LogisticRegression()
logistic_model.fit(x_train, y_train)

# Model evaluation
x_train_prediction = logistic_model.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)
# print("Accuracy on training data = ", training_data_accuracy)

x_test_prediction = logistic_model.predict(x_test)
test_data_accuracy = accuracy_score(y_test, x_test_prediction)
# print("Accuracy on test data = ", test_data_accuracy)

# Save the model
import pickle

filename = "breast_cancer_model.pkl"
pickle.dump(logistic_model, open(filename, "wb"))

# Load the model and make predictions
loaded_model = pickle.load(open("breast_cancer_model.sav", "rb"))
input_data = [
    842302,
    17.99,
    10.38,
    122.8,
    1001,
    0.1184,
    0.2776,
    0.3001,
    0.1471,
    0.2419,
    0.07871,
    1.095,
    0.9053,
    8.589,
    153.4,
    0.006399,
    0.04904,
    0.05373,
    0.01587,
    0.03003,
    0.006193,
    25.38,
    17.33,
    184.6,
    2019,
    0.1622,
    0.6656,
    0.7119,
    0.2654,
    0.4601,
    0.1189,
]
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = loaded_model.predict(input_data_reshaped)
if prediction[0] == 0:
    print("The Breast cancer is Benign.")
else:
    print("The Breast Cancer is Malignant.")
