# Team-SC1-_9
# 🏥 Hospital Readmission Risk Prediction using KNN

## 📌 Problem Statement

Hospital readmissions can often be reduced by analyzing patient history and test results. In this project, we aim to build a machine learning model that predicts whether a patient is likely to be readmitted to the hospital based on various features like age, test results, length of stay, and past admission frequency.

---

## 🎯 Objective

- To predict if a patient will be **readmitted (1)** or **not readmitted (0)**.
- Help hospitals identify at-risk patients using historical data and clinical factors.
- Use **K-Nearest Neighbors (KNN)** for binary classification.

---

## 📁 Dataset Information

**Dataset Name**: `healthcare_dataset.csv`  
**Source**: [Kaggle Healthcare Dataset](https://www.kaggle.com/datasets/prasad22/healthcare-dataset)  
**Simulated Field**: `Readmission` (created based on frequency of admission)

### 🔑 Key Features Used:

| Feature                     | Description                                     |
|----------------------------|-------------------------------------------------|
| `Age`                      | Age of the patient                              |
| `Test Results`             | Normal, Abnormal, Inconclusive                  |
| `Length of Stay`           | Number of days admitted                         |
| `Frequency of Admissions`  | Number of times the patient was admitted        |
| `Readmission`              | Target column: 1 = Readmitted, 0 = Not          |

> ⚠️ Many other columns were dropped because they were irrelevant or not useful for model learning (like Name, Doctor, Insurance Provider, etc.)

---

## 🧹 Preprocessing Steps

1. Loaded data using `pandas`
2. Mapped `Test Results` to ordinal values:
   - Normal = 0, Abnormal = 1, Inconclusive = 2
3. Converted `Date of Admission` and `Discharge Date` to datetime.
4. Calculated `Length of Stay` from date differences.
5. Lowercased patient names to avoid mismatch in counting.
6. Calculated `Frequency of Times Admitted` using `.value_counts()`.
7. Created the **target column** `Readmission`:
   - If a patient appears more than once in the dataset, they are marked as readmitted (1), else not (0).

---

## 🤖 Model Used

- **Algorithm**: `KNeighborsClassifier` (from `sklearn`)
- **Train-Test Split**: 80-20
- **Input Features**:
  - Age
  - Test_Results_Encoded
  - Length of Stay
  - Frequency of Times Admitted

---

## 🧪 Training & Prediction

- Model trained on user-engineered features.
- Accuracy evaluated using `accuracy_score` on test set.
- The model supports **manual input prediction** using:

```python
Age = int(input("Enter the age: "))
Test_result = int(input("Enter the test result (0-Normal, 1-Abnormal, 2-Inconclusive): "))
Length_of_stay = int(input("Enter the length of stay: "))
Frequency_of_times_admitted = int(input("Enter the frequency of times admitted: "))
result = knn.predict([[Age, Test_result, Length_of_stay, Frequency_of_times_admitted]])
