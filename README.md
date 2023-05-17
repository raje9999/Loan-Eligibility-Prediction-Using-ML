# Loan-Eligibility-Prediction-Using-ML
This project aims to predict the eligibility of individuals for a loan based on certain features and historical loan data. By utilizing machine learning algorithms, the project provides a prediction model that can assist in making loan eligibility decisions.

# Table of Contents

1.1 Project Overview
1.2 Dataset
1.3 Dependencies
1.4 Installation
1.5 Usage
1.6 Models
1.7 Evaluation

# 1.1 Project Overview
The Loan Eligibility Prediction project uses supervised machine learning algorithms to predict loan eligibility based on various features such as income, age, employment status, credit history, and more. The goal is to provide a reliable model that can help lenders assess the risk associated with granting loans to individuals.
The project involves data preprocessing, exploratory data analysis, feature engineering, model training, and evaluation. Multiple machine learning models are implemented and compared to select the best-performing model for loan eligibility prediction.

# 1.2 Dataset
The dataset used for this project contains historical loan data with corresponding features and loan eligibility labels. It includes the following information:
1.2.1 Income: The applicant's income (continuous value).
1.2.2 Age: The age of the applicant (continuous value).
1.2.3 Gender: The gender of the applicant.
1.2.4 Marriage status: The status of their marriage of the applicant.
1.2.5 Employment Status: The employment status of the applicant (categorical value).
1.2.6 Credit History: The credit history of the applicant (categorical value).
1.2.7 Loan Amount: The loan amount requested by the applicant (continuous value).
1.2.8 Loan Term: The term or duration of the loan (continuous value).
1.2.9Loan Eligibility: The target variable indicating loan eligibility (binary value: Yes/No).
The dataset is split into training and testing sets to train and evaluate the machine learning models.
# Dataset Link
# https://www.kaggle.com/datasets/burak3ergun/loan-data-set

# 1.3 Dependencies
The following dependencies are required to run the project:
1.3.1 Python 
1.3.2 Pandas
1.3.3 NumPy
1.3.4 Scikit-learn
1.3.5 Matplotlib
Install the necessary dependencies by following the instructions in the installation section.

# 1.4 Installation

1.4.1 Clone the repository:

# "git clone https://github.com/raje9999/Loan-Eligibility-Prediction-Using-ML"

1.4.2. Navigate to the project directory:

# cd loan-eligibility-prediction-using-ml

1.4.3. Install the Required Dependencies

# pip install -r requirements.txt

# 1.5 Usage

1. Place your loan dataset file ('loan-data-set.csv') in the project directory.

2. Modify the 'config.py' file to adjust any necessary settings (e.g., file paths, model parameters).

3. Run the 'loan-eligibility-prediction-using-ml'.py script:

# "python loan_eligibility_prediction-using-ml.py"

4. The script will preprocess the data, train the machine learning models, and generate predictions for loan eligibility.

5. The results will be displayed on the console and saved in the specified output file.

# 1.6 Models

The project utilizes the following machine learning models for loan eligibility prediction:

1.6.1 Logistic Regression
1.6.2 Random Forest Classifier
1.6.3 Support Vector Machines (SVM)
1.6.4 Gradient Boosting Classifier
1.6.5 K-Nearest Neighbour (KNN)
1.6.6 Naive Bayes
1.6.7 Decision Tree

Each model is trained and evaluated to determine its performance and select the best model for loan eligibility prediction.

# 1.7 Evaluation
The performance of the machine learning models is evaluated using various metrics such as accuracy, precision, recall, and F1-score. Additionally, cross-validation techniques might be employed to assess the models' robustness and generalization ability.
The evaluation results and metrics are displayed in the console and saved in the output file.
