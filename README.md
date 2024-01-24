# Customer-Churn-prediction
## Overview
This project focuses on predicting customer churn – a critical metric for many businesses, especially in the telecom, finance, and e-commerce sectors. Using historical bank customer data, the project aims to identify customers at high risk of churning (i.e., ceasing to be a customer) so that bank can take proactive steps to retain them.

## Project Objective
The primary objective of this project is to develop a machine learning model that can accurately predict the likelihood of a customer churning. By leveraging this predictive capability, businesses can implement targeted retention strategies to reduce churn and increase customer loyalty.

## Dataset Description
The dataset used for this project is from kaggle and it includes various customer attributes such as:
- Customer demographic information (age, gender, etc.)
- Account information (tenure, salary type, etc.)
- Customer location etc.

## Methodology
The project employs the following methodology:
1. **Data Preprocessing**: Cleaning data, handling missing values, and encoding categorical variables.
2. **Exploratory Data Analysis (EDA)**: Analyzing the dataset to uncover trends and patterns that influence churn.
3. **Feature Engineering**: Creating new features that could improve model performance.
4. **Model Selection**: Experimenting with various machine learning models such as Logistic Regression, Random Forest and Decision Tree.
5. **Model Evaluation**: Evaluating models based on metrics like Accuracy, Precision, Recall.
   
## Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn for visualizations
- Scikit-learn for machine learning
- Jupyter Notebook

## Results
The Exploratory data analysis results are:
1. Approximately 20.37% of the customers in the bank have exited.
2. Germany has the highest churn rate.
3. Customers who have only one product from the bank are more likely to leave.
4. Those who are not a active member in the bank and who have the credit card are more likely to leave the bank.
5. Females has the more churn rate then males.
6. Customers with Tenure of 2 , 7 , 8 years have the lowest churn rate.

## Model Evaluation and Selection for Customer Churn Prediction
In our quest to develop a robust predictive model for customer churn, we explored various machine learning algorithms, including Logistic Regression, Decision Tree, and Random Forest. After a comprehensive analysis, it became evident that the **Random Forest model** outperformed its counterparts, achieving an impressive accuracy of 86.80%.<br>
Given its high accuracy and reliability, we have decided to proceed with the Random Forest model for predicting customer churn. This model not only offers us a high degree of confidence in its predictions but also provides valuable insights into the factors influencing customer behavior. By leveraging this model, we can more effectively identify customers at risk of churn, enabling us to implement targeted interventions to enhance customer retention and foster long-term loyalty.
   
