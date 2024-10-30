# Handle unbalanced Datasets

Cristiane Mecca Giacomazzi (Data Analyst)
The following topics will be presented:
1.Problem Understanding
2.Data Understanding
3.Ethical Statements
4.Dictionary
5.Libraries used for this project
6.Project plan
7.Data Preprocessing
8.Handle unbalanced dataset
9.Interpretation
10.Conclusion

# 1. Problem Understanding

Consider a dataset where one label is more frequent than the other; in this case, the dataset is imbalanced. The predominant label in an imbalanced dataset is called the majority class, while the less frequent label is known as the minority class. For instance, in many real-world medical datasets, the number of patients with heart disease is significantly smaller than those without heart disease. Thus, heart disease represents a minority class in an imbalanced dataset.
In the dataset used for this project, class 0 is the majority class (indicating no churn). This is expected because the focus is on churn, making churn the minority class. The performance of machine learning algorithms can be significantly affected by an imbalanced dataset. When trained on such a dataset, the model tends to learn mostly from the majority class, resulting in a biased model. Consequently, the cost of false negatives (failing to detect the minority class) becomes much higher than false positives (incorrectly identifying a sample as belonging to the minority class), compromising metrics such as accuracy.
To address this issue, several techniques can be applied to manage imbalanced datasets. Methods such as Oversampling, Undersampling, and Synthetic Minority Oversampling Technique (SMOTE) can be employed. The objective of Part 2 of this project is to handle imbalanced datasets using a public bank dataset and to select the best approach for inclusion in my portfolio project.

# 2. Data Understanding

Data Source: This project used a dataset from the Kaggle Website called “Churn_modelling” (https://www.kaggle.com/datasets/shubh0799/churn-modelling/data), a structured dataset.
Tool: Python, using Google Collab.

# 3. Ethical Statements

Ethical issues: This project complies with the TCPS 2. The dataset is in the public domain.
Funding and conflict of interest: non-funded research, no conflict of interesting.

# 4. Dictionary

<img width="578" alt="Screen Shot 2024-10-30 at 9 44 50 AM" src="https://github.com/user-attachments/assets/cb72f44f-fcd6-44fb-b8b1-53463ba05396">

# 5. Libraries used to project

- import pandas as pd
- import matplotlib.pyplot as plt
- import seaborn as sns
- from imblearn.over_sampling import RandomOverSampler
- from imblearn.under_sampling import RandomUnderSampler 
- from imblearn.over_sampling import SMOTE
- from sklearn.linear_model import LogisticRegression
- from collections import Counter
- from sklearn.datasets import make_classification
- from sklearn.model_selection import train_test_split
- from sklearn.metrics import accuracy_score

# 6. Project plan

<img width="690" alt="Screen Shot 2024-10-30 at 9 46 15 AM" src="https://github.com/user-attachments/assets/6383703e-b7bc-4e85-af90-43f9f09fa33f">

# 7. Data Preprocessing

The dataset contains no missing or duplicated data. Furthermore, the columns "surnames," "row number," and "customer ID" have been removed. The project involved managing the unbalanced dataset and evaluating the metrics of various techniques to determine the most effective approach.

# 8. Handle unbalanced dataset
It was used 3 techniques to balance the dataset. After that, the metrics of each technique was measured. 

<img width="385" alt="Screen Shot 2024-10-30 at 9 47 58 AM" src="https://github.com/user-attachments/assets/7874c834-7872-4f46-9fb3-91b7a9fb659c">

# 9. Interpretation

The model trained with oversampling has the highest training accuracy among the three methods. However, it shows a tiny reduction in accuracy on the test set, indicating that it might be slightly overfitting the training data. 
The undersampling model has slightly lower accuracy on the training set compared to oversampling, but its test accuracy is barely higher. This suggests that undersampling may generalize better to new data, possibly because it avoids the risk of overfitting by reducing the training set size and focusing on fewer representative samples.
The model trained with SMOTE has the same training and test accuracy as the undersampling approach. This indicates that SMOTE provides balanced data similar to undersampling in terms of performance, but with synthetic samples that might better represent variations within the minority class.

# 10. Conclusion
Based on the metrics, SMOTE generated synthetic samples that might better represent variations within the minority class and it will be used for handle the dataset.


## References
Azank, F. Dados Desbalanceados — O que são e como lidar com eles. (2020). Medium. https://medium.com/turing-talks/dados-desbalanceados-o-que-s%C3%A3o-e-como-evit%C3%A1-los-43df4f49732b 

Faritha Banu, J., Neelakandan, S., Geetha, B. T., Selvalakshmi, V., Umadevi, A., & Martinson, E. O. (2022). Artificial Intelligence Based Customer Churn Prediction Model for Business Markets. Computational intelligence and neuroscience, 2022, 1703696. https://doi.org/10.1155/2022/1703696 

Wilde. A. (2023). How to approach customer churn measurement in banking. UXPRESSIA. https://uxpressia.com/blog/how-to-approach-customer-churn-measurement-in-banking#Five_steps_to_stop_customers_from_leaving 

Michael, J. (2022). Bank Customer Churn Prediction Using Machine Learning. Analytics Vidhya. 
https://www.analyticsvidhya.com/blog/2022/09/bank-customer-churn-prediction-using-machine-learning/ 

Trotta, F.; Ackerson, D. (2024). How to Handle Imbalanced Data for Machine Learning in Python.  https://semaphoreci.com/blog/imbalanced-data-machine-learning-python 







