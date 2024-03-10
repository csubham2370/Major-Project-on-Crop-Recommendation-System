# Major-Project-on-Crop-Recommendation-System
This is our MCA Major Project

## Here we explain our Crop Recommendation System.ipynb file.
![image](https://github.com/csubham2370/Major-Project-on-Crop-Recommendation-System/assets/144363196/4188539c-1623-4928-b1ed-384682c2d06c)
Certainly! Let's break down each line of your code:

1. `from __future__ import print_function`: This line is a directive to use the print function from Python 3.x even in Python 2.x environments. In Python 2.x, print was a statement rather than a function, so this import allows you to use the Python 3.x syntax for print.

2. `import pandas as pd`: This line imports the pandas library and assigns it the alias `pd`. Pandas is a powerful library for data manipulation and analysis, offering data structures and functions to work with structured data easily.

3. `import numpy as np`: This line imports the NumPy library and assigns it the alias `np`. NumPy is a fundamental package for numerical computing with Python, providing support for arrays, matrices, and mathematical functions.

4. `import matplotlib.pyplot as plt`: This line imports the pyplot module from the Matplotlib library and assigns it the alias `plt`. Matplotlib is a plotting library for Python, and pyplot provides a MATLAB-like interface for creating a variety of plots and visualizations.

5. `import seaborn as sns`: This line imports the seaborn library and assigns it the alias `sns`. Seaborn is a statistical data visualization library based on Matplotlib, offering a higher-level interface for creating attractive and informative statistical graphics.

6. `from sklearn.metrics import classification_report`: This line imports the `classification_report` function from the `metrics` module within scikit-learn. Scikit-learn is a popular machine learning library in Python, and `classification_report` is used to generate a text report showing the main classification metrics.

7. `from sklearn import metrics`: This line imports the `metrics` module from scikit-learn. This module provides various functions to evaluate the performance of machine learning models, such as accuracy, precision, recall, etc.

8. `from sklearn import tree`: This line imports the `tree` module from scikit-learn. This module provides functionality related to decision trees, including building, visualizing, and analyzing decision tree models.

9. `import warnings`: This line imports the warnings module, which is a built-in Python module used to handle warnings. Warnings are messages that indicate potential issues with your code, and this module provides functions to control how warnings are displayed or ignored.

10. `warnings.filterwarnings('ignore')`: This line sets up a filter to ignore warnings during the execution of your code. It suppresses the display of warning messages, which can be useful when you want to avoid distraction from non-critical warnings.

By combining these imports, you're setting up your Python environment for data analysis, visualization, and machine learning tasks while also configuring it to handle warnings in a specific way.

# Upload a file on Google Colab
![image](https://github.com/csubham2370/Major-Project-on-Crop-Recommendation-System/assets/144363196/b3523f34-327c-4b0b-8729-b1a130840f83)
* This code allows the user to upload a file from their local system to a Google Colab notebook environment, and the uploaded file(s) information is stored in the uploaded variable for further processing within the notebook.
# Importing the Data
![image](https://github.com/csubham2370/Major-Project-on-Crop-Recommendation-System/assets/144363196/7f5f823a-cd39-4139-a19a-097e5517053e)
* This code reads the contents of a CSV file named 'modified_crop.csv' using Pandas and stores it in a DataFrame named crop for further processing and analysis in Python.

# crop.shape
![image](https://github.com/csubham2370/Major-Project-on-Crop-Recommendation-System/assets/144363196/0bd8208b-21e1-486c-8c4d-21b1464ff73a)
* This code is use for quickly check the size of your DataFrame, which is especially useful when dealing with large datasets or when you need to understand the structure of the data you're working with.

* For example, if you run crop.shape and it returns (100, 5), it means that the DataFrame crop has 100 rows and 5 columns. and the number of elements is 500.

# crop.info()
![image](https://github.com/csubham2370/Major-Project-on-Crop-Recommendation-System/assets/144363196/52b97113-1fe4-4168-bcb2-e7361e13b02a)
* By using crop.info(), you can quickly get an overview of the DataFrame, including its size, data types, and missing values, which is useful for initial data exploration and understanding the dataset's characteristics.

# crop.head()
![image](https://github.com/csubham2370/Major-Project-on-Crop-Recommendation-System/assets/144363196/588ddb9e-8bcb-4618-bc59-a8671b016a3d)

* The purpose of using head() is to quickly inspect the structure and content of the DataFrame. It's often used as an initial step in data analysis to get a sense of what the data looks like before performing further operations. By examining the first few rows, you can check the column names, data types, and example values in the DataFrame.

# tail()
![image](https://github.com/csubham2370/Major-Project-on-Crop-Recommendation-System/assets/144363196/c3d3f055-ccb4-44aa-aa0b-61fbd5e01d39)

* The purpose of using tail() is to quickly inspect the end of the DataFrame. It's often used to check for patterns or trends in the data, especially if the data is ordered chronologically or by some other criteria. By examining the last few rows, you can see the most recent data entries and verify that the DataFrame has been properly loaded or processed.

# crop.isnull().sum()
![image](https://github.com/csubham2370/Major-Project-on-Crop-Recommendation-System/assets/144363196/7075e847-0338-4d9f-8e34-be2530f4c7f7)

* crop.isnull().sum() gives you a Series where each entry represents the number of missing values in the corresponding column of the crop DataFrame. This information is useful for identifying and handling missing data in your dataset.

# crop.isnull().sum().sum()
![image](https://github.com/csubham2370/Major-Project-on-Crop-Recommendation-System/assets/144363196/26ea78e0-fb66-4b1a-ae1c-3dd2aefecd17)

* crop.isnull().sum().sum() gives you the total number of missing values in the entire DataFrame crop, summing up the counts of missing values across all columns. This information is valuable for understanding the extent of missing data in your dataset.

#interpolation
![image](https://github.com/csubham2370/Major-Project-on-Crop-Recommendation-System/assets/144363196/738235c2-2a84-43a1-b348-048596c9ac87)

* this code snippet ensures that missing values in the DataFrame crop are handled by interpolating them (linear interpolation) and by replacing missing values in the 'label' column with the most frequent label value. The resulting DataFrame crop1 has missing values appropriately handled.

# Here we recheck if any null value is present or not.
![image](https://github.com/csubham2370/Major-Project-on-Crop-Recommendation-System/assets/144363196/c1e4c87b-f002-4a07-97e3-679a44b3d097)

# crop2 = crop1.drop_duplicates()
![image](https://github.com/csubham2370/Major-Project-on-Crop-Recommendation-System/assets/144363196/49cab2fd-dca6-47c2-8edb-bfc065016b30)

* the DataFrame crop2 contains the data from crop1 with duplicate rows removed. This operation ensures that each row in crop2 is unique, based on all columns by default.

# Here we recheck if any duplicate rows are present or not.
![image](https://github.com/csubham2370/Major-Project-on-Crop-Recommendation-System/assets/144363196/a56f6dfe-12b7-49b7-9519-8fc84559f8ad)

# crop2.isnull().any()
![image](https://github.com/csubham2370/Major-Project-on-Crop-Recommendation-System/assets/144363196/a9a49b5f-d964-4c0f-bdce-9a5b1cc1fdc8)

* returns a Series where each entry represents whether there are any missing values in the corresponding column of the DataFrame crop2. If the entry is True, it means that the column contains at least one missing value; otherwise, it's False.

# crop2.describe()
![image](https://github.com/csubham2370/Major-Project-on-Crop-Recommendation-System/assets/144363196/f2b7a1bd-7da2-4eb8-a5cc-0d379e8ec6bc)

* The output of crop2.describe() will be a DataFrame where each row represents a summary statistic, and each column represents a numerical column in the original DataFrame crop2. This summary statistics can provide insights into the distribution and spread of values in the dataset, helping with data exploration and analysis.

# corr()
![image](https://github.com/csubham2370/Major-Project-on-Crop-Recommendation-System/assets/144363196/75552c58-9db2-4cd9-9d57-5b89ded18e54)

* The corr() method is commonly used to identify relationships between variables in a dataset. High positive or negative correlation coefficients can indicate strong relationships between variables, while a correlation coefficient close to 0 suggests little to no relationship. These correlation coefficients can be further analyzed and interpreted to gain insights into the dataset.

# size
![image](https://github.com/csubham2370/Major-Project-on-Crop-Recommendation-System/assets/144363196/8a569038-5873-47b2-b64b-813e8a01508c)

* The .size method in Python is used to get the number of elements in an object, such as a list, tuple, set, or dictionary. However, it seems you've added parentheses to the method, which is not correct for this purpose.

# columns
![image](https://github.com/csubham2370/Major-Project-on-Crop-Recommendation-System/assets/144363196/d3f031a2-cc80-496a-99cd-b76bb819d51a)

*The .columns attribute is used to retrieve the column labels of a DataFrame in pandas.

# crop2['label'].unique() 
![image](https://github.com/csubham2370/Major-Project-on-Crop-Recommendation-System/assets/144363196/0618382b-2179-4e0b-9575-5f1764d5b629)

* The code crop2['label'].unique() is used to get the unique values in the 'label' column of the DataFrame crop2. Here's a breakdown:

* crop2['label']: This selects the 'label' column from the DataFrame crop2.
.unique(): This is a method that returns an array of unique values from the selected column.

# dtypes
![image](https://github.com/csubham2370/Major-Project-on-Crop-Recommendation-System/assets/144363196/5d74a906-70cb-48a0-8ed1-8259212dab0f)

* The dtypes attribute in pandas DataFrame is used to get the data types of each column in the DataFrame.

# crop2['label'].value_counts()
![image](https://github.com/csubham2370/Major-Project-on-Crop-Recommendation-System/assets/144363196/97cbce82-456f-44b4-a492-7741ea9eb41e)

* The output of crop2['label'].value_counts() will be a Series where each unique label in the 'label' column of crop2 is listed along with the count of occurrences of that label in the dataset. This information is useful for understanding the distribution of different labels in the dataset and can be valuable for various analytical purposes.

# Visualizing the correlation matrix
![image](https://github.com/csubham2370/Major-Project-on-Crop-Recommendation-System/assets/144363196/27b9b329-d057-4d41-b571-bbaeae56380b)

* By visualizing the correlation matrix as a heatmap, you can easily identify patterns and relationships between variables in the dataset. Positive correlations will appear in warm colors, negative correlations in cool colors, and no correlation in neutral colors. The annotations provide the exact correlation coefficients, making it easier to interpret the heatmap.

# Distribution of values in the 'N' column of the DataFrame crop2.
![image](https://github.com/csubham2370/Major-Project-on-Crop-Recommendation-System/assets/144363196/06bf8c44-3c3c-445c-b59a-9ad1ee48879d)

* By executing this code, you'll generate a distribution plot showing the distribution of values in the 'N' column of the DataFrame crop2. This visualization helps in understanding the distribution of values and identifying any patterns or outliers present in the data.

# Distribution of values in the 'P' column of the DataFrame crop2.
![image](https://github.com/csubham2370/Major-Project-on-Crop-Recommendation-System/assets/144363196/1e4bf1ed-2d2e-43e3-8ed3-8dc2a3c854e6)


* By executing this code, you'll generate a distribution plot showing the distribution of values in the 'P' column of the DataFrame crop2. This visualization helps in understanding the distribution of values and identifying any patterns or outliers present in the data.
# Distribution plot of K, Temperature, Humidity, ph, Rainfall using Seaborn's distplot() function.
![image](https://github.com/csubham2370/Major-Project-on-Crop-Recommendation-System/assets/144363196/9436bc57-3474-44df-9693-4321a1f0b0ca)
![image](https://github.com/csubham2370/Major-Project-on-Crop-Recommendation-System/assets/144363196/eba97eae-ac3d-48b5-9708-dbc79af69652)
![image](https://github.com/csubham2370/Major-Project-on-Crop-Recommendation-System/assets/144363196/38b48777-8cda-49d4-8e45-f85eaa0b7eec)
![image](https://github.com/csubham2370/Major-Project-on-Crop-Recommendation-System/assets/144363196/991ca40b-662c-4ac9-9769-61079c8e1bc4)
