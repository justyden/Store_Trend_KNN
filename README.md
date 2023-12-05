## Store Trend Prediction
*A data mining related application.*

### Team Members:
- Tyler Thompson
  - Email: tylert123@yahoo.com
- Xiang Liu
  - Email: 1784676846.xl@gmail.com  
- [Team Member 3]

### Introduction

#### Problem Statement
In all industries, it is important to understand the target consumers to maximize profits and sales. Consumer data is crucial for analyzing and identifying trends. Consumers exhibit certain tendencies based on their location, and companies need to comprehend which products they are inclined to purchase. Additionally, understanding the profitability of each purchase is essential to identify the most lucrative sales. This application utilizes multiple algorithms to achieve this. It uses KNN classification to pinpoint targeted cities and predict profits within the dataset. In addition to this, it uses random forest to determine the profit of each sale.

#### Objective
This application aims to predict the likelihood of a city purchasing a product from an office store, such as furniture and work supplies. It further classifies the profits associated with each purchase and identifies the specific products contributing to those profits. By discerning which products are frequently purchased in specific cities and understanding the profit margins, companies can tailor their sales strategies more effectively. The application is developed using the Anaconda distribution for Python, which incorporates various machine learning packages.

#### Motivation
This project is motivated by the need to simulate a real-world data mining application that involves collaborative teamwork and extensive knowledge within the field. Understanding consumer data is crucial for companies seeking insights into consumer behavior. The application provides valuable information on what products a company can expect its consumers to purchase and the associated profits. Furthermore, gaining a deep understanding of the algorithms, processes, and techniques used in this application is essential knowledge for the developers.

#### Related Work
Several related topics align with this application, including retail analytics, market basket analysis, geospatial analysis, customer segmentation, predictive modeling, and collaborative filtering. The project draws inspiration from techniques in retail analytics and predictive modeling. Similar projects involve determining average sales per order, identifying valuable consumers, and optimizing product orders based on location.

### Data

#### Data Source and Format
The dataset is sourced from [Sample Super Store](https://community.tableau.com/s/question/0D54T00000CWeX8SAL/sample-superstore-sales-excelxls) and is presented in an Excel format (.xls). This dataset comprises 9,994 purchases from various cities in the United States and Canada. Key features include category, product name, sales, quantity, discount, and profit. The primary label for the application is the city. The dataset incorporates numerical, categorical, and ordinal features, with a focus on numerical and categorical features for efficient model training. The sample data, consisting of around 10,000 entries, allows for robust experimentation.

#### Data Example
Below is an excerpt from the dataset before the preprocessing steps:
![Example Data](images/example_data.png)

### Methodology

#### Schematic Diagram/Framework
The application's structure and processes are depicted in the following schematic diagram:
![Schematic Diagram](images/schematic_diagram_format.png)

#### Data Visualization and Preprocessing
Data preprocessing involved several steps to prepare the dataset for model training. Firstly, the.xls file was converted to.xlsx to meet the updated format requirements of Pandas. Normalization was then performed using the minimum-maximum and Z_score normalization techniques. This involved scaling specific columns, such as sales, quantity, discount, and profit, to a range between 0 and 1, ensuring uniformity for effective model training. At first, we used the minimum-maximum normalization technique, but the scaling didn't come out correctly. For example, $-300 of profit and $20 of profit all have the same scaling value, which doesn't seem correct. Then we normalize the data using the Z-Score technique. The scaling looks correct when the profit is high, showing a positive value that is above 0, which means it is above the average profit, and when the profit is super low, it shows a negative value below 0, which means it is below the average profit or even negative. So we decided to use the Z-Score technique.

##### Normalization Technique
![Normalization Technique](images/normalization_technique.png)
#### Z-Score Technique
![image](https://github.com/justyden/Store_Trend_KNN/assets/117769320/48819b42-ccc4-4144-b418-80751568b4ca)

#### Procedures and Features
The methodology employed in this project encompasses several key procedures and features. The initial step involves exploratory data analysis (EDA) to gain insights into the distribution and relationships within the dataset. Following this, feature selection is conducted to identify the most influential variables for model training. Features such as city, category, sub-category, sales, quantity, discount, and profits are crucial for predicting consumer behavior and profitability.

The algorithm applied was the K-Nearest Neighbors (KNN) classification algorithm, Random Forest Regression, and Random Forest Classification

The original algorithm utilized is the K-Nearest Neighbors (KNN) classification algorithm. KNN identifies patterns based on the similarity of instances, making it suitable for predicting city preferences and associated profits and sub-categories. Additionally, feature scaling techniques are applied to ensure that no single feature dominates the model training process. But the result and the accuracy didn't come out great; the best accuracy we can get is 40% even with the parameter tuning. Then tested with the applied Random Forest Classification algorithm with the same features (profits and sub-categories) and target (city), the accuracy only increased by about 10%, which still didn't meet expectations.

Second, we decided to change our features and target to see if we could get better accuracy and training scores as well. The feature we focused on was subcategory, category, sales, quantity, and target profit using the random forest regression algorithm. The result is still not good because the profit is a continuous value, regression doesn’t perform well at around 50% accuracy, and the training score is 83%. Then we categorized the profit into high, low, and negative for-profit and used the same feature and a random forest classifier model to predict the result, which came out so much better for profit. In the categorization, if the value is greater than $200, the profit is set to be high, under $200-$0, and negative if the value is less than $0. The accuracy was able to get up to 87%, and the training score was 89%. We discussed the result with the team members, and we applied the "discount" column to our features as well. The result is surprisingly great; the accuracy went up to 95% and the training score went up to 99.6%



### Experiments

#### Data Division (Training/Testing)
To assess the model's performance accurately, the dataset is divided into training and testing sets. Approximately 80% of the data is allocated for training, allowing the model to learn patterns, while the remaining 20% is reserved for testing to evaluate its predictive capabilities. Stratified sampling is implemented to maintain the distribution of cities across both sets, ensuring representative training and testing subsets.

#### Parameter Tuning
Parameter tuning is a critical aspect of optimizing the KNN model. The selection of the optimal number of neighbors (K) is crucial for the model's accuracy. A systematic approach, such as cross-validation, is employed to iterate through various K values and identify the configuration that yields the best results.

#### Evaluation Metrics
The performance of the model is evaluated using several metrics, including accuracy, precision, recall, and F1-score. Accuracy provides an overall measure of the model's correctness, while precision and recall offer insights into the model's ability to predict positive instances correctly and capture all positive instances, respectively. The F1 score combines precision and recall, providing a balanced assessment of the model's performance.

#### Results (Tables and Graphs)
The results of the experiments are presented in the form of tables and graphs. A confusion matrix is generated to visualize the model's performance in predicting city preferences and associated profits. Additionally, graphical representations, such as ROC curves, provide insights into the trade-off between true positive and false positive rates.

#### Analysis of the Results
The analysis of results involves interpreting the metrics and visualizations to draw meaningful conclusions. Insights are gained into which cities exhibit similar purchasing behavior, the most profitable products in specific regions, and any patterns that may guide strategic business decisions. Any discrepancies between predicted and actual outcomes are thoroughly investigated to understand potential areas for improvement.

### Conclusion

#### Discuss Any Limitation
Despite the model's success in predicting city preferences and profits, certain limitations exist. The model assumes that consumer behavior remains constant over time, and external factors, such as economic changes or global events, are not considered. Additionally, the dataset's geographical scope is limited to the United States and Canada, potentially limiting the model's applicability to a broader international context.

#### Discuss Any Issue Not Resolved
One unresolved issue pertains to the interpretability of the model's decisions. While the model can make accurate predictions, understanding the underlying reasons for specific predictions remains a challenge. Further research into interpretable machine learning techniques may address this issue.

#### Future Direction
Future work could involve enhancing the model's predictive capabilities by incorporating more sophisticated machine learning algorithms, such as ensemble methods or neural networks. Additionally, expanding the dataset to include a more diverse set of regions and demographics would contribute to a more comprehensive understanding of consumer behavior. Collaboration with domain experts in retail and data science could provide valuable insights and further refine the model.

### Appendix

#### Snapshots and Others

### References
