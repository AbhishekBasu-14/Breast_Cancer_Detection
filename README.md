# Breast_Cancer_Detection
Dataset: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

This is the course project of Open Source Technology and Communication Lab (OSTCL) during our 5th Semester of Electronics and Telecommunication Engineering. The project compares accuracy of 5 Machine Learning Models that were implemented to predict the breast cancer stage - Benign or Malignant. The 5 ML approaches are as follows:
1. CNN
2. Random Forest
3. Logistic Regression
4. SVM
5. kNN

Models were trained after data pre-processing and feature selection using a correlation heatmap. Hypertuning of parameters was done in order to increase the accuracy of each model.

The important libraries used are as follows:
1. keras
2. tensorflow
3. sklearn
4. numpy
5. pandas
6. matplotlib
7. seaborn

The results are as follows:
| Algorithm      | Testing Accuracy | Precision      | Recall         | Specificity    |
|     :---:      |     :---:        |      :---:     |     :---:      |      :---:     | 
| CNN            |     98.24        |        96.875  | 98.412         |     98.148     |       
| RF             |     96.875       |        93.846  | 96.825         |     96.296     |  
| LR             |     98.245       |        98.387  | 96.825         |     99.074     |
| KNN            |     96.491       |        96.721  | 93.65          |     98.148     | 
| SVM            |     96.491       |        95.238  | 95.23          |     97.222     |

Comaparison Graph:
![COMPARE](https://user-images.githubusercontent.com/58266816/121813768-2f3d5080-cc8b-11eb-8f14-97561b46dc44.png)

Confusion matrix of each algorithm:

| CNN         | RF            | LR            |
| -------------- | -------------- | -------------- | 
| ![CNN](https://user-images.githubusercontent.com/58266816/121812879-40845e00-cc87-11eb-9154-252fbbfc3197.png)| ![RF](https://user-images.githubusercontent.com/58266816/121812899-5d209600-cc87-11eb-9fa5-450ab765901b.png) | ![LR](https://user-images.githubusercontent.com/58266816/121812952-95c06f80-cc87-11eb-9f07-b35c14b77ea4.png) |



| KNN                      | SVM                    |
| -------------- | -------------- |
| ![KNN](https://user-images.githubusercontent.com/58266816/121813004-cef8df80-cc87-11eb-8e04-acea93dc69b0.png) | ![SVM](https://user-images.githubusercontent.com/58266816/121813021-e506a000-cc87-11eb-813b-182bed7d9951.png) | 
