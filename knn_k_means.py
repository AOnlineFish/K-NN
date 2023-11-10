# -*- coding: utf-8 -*-
"""
______________________________
  Author: iMyFish
  Email : yyqq921@163.com
   Time : 2023/11/8 15:55
    File: knn_k_means.py
Software: PyCharm
______________________________
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer


def knn(data_csv, output_csv):
    # Load the dataset from a CSV file 李思莹：这里的data_csv是什么？是从哪里来的？
    file_path = data_csv

    # Load the dataset with 'GBK' encoding 中文加载
    data = pd.read_csv(file_path, encoding='GBK')

    # Convert '参加工作时间' to datetime and calculate years of experience
    # If the format is known, add 'format' parameter to specify it
    # 转换日期 格式化
    # 以你的数据时间格式为例
    data['参加工作时间'] = pd.to_datetime(data['参加工作时间'], errors='coerce', format='%m/%d/%y')

    # Calculate the work experience in years
    # 以当前年份为单位进行计算
    current_year = pd.Timestamp('now').year
    data['工作年限'] = current_year - data['参加工作时间'].dt.year

    # Handle missing values for '工作年限'
    # Here we fill missing values with the mean of the column
    # 处理缺失值
    # 这里我们用列的平均值来填充缺失值
    imputer = SimpleImputer(strategy='mean')
    data['工作年限'] = imputer.fit_transform(data[['工作年限']])

    # Encode categorical features - '学历' and '绩效'
    # 知道啥意思不？
    # 不知道看这：特征提取
    le_education = LabelEncoder()
    le_performance = LabelEncoder()

    # Assuming '学历' and '绩效' do not have missing values
    # If they do, fill them before encoding, e.g., data['学历'].fillna('most_common_value', inplace=True)
    # 这是啥意思？处理缺失值
    data['学历_encoded'] = le_education.fit_transform(data['学历'])
    data['2020绩效_encoded'] = le_performance.fit_transform(data['2020绩效'])
    data['2021绩效_encoded'] = le_performance.fit_transform(data['2021绩效'])
    data['2022绩效_encoded'] = le_performance.fit_transform(data['2022绩效'])

    # Select and scale the numerical features for clustering
    numerical_features = data[['工作年限', '基本工资', '年薪']]
    scaler = StandardScaler()
    scaled_numerical_features = scaler.fit_transform(numerical_features)

    # Add the scaled numerical features back to the dataframe
    data[['工作年限_scaled', '基本工资_scaled', '年薪_scaled']] = scaled_numerical_features
    data['平均月薪'] = data['年薪'] / 12
    data['经验'] = data['工龄']

    # Use only the encoded and scaled features for clustering
    features_for_clustering = data[
        ['学历_encoded', '工作年限_scaled', '2020绩效_encoded', '2021绩效_encoded', '2022绩效_encoded',
         '基本工资_scaled', '年薪_scaled']]

    # Perform KNN classification
    knn = KNeighborsClassifier(n_neighbors=3)
    # knn.fit(features_for_clustering, data['是否离职'])
    #
    # # Predict the target variable
    # data['是否离职_predicted'] = knn.predict(features_for_clustering)

    # Save the data with predicted target variable to a new CSV file
    output_file_path = output_csv
    data.to_csv(output_file_path, index=False, encoding='GBK')

    print('KNN classification complete. Output saved to:', output_file_path)


def predict_data(data, output_csv):
    file_path = data

    data = pd.read_csv(file_path, encoding='GBK')

    le_education = LabelEncoder()
    le_performance = LabelEncoder()

    data['学历_encoded'] = le_education.fit_transform(data['学历'])


    output_file_path = output_csv
    data.to_csv(output_file_path, index=False, encoding='GBK')

    print('KNN classification complete. Output saved to:', output_file_path)


if __name__ == '__main__':
    knn('data/人员信息数据集.csv', 'data/人员信息_knn.csv')
    predict_data('data/人力资源专员.csv', 'data/人力资源专员_knn.csv')
    predict_data('data/销售总监.csv', 'data/销售总监_knn.csv')
