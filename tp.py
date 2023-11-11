# -*- coding: utf-8 -*-
"""
______________________________
  Author: iMyFish
  Email : yyqq921@163.com
   Time : 2023/11/12 3:20
    File: tp.py
Software: PyCharm
______________________________
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import p_data

# 预处理
p_data.data_for_zz()
p_data.data_for_rl('data/人力资源专员.csv', 'data/人力.csv')
p_data.data_for_rl('data/销售总监.csv', 'data/销售.csv')
# 载入数据
# def tp(data):
#     file_zai_zhi = 'data/在职.csv'
#     file_ren_li = data
#     df_zai_zhi = pd.read_csv(file_zai_zhi, encoding='GBK')
#     df_ren_li = pd.read_csv(file_ren_li, encoding='GBK')
#     #
#     # # 选择相关特征
#     # features = ['学历', '经验_encoding', '平均月薪_encoding']
#     #
#     # # 对 '学历' 进行数值编码
#     # label_encoder = LabelEncoder()
#     # combined_education = pd.concat([df_zai_zhi['学历'], df_ren_li['学历']], ignore_index=True)
#     # label_encoder.fit(combined_education)
#     # df_zai_zhi['学历_encoded'] = label_encoder.transform(df_zai_zhi['学历'])
#     # df_ren_li['学历_encoded'] = label_encoder.transform(df_ren_li['学历'])
#     #
#     # # 选择特征和目标变量
#     # df_zai_zhi_encoded = df_zai_zhi[features + ['学历_encoded']]
#     # df_ren_li_encoded = df_ren_li[features + ['学历_encoded']]
#     # if data == 'data/人力.csv':
#     #     df_zai_zhi['target'] = (df_zai_zhi['职级'] == '2A').astype(int)
#     # if data == 'data/销售.csv':
#     #     df_zai_zhi['target'] = (df_zai_zhi['职级'] == '4B').astype(int)
#     #
#     # # 分割数据集
#     # X = df_zai_zhi_encoded.drop('学历', axis=1)
#     # y = df_zai_zhi['target']
#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     #
#     # # 训练 kNN 模型
#     # knn = KNeighborsClassifier(n_neighbors=3)
#     # knn.fit(X_train, y_train)
#     #
#     # # 模型在测试集上的准确率
#     # accuracy = knn.score(X_test, y_test)
#     #
#     # # 对候选人进行预测
#     # df_ren_li_encoded = df_ren_li_encoded.drop('学历', axis=1)
#     # predictions = knn.predict(df_ren_li_encoded)
#     # df_ren_li['Prediction'] = predictions
#     #
#     # # 筛选出与 "2A" 职级相似的候选人
#     # similar_candidates = df_ren_li[df_ren_li['Prediction'] == 1]
#     #
#     # # 输出结果
#     # similar_candidates.head()
#
#     # 处理逗号分隔的数字字符串
#     df_zai_zhi['平均月薪_encoding'] = df_zai_zhi['平均月薪_encoding'].apply(clean_numeric)
#     df_ren_li['平均月薪_encoding'] = df_ren_li['平均月薪_encoding'].apply(clean_numeric)
#
#     # 选择相关特征
#     features = ['学历', '经验_encoding', '平均月薪_encoding']
#
#     # 对 '学历' 进行数值编码
#     label_encoder = LabelEncoder()
#     combined_education = pd.concat([df_zai_zhi['学历'], df_ren_li['学历']], ignore_index=True)
#     label_encoder.fit(combined_education)
#     df_zai_zhi['学历_encoded'] = label_encoder.transform(df_zai_zhi['学历'])
#     df_ren_li['学历_encoded'] = label_encoder.transform(df_ren_li['学历'])
#
#     # 选择特征和目标变量
#     df_zai_zhi_encoded = df_zai_zhi[features + ['学历_encoded']]
#     df_ren_li_encoded = df_ren_li[features + ['学历_encoded']]
#     # df_zai_zhi['target'] = (df_zai_zhi['职级'] == '2A').astype(int)
#     if data == 'data/人力.csv':
#         df_zai_zhi['target'] = (df_zai_zhi['职级'] == '2A').astype(int)
#         output_file_path = 'data/人力推荐.csv'
#     if data == 'data/销售.csv':
#         df_zai_zhi['target'] = (df_zai_zhi['职级'] == '4B').astype(int)
#         output_file_path = 'data/销售推荐.csv'
#
#     # 分割数据集
#     X = df_zai_zhi_encoded.drop('学历', axis=1)
#     y = df_zai_zhi['target']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # 训练 kNN 模型
#     knn = KNeighborsClassifier(n_neighbors=6)
#     knn.fit(X_train, y_train)
#
#     # 模型在测试集上的准确率
#     accuracy = knn.score(X_test, y_test)
#     print(accuracy)
#
#     # 对候选人进行预测
#     df_ren_li_encoded = df_ren_li_encoded.drop('学历', axis=1)
#     predictions = knn.predict(df_ren_li_encoded)
#     df_ren_li['Prediction'] = predictions
#
#     # 筛选出与 "2A" 职级相似的候选人
#     similar_candidates = df_ren_li[df_ren_li['Prediction'] == 1]
#
#     # 输出结果
#
#     similar_candidates.to_csv(output_file_path, index=False, encoding='GBK')
# # 数据清洗函数：移除数字中的逗号，并转换为浮点数
# def clean_numeric(x):
#     if isinstance(x, str):
#         return float(x.replace(',', ''))
#     return x
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import p_data

# 数据清洗函数：移除数字中的逗号，并转换为浮点数
def clean_numeric(x):
    if isinstance(x, str):
        return float(x.replace(',', ''))
    return x

# 修改后的tp函数
def tp(data):
    file_zai_zhi = 'data/在职.csv'
    file_ren_li = data
    df_zai_zhi = pd.read_csv(file_zai_zhi, encoding='GBK')
    df_ren_li = pd.read_csv(file_ren_li, encoding='GBK')

    df_zai_zhi['平均月薪_encoding'] = df_zai_zhi['平均月薪_encoding'].apply(clean_numeric)
    df_ren_li['平均月薪_encoding'] = df_ren_li['平均月薪_encoding'].apply(clean_numeric)

    features = ['学历', '经验_encoding', '平均月薪_encoding']

    label_encoder = LabelEncoder()
    combined_education = pd.concat([df_zai_zhi['学历'], df_ren_li['学历']], ignore_index=True)
    label_encoder.fit(combined_education)
    df_zai_zhi['学历_encoded'] = label_encoder.transform(df_zai_zhi['学历'])
    df_ren_li['学历_encoded'] = label_encoder.transform(df_ren_li['学历'])

    df_zai_zhi_encoded = df_zai_zhi[features + ['学历_encoded']]
    df_ren_li_encoded = df_ren_li[features + ['学历_encoded']]

    if data == 'data/人力.csv':
        df_zai_zhi['target'] = (df_zai_zhi['职级'] == '2A').astype(int)
        output_file_path = 'data/人力推荐.csv'
    if data == 'data/销售.csv':
        df_zai_zhi['target'] = (df_zai_zhi['职级'] == '4B').astype(int)
        output_file_path = 'data/销售推荐.csv'

    X = df_zai_zhi_encoded.drop('学历', axis=1)
    y = df_zai_zhi['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 使用随机森林
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_accuracy = rf.score(X_test, y_test)
    print("随机森林准确率:", rf_accuracy)

    # # 使用梯度提升树
    # gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    # gb.fit(X_train, y_train)
    # gb_accuracy = gb.score(X_test, y_test)
    # print("梯度提升树准确率:", gb_accuracy)

    df_ren_li_encoded = df_ren_li_encoded.drop('学历', axis=1)
    df_ren_li['RF_Prediction'] = rf.predict(df_ren_li_encoded)
    # df_ren_li['GB_Prediction'] = gb.predict(df_ren_li_encoded)

    similar_candidates_rf = df_ren_li[df_ren_li['RF_Prediction'] == 1]
    # similar_candidates_gb = df_ren_li[df_ren_li['GB_Prediction'] == 1]

    # 输出随机森林和梯度提升树的预测结果
    similar_candidates_rf.to_csv(output_file_path.replace('.csv', '_rf.csv'), index=False, encoding='GBK')
    # similar_candidates_gb.to_csv(output_file_path.replace('.csv', '_gb.csv'), index=False, encoding='GBK')

def print_out_rl():
    # duqu renli tuijian rf csv
    file_ren_li = 'data/人力推荐_rf.csv'
    df_ren_li = pd.read_csv(file_ren_li, encoding='GBK')
    # shuchu RF_Prediction = 1 de renli
    print(df_ren_li[df_ren_li['RF_Prediction'] == 1])
def print_out_xs():
    # duqu renli tuijian rf csv
    file_ren_li = 'data/销售推荐_rf.csv'
    df_ren_li = pd.read_csv(file_ren_li, encoding='GBK')
    # shuchu RF_Prediction = 1 de renli
    print(df_ren_li[df_ren_li['RF_Prediction'] == 1])
if __name__ == '__main__':
    tp('data/人力.csv')
    print('------------------')
    tp('data/销售.csv')
    print_out_rl()
    print_out_xs()
