# -*- coding: utf-8 -*-
"""
______________________________
  Author: iMyFish
  Email : yyqq921@163.com
   Time : 2023/11/12 2:10
    File: p_data.py
Software: PyCharm
______________________________
"""
import pandas as pd

education_levels_hierarchical = {
    "博士研究生": 7, "博士": 6, "硕士研究生": 5,
    "本科": 4, "大专": 3, "高中/中专/中技": 2, "中专": 2, "初中及以下": 1
}
experience_levels = {
    "10年以上经验": 6,
    "8-9年经验": 5,
    "5-7年经验": 4,
    "3-4年经验": 3,
    "2年经验": 2,
    "1年经验": 1,
    "无明确要求": 0,
    "在校生/应届生": 0  # Assigning a negative value to indicate entry-level or no experience
}

def data_for_zz():
    # Load the dataset
    file_path = 'data/人员信息数据集.csv'  # Replace with your file path
    data = pd.read_csv(file_path, encoding='GBK')

    # Updating the education encoding to reflect the hierarchy of education levels

    data['学历_encoding'] = data['学历'].map(education_levels_hierarchical)

    # Creating experience encoding (converting working years to integers)
    data['经验_encoding'] = data['工龄'].astype(int)

    # Creating average monthly salary encoding (yearly salary divided by 12)
    data['平均月薪_encoding'] = data['年薪'] / 12

    # Saving the modified dataset
    output_file_path = 'data/在职.csv' # Replace with your desired output file path
    data.to_csv(output_file_path, index=False, encoding='GBK')


def data_for_rl(in_data, out_put):
    file_path = in_data  # Replace with your file path
    data = pd.read_csv(file_path, encoding='GBK')

    # Updating the education encoding to reflect the hierarchy of education levels

    data['学历_encoding'] = data['学历'].map(education_levels_hierarchical)

    # Creating experience encoding (converting working years to integers)
    data['经验_encoding'] = data['经验'].map(experience_levels).astype(int)

    # Creating average monthly salary encoding (yearly salary divided by 12)
    data['平均月薪_encoding'] = data['平均月薪']

    # Saving the modified dataset
    output_file_path = out_put  # Replace with your desired output file path
    data.to_csv(output_file_path, index=False, encoding='GBK')

# if __name__ == '__main__':
#     data_for_zz()
#     data_for_rl('data/人力资源专员.csv', 'data/人力.csv')
#     data_for_rl('data/销售总监.csv', 'data/销售.csv')