'''
Author: chenpirate chensy293@mail2.sysu.edu.cn
Date: 2023-09-06 15:18:45
LastEditors: chenpirate chensy293@mail2.sysu.edu.cn
LastEditTime: 2023-09-12 13:09:28
FilePath: /resnetV2/utils/exchange_json_key_value.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import json

def exchange_json_key_value(json_):
    have_exchange_json = {}
    for key, value in json_.items():
        have_exchange_json[value] = key
    return have_exchange_json


if __name__ == "__main__":
    json_label_path = "class_indices.json"
    labels_json = open(json_label_path, 'r')
    class_json = json.load(labels_json)
    exchange_json = exchange_json_key_value(class_json)
    for key, value in exchange_json.items():
        print(f"key:{type}, value:{type(value)}")