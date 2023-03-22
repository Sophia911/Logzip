# !/usr/bin/env python3.8
# -*- coding: utf-8 -*-
# @Time : 2023/1/23
# @Author : Siddhartha
# @Email : cuiq332@gmail.com
# @File : prefix_tree.py
# @Software: PyCharm 2022.1.4

import json
import os.path
import threading
import copy
from functools import reduce


class PrefixTree:
    """
    前缀树通过 json格式保存结构信息，并且属性格式有一定要求，方便前端渲染显示，如 pid属性，根节点为 START，
    每个节点 id唯一
    """

    def __init__(self):
        self.tree_json = {
            "id": 1,
            "token": "START",
            "children": []
        }
        self.node_id = 2

    def add_templates(self, templates: list):
        """
        根据模板列表构建一颗前缀树
        :param templates: 模板列表，二维，即每个模板用 token list表示
        :return: None
        """
        for tmp_index, template in enumerate(templates):
            # 每个模板都搜索一次前缀树
            current_nodes = self.tree_json['children']
            for token_index in range(len(template)):
                search_result = self.__search_node(current_nodes, template[token_index])
                if search_result[0]:  # 找到重复节点，接着搜索下一层
                    current_nodes = search_result[1]['children']
                else:  # 未找到重复节点，需要创建新的分支
                    current_nodes.append(self.__create_branch(template[token_index:], tmp_index))
                    break

    def __create_branch(self, tokens: list, index: int) -> dict:
        """
        在前缀树中创建一条新的分支
        :param tokens: 分支路径上的 token
        :param index: 此分支对应的模板编号
        :return: 一个类似前缀树的分支结构
        """
        result_list = []
        temp_list = []
        result_list.append(temp_list)
        for token in tokens:
            new_node = {
                "id": self.node_id,
                "token": token,
                "children": []
            }
            self.node_id += 1
            temp_list.append(new_node)
            temp_list = new_node['children']
        temp_list.append({'END': index})
        return result_list[0][0]

    @staticmethod
    def __search_node(nodes: list, token: str) -> tuple:
        """
        搜索当前字典节点列表中是否有 token对应的节点
        :param nodes: 节点列表
        :param token: 寻找的目标
        :return: 三元组，第一个是搜索结果，如果找到满足条件节点则为 True，第二个放找到的对应节点
        第三个用于匹配，查找当前节点列表中是否存在通配符‘*’，即参数
        """
        has_node = False
        target_node = None
        param_node = None
        for node in nodes:
            if node['token'] == token:
                has_node = True
                target_node = node
            elif node['token'] == '*':
                param_node = node
        return has_node, target_node, param_node

    def create_tree(self, inter_dir):
        """
        将当前前缀树转换为 json格式
        :param inter_dir: 存放 json文件的目录
        :return: None
        """
        with open(os.path.join(inter_dir, 'templates.json'), 'w') as templates_json:
            templates_json.write(json.dumps(self.tree_json))

    def create_chart(self, target_dir, filename):
        """
        将前缀树转换为前端显示的 json文件
        :param filename: 文件名
        :param target_dir: 文件目录
        :return: None
        """
        orgchart = copy.deepcopy(self.tree_json)

        def search(nodes):
            for node in nodes:
                if node['children'][0].get('END') is not None:
                    node['children'].clear()
                else:
                    search(node['children'])
        search(orgchart['children'])
        with open(os.path.join(target_dir, filename + '.json'), 'w') as chart_json:
            chart_json.write(json.dumps(orgchart))

    def extract_param(self, raw_logs: list) -> list:
        """
        将原始日志转换为（模板id + 参数列表）的形式，因为只需要读取前缀树，因此可以方便地使用多线程并行处理
        :param raw_logs: 原始日志体
        :return: （模板id，参数列表）列表
        """
        extract_result = []
        for log in raw_logs:
            current_nodes = self.tree_json['children']
            params = []
            for token in log:
                search_result = self.__search_node(current_nodes, token)
                if search_result[0]:
                    # 找到相同 token，则此 token属于模板，继续处理下一个 token
                    current_nodes = search_result[1]['children']
                elif search_result[2] is not None:
                    # 未找到相同 token，但有'*'，表示此处为变量部分
                    current_nodes = search_result[2]['children']
                    params.append(token)
                else:
                    raise Exception('异常的原始日志（无法匹配）: ' + log)
            # 对 log的每个 token都遍历后，检查当前路径最后的路径编号
            number = current_nodes[0].get('END')
            if number is None:
                raise Exception('此模板没有编号: ' + log)
            extract_result.append((number, params))
        return extract_result


class MatchThread(threading.Thread):
    """
    自定义多线程类，用于日志匹配过程
    """

    def __init__(self, ptree: PrefixTree, logs: list):
        threading.Thread.__init__(self)
        self.ptree = ptree
        self.logs = logs
        self.container = None

    def run(self) -> None:
        self.container = self.ptree.extract_param(self.logs)


def match_by_thread(ptree: PrefixTree, logs: list, thread_num=4) -> list:
    """
    多线程处理转换原始日志
    :param ptree: 存有模板结构的前缀树
    :param logs: 原始日志体
    :param thread_num: 需要的线程数量
    :return:（模板id，参数列表）列表
    """
    workload = len(logs) // thread_num
    threads = []
    for num in range(thread_num - 1):
        thread = MatchThread(ptree, logs[num * workload: (num + 1) * workload])
        threads.append(thread)
        thread.start()
    thread = MatchThread(ptree, logs[(thread_num - 1) * workload:])
    threads.append(thread)
    thread.start()
    for th in threads:
        th.join()
    # 在所有线程都处理完后，合并之后返回列表
    return reduce(lambda a, b: a + b, [th.container for th in threads])


def broad_search(prefix: list, nodes: list) -> list:
    """
    深度优先展开前缀树，递归返回一个模板列表
    :param prefix: token前缀列表
    :param nodes: 子节点列表
    :return: 模板列表
    """
    if nodes[0].get('END') is not None:
        return [(nodes[0]['END'], copy.deepcopy(prefix))]
    branches = []
    for node in nodes:
        this_prefix = prefix + [node['token']]
        branches.extend(broad_search(this_prefix, node['children']))
    return branches


def get_templates_list(template_dir: str):
    """
    将前缀树按编号大小转化为模板列表，用于后续解压缩操作
    :param template_dir: 存放模板 json文件的目录
    :return: 模板列表，每个模板表示为 token list形式
    """
    with open(os.path.join(template_dir, 'templates.json'), 'r') as templates_json:
        templates_tree = json.load(templates_json)
        templates_tuple_list = broad_search([], templates_tree['children'])
        templates_tuple_list.sort(key=lambda item: item[0])
        return [template_tuple[1] for template_tuple in templates_tuple_list]

