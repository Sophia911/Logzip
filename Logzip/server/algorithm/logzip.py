# !/usr/bin/env python3.8
# -*- coding: utf-8 -*-
# @Time : 2023/1/24
# @Author : Siddhartha
# @Email : cuiq332@gmail.com
# @File : logzip.py
# @Software: PyCharm 2022.1.4

import re
import os
import pandas as pd
import numpy as np
import random
import shutil
import copy
import logging
from pathlib import Path
import server.algorithm.prefix_tree as ptree
import time
from functools import reduce


class LogZip:
    """
    日志压缩类，主要压缩思路为：
    自动提取日志模板，将原来的日志改为 模板id + 参数 的方式，减少模板带来的大量重复
    """

    def __init__(self, raw_log_path: str, result_path: str, delimiter: str = ' '):
        """
        :param raw_log_path: 原始日志路径
        :param result_path: 存储压缩后文件的路径
        :param delimiter: 日志分隔符，主要用于将日志体分割为 tokens
        """
        self.ratio = None
        self.raw_log_path = raw_log_path
        self.result_path = os.path.join(result_path, 'zips')
        self.delimiter = delimiter
        self.prefix_tree = ptree.PrefixTree()
        if not os.path.isdir(self.result_path):  # zips文件夹下存储所有压缩信息
            logging.info("创建zips文件夹,路径为: " + result_path)
            os.makedirs(self.result_path)

    def __split_header(self, pattern: re.Pattern) -> list:
        """
        分离出消息头，并存储消息头和消息内容，其中消息头用 csv存储，并压缩为 gzip，返回消息内容
        :param pattern: 编译后的正则表达式对象，默认不对日志做处理
        :return 消息内容列表
        """
        if pattern is None:
            logging.warning("未找到分离消息头和消息内容的正则表达式.")
            with open(self.raw_log_path, 'r', encoding='utf-8') as logs_handle:
                return logs_handle.readlines()

        msg_head = []  # 存储消息头
        msg_content = []  # 存储消息内容

        with open(self.raw_log_path, 'r', encoding='utf-8') as logs_handle:
            logging.info("开始进行消息头和消息内容的分离.")
            start_time = time.time()
            for log in logs_handle.readlines():  # 日志按正则表达式处理后，分别存储消息头和消息内容
                search_result = pattern.search(log)
                msg_head.append(search_result.groups())  # 所有消息头子字段
                msg_content.append(log[search_result.span()[1]:])
            end_time = time.time()
            logging.info(f'消息头和消息内容分离耗时: {round(end_time - start_time, 4)} s')

        # 将所有子字段通过 csv进行存储，同时以 gzip方式压缩
        pd.DataFrame(msg_head).to_csv(os.path.join(self.result_path, 'head.csv.gz'),
                                      compression='gzip', index=False, encoding='utf-8')
        logging.info(f"已压缩存储消息头,目录为: {os.path.join(self.result_path)}")
        return msg_content

    def __tokenize(self, contents: list) -> list:
        """
        将消息内容串分割为单词列表，并去除结尾的换行符
        :param contents: 日志字符串列表
        :return: 分割后的日志列表
        """
        return [re.split(self.delimiter, content.rstrip('\n')) for content in contents]

    @staticmethod
    def __sample(contents: list, rate: float) -> list:
        """
        随机取样，返回一定比例的样本
        :param contents: 消息内容列表
        :param rate: 取样比例
        :return: 样本列表
        """
        number = len(contents)
        return [contents[i] for i in random.sample(range(number - 1), int(number * rate))]

    def __cluster_by_rank(self, logs: list, top_N: int) -> list:
        """
        按各 token频率排名对日志内容进行聚类
        :param logs: 分割后的日志，二维数组
        :param top_N: 按第几名分类
        :return: 聚类后的日志，三维
        """
        # 将所有不同的 token提取出来
        token_set = set()
        for log in logs:
            token_set = token_set | set(log)
        token_set = list(token_set)

        if len(token_set) < top_N:
            logging.error('聚类发生错误 -> ' + str(logs[0]))
            raise Exception(f'没有足够数量的token可供按排名{top_N}聚类！')

        # 计算每个 token出现的频率
        frequency = [len(list(filter(lambda l: token in l, logs))) / len(logs)
                     for token in token_set]
        f = sorted(frequency, reverse=True)
        if f[top_N - 1] == 1.:  # 全部在一个聚类中
            return [logs]
        # result[1]是存在某 token的日志，result[0]是其余日志，之后还需要递归，继续分类
        result = [[], []]
        flag = token_set[frequency.index(f[top_N - 1])]
        for log in logs:
            result[int(flag in log)].append(log)
        return [result[1]] + self.__cluster_by_rank(result[0], top_N)

    def __cluster(self, logs: list, N: int) -> list:
        """
        对日志进行初步聚类
        :param logs: 需要聚类的日志（已分割为 token列表）
        :param N: 初始聚类精度，相当于预设一个初始条件：每个模板有多少个相同的 token（除变量部分）
        :return: 聚类后的日志，三维列表
        """
        clustered = self.__cluster_by_rank(logs, 1)
        logging.info("开始聚类")
        start_time = time.time()
        if N == 1:
            return clustered
        for i in range(2, N + 1):
            clustered = reduce(lambda a, b: a + b,
                               [self.__cluster_by_rank(cl, i) for cl in clustered])
        end_time = time.time()
        logging.info(f"聚类完成，用时 {round(end_time - start_time, 4)} s")
        return clustered

    @staticmethod
    def __extract_templates(logs: list, raw_templates: list, threshold: float) -> list:
        """
        将日志与未提取参数的“模板”尝试匹配，若相似度没有达到设置的阈值则视为具有新模板
        :param logs: 日志体列表
        :param raw_templates: 由聚类得来的有代表性的日志，可视为每个都具有不同模板，但还不知道参数在哪个位置
        :param threshold: 判定为有相同模板的相似度阈值
        :return: 真正的模板列表
        """
        start_time = time.time()
        for log in logs:
            j = 0
            while j < len(raw_templates):
                # 此算法只考虑每个参数位置的参数数量相同的情况
                if len(raw_templates[j]) == len(log):
                    similarity = len(list(filter(lambda tk: tk in raw_templates[j], log))) / len(log)
                    # 相似度大于等于阈值判定为有相同模板
                    if similarity >= threshold:
                        for index in range(len(log)):
                            if log[index] != raw_templates[j][index]:
                                raw_templates[j][index] = '*'
                        break
                j += 1
            if j == len(raw_templates):  # 没有找到满足条件的模板，判定为新模板
                raw_templates.append(copy.deepcopy(log))
        end_time = time.time()
        logging.info(f'提取模板成功，共有{len(raw_templates)}个模板，用时: ' +
                     str(round(end_time - start_time, 4)) + ' s')
        return copy.deepcopy(raw_templates)

    def __extract_param(self, logs: list, templates: list) -> list:
        """
        提取日志参数，并返回其模板 id和参数列表
        :param logs: 消息内容
        :param templates: 模板列表
        :return: [(id, [param]), ]
        """
        start_time = time.time()
        self.prefix_tree.add_templates(templates)
        extract_result = ptree.match_by_thread(self.prefix_tree, logs)
        end_time = time.time()
        logging.info(f'日志转换完成，用时: {round(end_time - start_time, 4)} s')
        return extract_result

    def __map(self, logs: list, templates: list):
        """
        将日志转换为模板ID和参数，分别存储在 mapping.npy 和 parameter.txt里
        :param logs: 消息内容（已分割为单词列表）
        :param templates: 模板列表
        """
        with open(os.path.join(self.result_path, 'parameter.txt'), 'w', encoding='utf-8') as param_file:
            mapping = np.zeros(len(logs), dtype=np.int32)
            params = []
            logging.info("开始转换日志")
            result = self.__extract_param(logs, templates)
            for index in range(len(logs)):
                mapping[index] = result[index][0]
                params.append(self.delimiter.join(result[index][1]))
            # 保存文件
            np.save(os.path.join(self.result_path, 'mapping.npy'), mapping)
            logging.info("映射文件已保存")
            param_file.write('\n'.join(params))
            logging.info("日志参数已写入文件.")

    def zip(self, rate: float = 0.04, threshold: float = 0.6, N: int = 2, pattern: re.Pattern = None):
        """
        日志压缩的完整流程
        :param rate: 取样比例
        :param threshold: 判定为有相同模板的相似度阈值
        :param N: 初始聚类精度，相当于预设一个初始条件：每个模板有多少个相同的 token（除变量部分）
        :param pattern: 编译后的正则表达式对象
        """
        # 分离消息头和消息内容
        contents = self.__tokenize(self.__split_header(pattern))
        # 对消息内容进行聚类，初步提取出模板
        raw_templates = [copy.deepcopy(cl[0]) for cl in self.__cluster(self.__sample(contents, rate), N)]
        # 遍历消息内容，提取所有模板
        templates = self.__extract_templates(contents, raw_templates, threshold)
        # 转换消息内容，得到映射数据
        self.__map(contents, templates)
        # 利用 json存储模板前缀树
        self.prefix_tree.create_tree(self.result_path)
        logging.info("模板已保存")
        # 压缩所有文件
        path = os.path.join(self.result_path, '..', Path(self.raw_log_path).stem + '.zip')
        # 在外部使用命令进行压缩，默认运行环境已经安装 zip
        os.system(f"zip -qrj9 {path} {self.result_path}")
        logging.info("中间文件已压缩，压缩包路径 %s", os.path.abspath(path))
        # 计算压缩比
        raw_size = os.path.getsize(self.raw_log_path)
        zip_size = os.path.getsize(path)
        self.ratio = zip_size / raw_size
        logging.info("压缩比计算完成,为 %5.4f", self.ratio)
        # 删除中间文件
        shutil.rmtree(self.result_path)
        logging.info("中间文件已删除.")

    def export_tree(self, path):
        self.prefix_tree.create_tree(path)


def decompress(zip_file_path: str, log_path: str, delimiter: str = ' '):
    """
    解压缩还原日志
    :param zip_file_path: 压缩文件路径
    :param log_path: 解压缩路径
    :param delimiter: 分隔符
    """
    # 将 zip文件解压到指定路径
    zips_dir = os.path.join(log_path, 'zips')
    os.system(f"unzip -q {zip_file_path} -d {zips_dir}")
    logging.info('已获得中间文件，开始复原日志...')
    # 打开中间文件，获取映射等数据
    templates = ptree.get_templates_list(zips_dir)
    with open(os.path.join(zips_dir, 'parameter.txt'), 'r', encoding='utf-8') as param_file:
        params = [re.split(delimiter, param.rstrip('\n')) for param in param_file.readlines()]
    maps = np.load(os.path.join(zips_dir, 'mapping.npy'))
    msg_head = pd.read_csv(os.path.join(zips_dir, 'head.csv.gz'), compression='gzip', encoding='utf-8').values.tolist()
    logging.info("已获取模板和参数等数据")
    # 还原日志内容
    logs = []
    start_time = time.time()
    for i in range(maps.size):
        template = copy.deepcopy(templates[maps[i]])
        if params[i][0] != '':
            for j in params[i]:
                template[template.index('*')] = j
        head = ''.join(msg_head[i])
        content = delimiter.join(template)
        logs.append(head + content)
    end_time = time.time()
    logging.info(f"已完成日志的还原，用时: {round(end_time - start_time, 5)} s")
    # 创建并写入日志
    with open(log_path + os.sep + f"{Path(zip_file_path).stem}.log", 'w', encoding='utf-8') as f:
        f.write('\n'.join(logs))
    shutil.rmtree(zips_dir)
    logging.info("中间文件已删除，压缩结束")

