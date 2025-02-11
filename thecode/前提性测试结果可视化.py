from functools import reduce
from Crypto.Util.number import getPrime, inverse, GCD
import random
import time
import gmpy2
import numpy as np
import matplotlib.pyplot as plt
import os
from functools import reduce

def time_and_run(function, *args):
    start_time = time.time()
    result = function(*args)
    end_time = time.time()
    return end_time - start_time, result 

def lcm(a, b):
    return abs(a*b) // GCD(a, b)

def L(x, n):
    return (x - 1) // n

def generate_data(size_in_bytes):
    return int.from_bytes(os.urandom(size_in_bytes), 'big')

def generate_random_large_integers(l, w):
    random_integers = []
    for _ in range(l):
        # 生成一个随机整数，位数为w
        random_integer = random.randint(10**(w-1), 10**w - 1)
        random_integers.append(random_integer)
    return random_integers

class Paillier:
    def __init__(self, bit_length):
        self.p = getPrime(bit_length // 2)
        self.q = getPrime(bit_length // 2)
        self.n = self.p * self.q
        self.nsqr = self.n * self.n
        self.lmbda = lcm(self.p-1, self.q-1)
        self.g = self.n + 1
        x = pow(self.g, self.lmbda, self.nsqr)
        self.mu = inverse(L(x, self.n), self.n)

    def encrypt(self, m):
        r = random.randint(1, self.n)
        c = pow(self.g, m, self.nsqr) * pow(r, self.n, self.nsqr) % self.nsqr
        return c

    def decrypt(self, c):
        x = pow(c, self.lmbda, self.nsqr)
        l = L(x, self.n)
        m = (l * self.mu) % self.n
        return m



def encrypt_paillier(paillier, g, n, m_list):
    c_list = [paillier.encrypt(m, n, g) for m in m_list]
    return c_list

def decrypt_paillier(paillier, c_list,lmbda,n,mu):
    m_list = [paillier.decrypt(c,lmbda,n,mu) for c in c_list]
    return m_list

def hadamard_product_mod(matrix1, matrix2):
    result = np.multiply(matrix1, matrix2)
    return result

def list_to_matrix(lst):
    array = np.array(lst)
    matrix = array.reshape(1, -1)
    return matrix


def optimized_aggregation(c_list, paillier):
    a = len(c_list)  # 获取总元素个数
    b = gmpy2.mpz(1)  # 初始化 b 为乘法中的1，对应加密的1

    # 将列表转换为行数为1，列数为a的数组
    matrix = np.array(c_list, dtype=object)
    
    if a < 4:
        result = reduce(lambda x, y: gmpy2.mul(x, y) % paillier.nsqr, matrix)  # 如果元素个数小于4，则直接相乘
    else:
        if a % 2 == 1:
            b = matrix[-1]  # 如果元素个数为奇数，将最后一个元素赋值给b
            matrix = np.delete(matrix, -1)  # 并且从矩阵中删除这一元素
        mid = a // 2
        left_matrix = matrix[:mid]  # 将矩阵拆分成两个一样的子矩阵
        right_matrix = matrix[mid:]
        result_matrix = [gmpy2.mul(l, r) % paillier.nsqr for l, r in zip(left_matrix, right_matrix)]  # 进行哈达玛积（element-wise乘法）
        result = reduce(lambda x, y: gmpy2.mul(x, y) % paillier.nsqr, result_matrix) * b % paillier.nsqr  # 所有元素相乘得到最终结果
    return result


def traditional_aggregate(C_list, paillier):
    result = 1
    for c in C_list:
        result *= c 
        result=result % paillier.nsqr
    return result
"""
def test_aggregation_methods():
    paillier = CRT_Paillier(512)  # 初始化Paillier系统
    my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 定义一个测试列表
    C_list = [paillier.encrypt(m) for m in my_list]  # 加密列表中的每个元素

    # 使用传统方法聚合
    traditional_result = traditional_aggregate(C_list, paillier)
    print("Traditional aggregation result (encrypted):", traditional_result)

    # 使用优化方案聚合
    optimized_result = optimized_aggregation(C_list, paillier)
    print("Optimized aggregation result (encrypted):", optimized_result)"""



def reduce_aggregate(c_list, paillier):
    # 使用 functools.reduce 和 lambda 函数进行对密文进行累乘
    result = reduce(lambda x, y: (x * y) % paillier.nsqr, c_list)
    return result




"""def test_homomorphic_property():
    paillier = Paillier(512)  # 初始化 Paillier 系统
    my_list = generate_random_large_integers(1000, 512)  # 定义一个测试列表
    C_list = [paillier.encrypt(m) for m in my_list]  # 加密列表中的每个元素

    # 使用传统方法聚合
    traditional_result = traditional_aggregate(C_list, paillier)
    traditional_decrypted = paillier.decrypt(traditional_result)
    traditional_expected = sum(my_list) % paillier.n
    print("Traditional aggregation result (decrypted):", traditional_decrypted)
    print("Traditional expected result:", traditional_expected)
    print("Homomorphic property holds for traditional aggregation:", traditional_decrypted == traditional_expected)

    # 使用优化方案聚合
    optimized_result = optimized_aggregation(C_list, paillier)
    optimized_decrypted = paillier.decrypt(optimized_result)
    print("Optimized aggregation result (decrypted):", optimized_decrypted)
    print("Optimized expected result:", traditional_expected)
    print("Homomorphic property holds for optimized aggregation:", optimized_decrypted == traditional_expected)

    # 使用 reduce 方法聚合
    reduce_result = reduce_aggregate(C_list, paillier)
    reduce_decrypted = paillier.decrypt(reduce_result)
    print("Reduce aggregation result (decrypted):", reduce_decrypted)
    print("Reduce expected result:", traditional_expected)
    print("Homomorphic property holds for reduce aggregation:", reduce_decrypted == traditional_expected)

test_homomorphic_property()

"""

import seaborn as sns
import pandas as pd

def test_homomorphic_property_heatmap():
    paillier = Paillier(512)  # Initialise Paillier system
    my_list = generate_random_large_integers(1000, 512)  # Define a list
    C_list = [paillier.encrypt(m) for m in my_list]  # Encypt each element in the list

    # Aggregation using traditional method
    traditional_result = traditional_aggregate(C_list, paillier)
    traditional_decrypted = paillier.decrypt(traditional_result)
    traditional_expected = sum(my_list) % paillier.n 

    # Aggregation with optimal method
    optimized_result = optimized_aggregation(C_list, paillier)
    optimized_decrypted = paillier.decrypt(optimized_result)

    # Aggregation using reduce method
    reduce_result = reduce_aggregate(C_list, paillier)
    reduce_decrypted = paillier.decrypt(reduce_result)

    # Comparisons
    traditional_comparison = traditional_decrypted == traditional_expected
    optimized_comparison = optimized_decrypted == traditional_expected
    reduce_comparison = reduce_decrypted == traditional_expected

    results = [traditional_comparison, optimized_comparison, reduce_comparison]
    methods = ['Traditional', 'Optimized', 'Reduce']

    data = pd.DataFrame(data=results, index=methods, columns=["Result"])
    # Transpose the data to get 3 x 1 dimension
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']  # 设置字体为Times New Roman
    plt.figure(figsize=(5, 5))
    ax = sns.heatmap(data, cmap="coolwarm", linewidths=0.5, cbar_kws={"orientation": "horizontal"})
    ax.set_title('Homomorphic Property Comparison')
    plt.show()

test_homomorphic_property_heatmap()

"""import matplotlib as mpl
def test_homomorphic_property_heatmap():
    paillier = Paillier(512)  # Initialise Paillier system
    my_list = generate_random_large_integers(1000, 512)  # Define a list
    C_list = [paillier.encrypt(m) for m in my_list]  # Encypt each element in the list

    # Aggregation using traditional method
    traditional_result = traditional_aggregate(C_list, paillier)
    traditional_decrypted = paillier.decrypt(traditional_result)
    traditional_expected = sum(my_list) % paillier.n 
    
    # Aggregation with optimal method
    optimized_result = optimized_aggregation(C_list, paillier)
    optimized_decrypted = paillier.decrypt(optimized_result)

    # Aggregation using reduce method
    reduce_result = reduce_aggregate(C_list, paillier)
    reduce_decrypted = paillier.decrypt(reduce_result)

    # Comparisons, convert True to 1, False to 0
    traditional_comparison = int(traditional_decrypted == traditional_expected)
    optimized_comparison = int(optimized_decrypted == traditional_expected)
    reduce_comparison = int(reduce_decrypted == traditional_expected)

    results = [traditional_comparison, optimized_comparison, reduce_comparison]
    methods = ['Traditional', 'Optimized', 'Reduce']

    data = pd.DataFrame(data=results, index=methods, columns=["Result"])

    # Color map : 1 - Red, 0 - Blue
    cmap = mpl.colors.ListedColormap(['blue', 'red'])
    bounds = [0,0.5,1]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(5, 5))
    ax = sns.heatmap(data, cmap=cmap, norm=norm, cbar_kws={"ticks": [0, 1], "orientation": "horizontal"})
    ax.set_title('Homomorphic Property Comparison')
    plt.show()

test_homomorphic_property_heatmap()"""






"""
import matplotlib.pyplot as plt

def test_homomorphic_property_errorbar():
    paillier = Paillier(512)
    my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    C_list = [paillier.encrypt(m) for m in my_list]

    traditional_result = traditional_aggregate(C_list, paillier)
    traditional_decrypted = paillier.decrypt(traditional_result)
    traditional_expected = sum(my_list) % paillier.n 
    traditional_comparison = traditional_decrypted == traditional_expected

    optimized_result = optimized_aggregation(C_list, paillier)
    optimized_decrypted = paillier.decrypt(optimized_result)
    optimized_comparison = optimized_decrypted == traditional_expected

    reduce_result = reduce_aggregate(C_list, paillier)
    reduce_decrypted = paillier.decrypt(reduce_result)
    reduce_comparison = reduce_decrypted == traditional_expected

    results = [traditional_comparison, optimized_comparison, reduce_comparison]
    methods = ['Traditional', 'Optimized', 'Reduce']

    y_pos = range(len(methods))
    
    plt.errorbar(y_pos, results, fmt='o')
    plt.xticks(y_pos, methods)
    plt.ylabel('Test Result')
    plt.title('Homomorphic Property Test Results')

    plt.show()

test_homomorphic_property_errorbar()
"""