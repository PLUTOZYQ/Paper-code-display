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

def reduce_aggregate(c_list, paillier):
    # 使用 functools.reduce 和 lambda 函数进行对密文进行累乘
    result = reduce(lambda x, y: (x * y) % paillier.nsqr, c_list)
    return result








"""def compare_aggregation_methods_viz_fine_grained():
    # 初始化Paillier系统以及一些列表用于存储测试结果
    paillier = Paillier(512)
    num_list = list(range(1, 101))  # 测试数据个数列表，1-100
    traditional_times = []
    optimized_times = []
    reduce_times = []

    # 对每一个数量的数据进行测试
    for num in num_list:
        numbers = generate_random_large_integers(num, 10)
        encrypted_numbers = [paillier.encrypt(number) for number in numbers]

        start = time.time()
        traditional_aggregate(encrypted_numbers, paillier)
        traditional_times.append(time.time() - start)

        start = time.time()
        optimized_aggregation(encrypted_numbers, paillier)
        optimized_times.append(time.time() - start)

        start = time.time()
        reduce_aggregate(encrypted_numbers, paillier)
        reduce_times.append(time.time() - start)

    # 绘图
    plt.plot(num_list, traditional_times, label='traditional')
    plt.plot(num_list, optimized_times, label='optimized')
    plt.plot(num_list, reduce_times, label='reduce')
    plt.xlabel('Number of messages')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.show() 

compare_aggregation_methods_viz_fine_grained()"""

def compare_aggregation_methods_viz():
    # 初始化Paillier系统以及一些列表用于存储测试结果
    paillier = Paillier(512)
    num_list = [100,1000,2000,3000,4000,5000,10000]  # 测试数据个数列表
    traditional_times = []
    optimized_times = []
    reduce_times = []

    # 对每一个数量的数据进行测试
    for num in num_list:
        numbers = generate_random_large_integers(num, 512)
        encrypted_numbers = [paillier.encrypt(number) for number in numbers]

        start = time.time()
        traditional_aggregate(encrypted_numbers, paillier)
        traditional_times.append(time.time() - start)

        start = time.time()
        optimized_aggregation(encrypted_numbers, paillier)
        optimized_times.append(time.time() - start)

        start = time.time()
        reduce_aggregate(encrypted_numbers, paillier)
        reduce_times.append(time.time() - start)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(num_list, traditional_times, 'o-', label='traditional')
    plt.plot(num_list, optimized_times, 's-', label='optimized')
    plt.plot(num_list, reduce_times, '^-', label='reduce')
    plt.xlabel('Number of messages')
    plt.ylabel('Time (s)')
    plt.grid(True) # 添加网格
    plt.legend()
    plt.show()  

compare_aggregation_methods_viz()

"""
def compare_aggregation_methods_viz_log_scale():
    # 初始化Paillier系统以及一些列表用于存储测试结果
    paillier = Paillier(512)
    num_list = list(range(1, 11)) + list(range(10, 101, 10)) + list(range(100, 1001, 100)) + list(range(1000, 10001, 1000)) + list(range(10000, 100001, 10000))
    # 数据列表，包含从1至10步长为1，10至100步长为10，100至1000步长为100，1000至10000步长为1000，10000至100000步长为10000
    traditional_times = []
    optimized_times = []
    reduce_times = []

    # 对每一个数量的数据进行测试
    for num in num_list:
        numbers = generate_random_large_integers(num, 10)
        encrypted_numbers = [paillier.encrypt(m) for m in numbers]

        start = time.time()
        traditional_aggregate(encrypted_numbers, paillier)
        traditional_times.append(time.time() - start)

        start = time.time()
        optimized_aggregation(encrypted_numbers, paillier)
        optimized_times.append(time.time() - start)

        start = time.time()
        reduce_aggregate(encrypted_numbers, paillier)
        reduce_times.append(time.time() - start)

    # 绘图
    plt.plot(num_list, traditional_times, label='traditional')
    plt.plot(num_list, optimized_times, label='optimized')
    plt.plot(num_list, reduce_times, label='reduce')
    plt.xlabel('Number of messages')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.show()

compare_aggregation_methods_viz_log_scale()
"""