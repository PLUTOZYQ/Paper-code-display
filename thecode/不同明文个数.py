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


def compare_aggregation_methods():
    paillier = Paillier(512)
    numbers = generate_random_large_integers(1000, 1024)  # 生成100个10位数的随机整数
    encrypted_numbers = [paillier.encrypt(num) for num in numbers]

    print("Aggregating 100 numbers using different methods...")

    start = time.time()
    traditional_aggregate(encrypted_numbers, paillier)
    traditional_time = time.time() - start
    print(f"The traditional aggregation took {traditional_time} seconds.")

    start = time.time()
    optimized_aggregation(encrypted_numbers, paillier)
    optimized_time = time.time() - start
    print(f"The optimized aggregation took {optimized_time} seconds.")
    
    start = time.time()
    reduce_aggregate(encrypted_numbers, paillier)
    reduce_time = time.time() - start
    print(f"The reduce aggregation took {reduce_time} seconds.")
    
compare_aggregation_methods()