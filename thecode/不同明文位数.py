""""from functools import reduce
from Crypto.Util.number import getPrime, inverse, GCD
import random
import time
import gmpy2
import numpy as np
import matplotlib.pyplot as plt
import os

def time_and_run(function, *args):
    start_time = time.perf_counter()
    result = function(*args)
    end_time = time.perf_counter()
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

def optimized_aggregation(c_list, paillier):
    a = len(c_list)
    b = gmpy2.mpz(1)
    matrix = np.array(c_list, dtype=object)
    if a < 4:
        result = reduce(lambda x, y: gmpy2.mul(x, y) % paillier.nsqr, matrix)
    else:
        if a % 2 == 1:
            b = matrix[-1]
            matrix = np.delete(matrix, -1)
        mid = a // 2
        left_matrix = matrix[:mid]
        right_matrix = matrix[mid:]
        result_matrix = [gmpy2.mul(l, r) % paillier.nsqr for l, r in zip(left_matrix, right_matrix)]
        result = reduce(lambda x, y: gmpy2.mul(x, y) % paillier.nsqr, result_matrix) * b % paillier.nsqr
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






def compare_aggregation_methods_viz_bit_length():
    paillier = Paillier(512)
    bit_length_list = [64, 128, 256, 512, 1024, 2048, 4096]
    traditional_times = []
    optimized_times = []
    reduce_times = []

    for bit_length in bit_length_list:
        numbers = generate_random_large_integers(100, bit_length)
        encrypted_numbers = [paillier.encrypt(number) for number in numbers]

        start = time.perf_counter()
        traditional_aggregate(encrypted_numbers, paillier)
        traditional_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        optimized_aggregation(encrypted_numbers, paillier)
        optimized_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        reduce_aggregate(encrypted_numbers, paillier)
        reduce_times.append(time.perf_counter() - start)

    plt.figure(figsize=(10, 6))
    plt.plot(bit_length_list, traditional_times, 'o-', label='Traditional')
    plt.plot(bit_length_list, optimized_times, 's-', label='Optimized')
    plt.plot(bit_length_list, reduce_times, '^-', label='Reduce')
    plt.xlabel('Bit Length of messages')
    plt.ylabel('Time (s)')
    plt.grid(True)
    plt.legend()
    plt.show()  

compare_aggregation_methods_viz_bit_length()
"""


from functools import reduce
from Crypto.Util.number import getPrime, inverse, GCD
import random
import time
import gmpy2
import numpy as np
import matplotlib.pyplot as plt
import os

def optimized_aggregation(c_list, paillier):
    a = len(c_list)
    b = gmpy2.mpz(1)
    matrix = np.array(c_list, dtype=object)
    if a < 4:
        result = reduce(lambda x, y: gmpy2.mul(x, y) % paillier.nsqr, matrix)
    else:
        if a % 2 == 1:
            b = matrix[-1]
            matrix = np.delete(matrix, -1)
        mid = a // 2
        left_matrix = matrix[:mid]
        right_matrix = matrix[mid:]
        result_matrix = [gmpy2.mul(l, r) % paillier.nsqr for l, r in zip(left_matrix, right_matrix)]
        result = reduce(lambda x, y: gmpy2.mul(x, y) % paillier.nsqr, result_matrix) * b % paillier.nsqr
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


def time_and_run(function, *args):
    start_time = time.perf_counter()
    result = function(*args)
    end_time = time.perf_counter()
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

import time

def compare_aggregation_methods_viz_bit_length():
    paillier = Paillier(512)
    bit_length_list = [128,256,512,1024,4096,8192]
    traditional_times = []
    optimized_times = []
    reduce_times = []

    for bit_length in bit_length_list:
        numbers = generate_random_large_integers(100, bit_length)
        encrypted_numbers = [paillier.encrypt(number) for number in numbers]

        start = time.perf_counter()
        traditional_aggregate(encrypted_numbers, paillier)
        traditional_times.append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        optimized_aggregation(encrypted_numbers, paillier)
        optimized_times.append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        reduce_aggregate(encrypted_numbers, paillier)
        reduce_times.append((time.perf_counter() - start) * 1000)

    plt.figure(figsize=(10, 6))
    plt.plot(bit_length_list, traditional_times, 'o-', label='Traditional')
    plt.plot(bit_length_list, optimized_times, 's-', label='Optimized')
    plt.plot(bit_length_list, reduce_times, '^-', label='Reduce')

    """ # 添加坐标点显示，并调整位置
    offset = max(traditional_times + optimized_times + reduce_times) * 0.01  # 定义一个偏移量
    for i in range(len(bit_length_list)):
        plt.text(bit_length_list[i], traditional_times[i] + offset, f'({bit_length_list[i]}, {traditional_times[i]:.2f} ms)')
        plt.text(bit_length_list[i], optimized_times[i] + 2*offset, f'({bit_length_list[i]}, {optimized_times[i]:.2f} ms)')
        plt.text(bit_length_list[i], reduce_times[i] - offset, f'({bit_length_list[i]}, {reduce_times[i]:.2f} ms)')
    """
    plt.xlabel('Bit Length of messages')
    plt.ylabel('Time (ms)')
    plt.grid(True)
    plt.legend()
    plt.show()  

compare_aggregation_methods_viz_bit_length()