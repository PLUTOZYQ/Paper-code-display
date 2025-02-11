import logging
from functools import reduce
from Crypto.Util.number import getPrime, inverse, GCD
import random
import time
import gmpy2
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

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
        result = result % paillier.nsqr
    return result

def reduce_aggregate(c_list, paillier):
    result = reduce(lambda x, y: (x * y) % paillier.nsqr, c_list)
    return result

def lcm(a, b):
    return abs(a * b) // GCD(a, b)

def L(x, n):
    return (x - 1) // n

def generate_random_large_integers(l, w):
    random_integers = []
    for _ in range(l):
        random_integer = random.getrandbits(w)
        random_integers.append(random_integer)
    return random_integers

class Paillier:
    def __init__(self, bit_length):
        self.p = getPrime(bit_length // 2)
        self.q = getPrime(bit_length // 2)
        self.n = self.p * self.q
        self.nsqr = self.n * self.n
        self.lmbda = lcm(self.p - 1, self.q - 1)
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

def remove_outliers(data):
    data = np.array(data)
    mean = np.mean(data)
    std_dev = np.std(data)
    filtered_data = [x for x in data if (mean - 2 * std_dev < x < mean + 2 * std_dev)]
    return np.mean(filtered_data)

def compare_aggregation_methods_viz_bit_length():
    paillier = Paillier(512)
    bit_length_list = [128, 256, 512, 1024, 2048, 4096, 8192]
    traditional_times = []
    optimized_times = []
    reduce_times = []

    trials = 100  # Number of trials

    for bit_length in bit_length_list:
        traditional_trial_times = []
        optimized_trial_times = []
        reduce_trial_times = []

        for _ in range(trials):
            numbers = generate_random_large_integers(100, bit_length)
            encrypted_numbers = [paillier.encrypt(number) for number in numbers]

            start = time.perf_counter()
            traditional_aggregate(encrypted_numbers, paillier)
            traditional_trial_times.append((time.perf_counter() - start) * 1000)

            start = time.perf_counter()
            optimized_aggregation(encrypted_numbers, paillier)
            optimized_trial_times.append((time.perf_counter() - start) * 1000)

            start = time.perf_counter()
            reduce_aggregate(encrypted_numbers, paillier)
            reduce_trial_times.append((time.perf_counter() - start) * 1000)

        traditional_times.append(remove_outliers(traditional_trial_times))
        optimized_times.append(remove_outliers(optimized_trial_times))
        reduce_times.append(remove_outliers(reduce_trial_times))

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']  # 设置字体为Times New Roman

    plt.figure(figsize=(14, 10))
    plt.plot(bit_length_list, traditional_times, 'o-', label='Traditional', markersize=8, linewidth=2, color='red')
    plt.plot(bit_length_list, optimized_times, 's-', label='Optimized', markersize=8, linewidth=2, color='blue')
    plt.plot(bit_length_list, reduce_times, '^-', label='Reduce', markersize=8, linewidth=2, color='green')

    #plt.title('Comparison of Encryption Aggregation Times for Different Message Sizes', fontsize=18)
    plt.xlabel('Bit Length of messages', fontsize=24)
    plt.ylabel('Time (ms)', fontsize=24)
    plt.ylim(0.15, 0.7)  # Adjust y-axis limits
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend(fontsize=20, loc='upper left', frameon=True, framealpha=1, facecolor='white', edgecolor='black')
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # 保存为SVG格式
    plt.savefig('E:\\论文重现\\the code\\code\\shiyan\\zuizhongshiyan\\hadamajishiyan\\最终\\tu\\三个方案_不同明文位数.svg', format='svg')
    plt.show()  

compare_aggregation_methods_viz_bit_length()
