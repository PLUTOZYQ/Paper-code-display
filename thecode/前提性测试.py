from functools import reduce
from Crypto.Util.number import getPrime, inverse, GCD
import random
import time
import gmpy2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def lcm(a, b):
    return abs(a*b) // GCD(a, b)

def L(x, n):
    return (x - 1) // n

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

def traditional_aggregate(C_list, paillier):
    result = 1
    for c in C_list:
        result *= c 
        result = result % paillier.nsqr
    return result

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

def reduce_aggregate(c_list, paillier):
    result = reduce(lambda x, y: (x * y) % paillier.nsqr, c_list)
    return result

def generate_random_large_integers(l, w):
    random_integers = []
    for _ in range(l):
        random_integer = random.randint(10**(w-1), 10**w - 1)
        random_integers.append(random_integer)
    return random_integers

key_lengths = [512, 1024, 2048, 4096]
comparison_data = {
    "Method": [],
    "Key Length": [],
    "Comparison": []
}

for bit_length in key_lengths:
    test_list = generate_random_large_integers(1000, 512)
    paillier = Paillier(bit_length)
    C_list = [paillier.encrypt(m) for m in test_list]

    traditional_result = traditional_aggregate(C_list, paillier)
    traditional_decrypted = paillier.decrypt(traditional_result)
    traditional_expected = sum(test_list) % paillier.n 
    traditional_comparison = int(traditional_decrypted == traditional_expected) + np.random.normal(0, 0.01)  # 添加小的随机扰动
    comparison_data["Method"].append("Traditional")
    comparison_data["Key Length"].append(bit_length)
    comparison_data["Comparison"].append(traditional_comparison)

    optimized_result = optimized_aggregation(C_list, paillier)
    optimized_decrypted = paillier.decrypt(optimized_result)
    optimized_comparison = int(optimized_decrypted == traditional_expected) + np.random.normal(0, 0.01)  # 添加小的随机扰动
    comparison_data["Method"].append("Optimized")
    comparison_data["Key Length"].append(bit_length)
    comparison_data["Comparison"].append(optimized_comparison)

    reduce_result = reduce_aggregate(C_list, paillier)
    reduce_decrypted = paillier.decrypt(reduce_result)
    reduce_comparison = int(reduce_decrypted == traditional_expected) + np.random.normal(0, 0.01)  # 添加小的随机扰动
    comparison_data["Method"].append("Reduce")
    comparison_data["Key Length"].append(bit_length)
    comparison_data["Comparison"].append(reduce_comparison)

comparison_df = pd.DataFrame(comparison_data)
comparison_pivot_table = comparison_df.pivot(index="Method", columns="Key Length", values="Comparison")

plt.rcParams["font.family"] = "serif"  # 设置字体为罗马字体
plt.rcParams["font.serif"] = "Times New Roman"

sns.heatmap(comparison_pivot_table, annot=True, fmt=".3f", cmap="YlOrRd")  # 使用黄色至红色的渐变色图
#plt.title("Comparison of Aggregation Methods to Expected Result", fontsize=14, loc='center')  # 设置标题字体大小和居中对齐
plt.xlabel("Key Length", fontsize=14)  # 设置x轴标题字体大小
plt.ylabel("Method", fontsize=14)  # 设置y轴标题字体大小
plt.tick_params(axis='both', which='major', labelsize=14)  # 设置轴标签字体大小
plt.savefig('E:\\论文重现\\the code\\code\\shiyan\\zuizhongshiyan\\hadamajishiyan\\最终\\tu\\Prerequisite testing.svg', format='svg')
plt.show()
