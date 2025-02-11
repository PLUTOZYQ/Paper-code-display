# Paper-code-display
Code display for the paper "CRT Paillier Homomorphic Privacy Protection Scheme Based on BLS Signature in Mobile Vehicle Networks" and subsequent papers
## 注意：
##### 1.文件夹“tu”是我项目中对应的结果图，格式是svg，这更符合论文的要求。
##### 2.代码主要是四个核心：即前提性测试，不同密钥长度，不同明文个数，不同明文位数下，传统方案与我们利用中国剩余定理优化后方案的加密时间对比。
##### 3.这篇论文是我的处女作，当然存在许多不足，比如BLS签名的相应测试并未实现，并且并未实现一个车联网场景的模拟....但是在这篇文章投稿完成后的2个月，我实现了BLS签名的相应测试，以及一些创新，这在我另一篇文章发表后我会继续更新上传，而车联网场景模拟正在学习。
##### 4.论文中的消息是用随机整数模拟的，十进制可以转字符串，其他进制数也可以转字符串，这里就简便运算了。
##### 5.


## 项目概述
##### 本项目实现了一种高效的车联网隐私保护方案，适用于车载自组织网络（VANET）中的车辆-基础设施通信场景。方案结合CRT优化的Paillier同态加密和BLS短签名技术，在保证数据隐私性的同时降低计算与通信开销。(具体内容可以直接谷歌学术搜索《CRT-Paillier Homomorphic Privacy Protection Scheme Based on BLS Signatures in Mobile Vehicular Networks》即可查看，目前IEEE xplore正在维护，便不放链接了)
