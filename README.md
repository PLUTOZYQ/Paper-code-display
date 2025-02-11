# Paper-code-display
Code display for the paper "CRT Paillier Homomorphic Privacy Protection Scheme Based on BLS Signature in Mobile Vehicle Networks" and subsequent papers

## 注意事项
1. **结果图文件夹**：项目中对应的结果图存放在“tu”文件夹中，格式为svg，这种格式更符合论文要求。但是注意，这个文件夹中的有关BLS签名的结果是之后进行完善时加入的并不在此论文中。
2. **核心代码内容**：代码主要包含四个核心部分，分别是在前提性测试、不同密钥长度、不同明文个数、不同明文位数的情况下，对传统方案与利用中国剩余定理优化后的方案进行加密时间对比。
3. **论文后续完善计划**：此篇论文是我的处女作，存在一些不足。例如，BLS签名的相应测试尚未实现，也未完成车联网场景的模拟。不过，在论文投稿完成后的2个月，我已实现BLS签名的相应测试，并取得了一些创新成果。这些内容将在另一篇文章发表后继续更新上传，目前车联网场景模拟正在学习中。未来打算模拟车联网场景（如用Socket模拟车辆通信），测试加密方案的性能（吞吐量、延迟）。工具链示例：Python + Docker（容器化部署） + Jupyter Notebook（可视化分析）。但目前只用python实现了相应的性能测试以及结果可视化。
4. **消息模拟方式**：论文中的消息采用随机整数模拟，由于十进制数可以转字符串，其他进制数也可转字符串，为简便运算，直接采用此方式。

## 项目概述
本项目实现了一种高效的车联网隐私保护方案，适用于车载自组织网络（VANET）中的车辆 - 基础设施通信场景。方案结合CRT优化的Paillier同态加密和BLS短签名技术，在保证数据隐私性的同时降低计算与通信开销。（具体内容可直接在谷歌学术搜索《CRT - Paillier Homomorphic Privacy Protection Scheme Based on BLS Signatures in Mobile Vehicular Networks》查看，目前IEEE xplore正在维护，暂无法提供链接。）
