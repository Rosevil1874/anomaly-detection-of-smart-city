# 岚皋路166弄数据分析

新的一轮数据，用于分析上一轮数据中总结出来的告警规则表现。

## 代码使用

main.py：主函数。

**数据处理：**
- dateparser.py：根据不同情况进行日期时间解析的几个函数；

- preprocess.py：数据预处理
	- 读取所有原文件，并聚合为同一个文件；
	- 将数据按照地址分解为小的独立的csv文件；
	- 删除同一设备在15秒内同一状态的数据；
	- 数据缺失的状态（主要是超时报警和报警解除）补充。

**超时告警下降率**：
alram_reduce_rate.py：根据新制定的超时告警规则，计算告警下降率。


## 文件目录
1. code: 源码；
2. dataset: 数据集
	- excel文件：原始文件；
	- units: 按地址划分后的数据；
	- redundancy_deleted: 删除重复状态后的数据；
	- complete_units: 补充缺失状态后的数据；
3. counts: 数据统计计算
	- open_to_close_time：每个设备“正常开门4-超时未关门报警6-开门状态0-报警解除7”事件（超时事件）的开始时间、结束事件、时长；

8. alarm_rules
	- merged：手动调参聚类结果进行合并后的规则；
	- auto_merged: 自动调参聚类结果进行合并后的规则；
	- alarm_reduce_rate.csv：手动调参规则告警下降率；
	- auto_alarm_reduce_rate.csv：自动调参规则告警下降率；