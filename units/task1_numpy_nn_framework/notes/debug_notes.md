# Task1 调试记录

## 配置层级

初版训练器按扁平字典读取 `epochs`，而正式配置将参数放在 `training` 节点。baseline 在首个 batch 前触发 `KeyError`，未产生训练结果。修正后训练器兼容完整任务配置，并继续使用完整配置摘要校验 checkpoint。

## 聚合 CSV 字段

学习率搜索完成训练后，选中行新增 `selected_by_validation` 字段，其他行缺少同名字段，导致 CSV 聚合失败。修正为每行预先写入布尔字段，再更新选中行。已完成的 epoch checkpoint 被正常恢复，没有重复训练或丢失 trace。

## 控制变量 seed

第一次探索运行给不同 run 叠加了不同 seed，使学习率或模块变化与初始化变化同时发生。该结果不用于归档结论。runner 已改为 suite 内固定模型 seed、batch shuffle seed 和数据划分，随后使用 `--force` 重跑模型对比、学习率搜索和 BN/Dropout 消融。

## Dropout 随机状态

Dropout 使用实例级 Generator，固定 seed 只确定整条随机序列，不会在每次 forward 重置 mask，也不会改变训练 batch 抽样使用的 Generator。该设计避免 Assignment 1 中固定 seed 污染全局随机状态的问题再次出现。
