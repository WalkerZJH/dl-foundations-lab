# 调试记录

## 环境

本次使用 `D:\anaconda\envs\minimind\python.exe`，PyTorch 2.6.0+cu124，CUDA 可用。AG-News 下载时 Hugging Face 在 Windows 上提示 symlink 降级和 `hf_xet` fallback，但数据导出成功，不影响 CSV 协议。

## 检查

* `compileall` 通过。
* smoke test 通过，只用于确认链路，临时输出已清理。
* correctness checks 通过：split 数量、label、tokenizer、padding/length、single batch loss、train/eval mode 均符合预期。

## 实验现象

学习率 0.0003 在 6 epoch 内明显欠收敛；dropout=0.2 的 validation 最好；max_seq_len=256 的 test 观察较好但 validation 未超过最优配置，因此不作为选中配置。embedding_dim=256 增加参数和显存，但收益不稳定。
