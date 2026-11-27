# ml-from-scratch

把学习过程中值得亲手实现的模型和算法从空白写出来，并定期重写。

每道题有固定的要求、测试和参考实现。每次重写都从空白骨架开始，写完由测试判定对错，再由 [FSRS](https://github.com/open-spaced-repetition/py-fsrs) 算法根据这次的表现安排下一次重写的时间。同一个实现文件的 git 历史就是历次手写的记录。

## 题目

| 题目 | 内容 | 建议用时 |
| --- | --- | --- |
| [micrograd](topics/micrograd) | 标量自动微分 | 30 分钟 |
| [micrograd_nn](topics/micrograd_nn) | 用自动微分引擎搭神经元、层和多层感知机 | 15 分钟 |
| [logistic_regression](topics/logistic_regression) | 用 numpy 实现逻辑回归与梯度下降 | 15 分钟 |
| [auc](topics/auc) | AUC 与 GAUC | 15 分钟 |
| [topk_metrics](topics/topk_metrics) | Recall@K 与 NDCG@K | 10 分钟 |
| [kmeans](topics/kmeans) | 用 numpy 实现 K-means | 15 分钟 |
| [mf](topics/mf) | Embedding 与矩阵分解 | 5 分钟 |
| [skipgram](topics/skipgram) | Word2Vec Skip-gram 与负采样损失 | 10 分钟 |
| [fm](topics/fm) | 因子分解机 | 15 分钟 |
| [bpr](topics/bpr) | BPR 损失 | 5 分钟 |
| [gru_cell](topics/gru_cell) | GRU 单元：原论文写法与 PyTorch 写法 | 25 分钟 |
| [transformer_block](topics/transformer_block) | 缩放点积注意力、多头注意力、Pre-LN 编码器块 | 30 分钟 |
| [din_attention](topics/din_attention) | DIN 的注意力单元 | 10 分钟 |
| [dssm](topics/dssm) | DSSM 双塔 | 10 分钟 |
| [inbatch_softmax](topics/inbatch_softmax) | 双塔的批内负采样损失与 logQ 纠偏 | 10 分钟 |
| [dcn_cross](topics/dcn_cross) | DCN-v2 的交叉层 | 5 分钟 |
| [mmoe](topics/mmoe) | MMoE 的专家与门控 | 10 分钟 |
| [wide_deep](topics/wide_deep) | Wide & Deep | 10 分钟 |
| [deepfm](topics/deepfm) | DeepFM | 15 分钟 |
| [sasrec](topics/sasrec) | SASRec（Transformer Block 作为现成部件） | 20 分钟 |
| [rq_vae](topics/rq_vae) | RQ-VAE 的残差量化（TIGER 的语义 ID） | 15 分钟 |
| [youtube_dnn](topics/youtube_dnn) | YouTubeDNN 召回 | 10 分钟 |
| [augru](topics/augru) | DIEN 的 AUGRU 单元 | 15 分钟 |
| [esmm](topics/esmm) | ESMM | 10 分钟 |

## 用法

```bash
uv sync                  # 首次使用：创建虚拟环境并安装依赖
uv run drill             # 查看哪些题目到期
uv run drill start fm    # 开始：把实现文件重置为空白骨架并在 VSCode 中打开（省略题目名则选逾期最久的）
# 编辑 topics/fm/fm.py
uv run drill done        # 写完：跑测试、评分、记录、提交，然后在 VSCode 中打开与参考实现、与上一次的对比
```

评分由事实决定：看过参考或放弃记为 again；独立写出，但出过概念性错误（算法、公式、定义写错）记为 hard；独立写出，没有出错或只有笔误、接口细节之类的小错记为 good。测试失败本身不影响评分，写的过程中可以随时用 `drill check` 自测。

一道题第一次写的时候如果看了参考，算作首次学习，不参与间隔推算，第二天到期，从那一次独立重写开始评分。

日期按北京时间计算，以凌晨 4 点为一天的开始：显示“今天到期”的题目，当天任何时候写都可以。同一时间只能有一次手写在进行，写完或放弃后才能开始下一道。

其他命令：

```bash
uv run drill check       # 只跑测试，不记录
uv run drill diff fm     # 与上一次手写的版本对比；加 --ref 与参考实现对比
uv run drill verify      # 检验测试本身是否有效
uv run drill pause fm    # 暂停某道题的复习计划；--all 暂停所有已经写过的题目
uv run drill resume fm   # 恢复；--all 恢复全部
```

暂停只是不提醒：题目不再出现在到期列表和每日提醒中，但遗忘不会因此停止，所以到期时间不顺延。恢复后按原计划继续，已经过期的立即进入到期列表；隔了多久才复习，由 FSRS 在下一次评分时按实际间隔处理。暂停期间仍然可以用 `drill start <题目>` 主动手写。

## 怎样保证测试可靠

- 测试让手写的模块与参考实现持有同一组随机权重，比较二者的输出，所以数值上的任何偏差都会暴露。
- 参考实现本身经过独立检验：对照 PyTorch 官方模块、按定义逐样本计算，或数值微分。
- 每道题有一份典型错误写法清单（`mutants.toml`）。`uv run drill verify` 把每种错误写法当作提交跑一遍测试，确认全部被拦住。
- 如果某次手写出现了测试没有拦住的错误，就把它加进清单并补上测试。

## 目录

```
topics/<题目>/
    README.md         要求
    stub.py           空白骨架
    <题目>.py         手写的实现
    test_<题目>.py    测试
    <题目>_ref.py     参考实现（VSCode 中默认隐藏，写完后在对比视图中查看）
    mutants.toml      典型错误写法清单（VSCode 中默认隐藏）
    NOTES.md          每次手写后的错误记录
reviews.jsonl         复习记录，以及暂停、恢复的记录
src/drill/            调度工具
```

每天北京时间 8:00，GitHub Actions 会检查有没有到期的题目，有则新建一个 issue 提醒。
