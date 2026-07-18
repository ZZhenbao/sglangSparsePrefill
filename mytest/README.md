# Sparse Prefix 测试数据

Qwen2.5-14B 的服务启动与完整测试命令见 [RUN_QWEN14B.md](./RUN_QWEN14B.md)。

`datasets/` 包含 5 组确定性的 token-ID 数据：

| 文件 | Prefix tokens | Suffix 数量 | 每个 suffix tokens |
| --- | ---: | ---: | ---: |
| `prefix_1k.json` | 1,024 | 2 | 32 |
| `prefix_2k.json` | 2,048 | 2 | 32 |
| `prefix_4k.json` | 4,096 | 2 | 32 |
| `prefix_10k.json` | 10,240 | 2 | 32 |
| `prefix_16k.json` | 16,384 | 2 | 32 |

每个文件的结构为：

```json
{
  "name": "prefix_1k",
  "prefix_length": 1024,
  "prefix_input_ids": ["..."],
  "suffixes": [
    {"name": "suffix_1", "length": 32, "input_ids": ["..."]},
    {"name": "suffix_2", "length": 32, "input_ids": ["..."]}
  ]
}
```

一次测试的完整输入由下式构造：

```python
full_input_ids = dataset["prefix_input_ids"] + suffix["input_ids"]
```

数据使用 `[1000, 30000)` 的普通 token ID，适用于当前 Qwen2.5 实验。测试前仍应断言目标模型的 `vocab_size >= 30000`。

重新生成数据：

```bash
python mytest/generate_datasets.py
```

修改 suffix 长度：

```bash
python mytest/generate_datasets.py --suffix-length 64
```

`datasets/manifest.json` 记录精确长度和每个数据文件的 SHA-256。
