# Qwen2.5-14B Sparse Prefix 测试命令

本说明使用单张 A800、本地 Qwen2.5-14B-Instruct、`sparse_prefill` prefill backend 和 TorchNative decode。首次 cache miss 走 dense warm，后续完整 prefix hit 走 sparse 路径。

## 1. 进入环境

```bash
cd /home/zhenbao/Pro/SparsePrefill/sglangSparsePrefill
conda activate sglang-prefill
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
```

确认加载的是当前仓库，而不是相邻的 SGLang checkout：

```bash
python -c "import importlib.util; print(importlib.util.find_spec('sglang').origin)"
```

期望输出：

```text
/home/zhenbao/Pro/SparsePrefill/sglangSparsePrefill/python/sglang/__init__.py
```

## 2. 启动服务

在终端 1 执行：

```bash
export MODEL_PATH=/data0/zhenbao/weight/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port 30000 \
  --dtype bfloat16 \
  --tp-size 1 \
  --dp-size 1 \
  --pp-size 1 \
  --max-total-tokens 20000 \
  --max-running-requests 1 \
  --attention-backend torch_native \
  --prefill-attention-backend sparse_prefill \
  --decode-attention-backend torch_native \
  --sparse-policy token_h2o \
  --sparse-ratio 1.0 \
  --sparse-sink-tokens 0 \
  --sparse-recent-tokens 0 \
  --selection-unit-size 1 \
  --page-size 1 \
  --chunked-prefill-size -1 \
  --cuda-graph-backend-prefill disabled \
  --cuda-graph-backend-decode disabled \
  --disable-overlap-schedule
```

必须显式设置全局 `--attention-backend torch_native`。当 prefill 和 decode backend 不同时，若省略全局 backend，SGLang 会为它选择默认值；当前 A800 环境会默认选择 FlashInfer，并触发不必要的 FlashInfer 版本检查。

`--disable-overlap-schedule` 是布尔开关，后面不加 `disabled`；正确拼写只有一个结尾 `e`。

这里固定 `sparse-ratio=1.0`，目标是先验证 prefix KV 与 dense 等价，不是测稀疏收益。Qwen2.5-14B 本地配置的 context length 为 32K，因此最长的 `16K prefix + 32 suffix + 1 output` 在范围内。

`max-total-tokens=20000` 足以容纳最长 case，同时避免把剩余显存全部分给 KV pool，为 16K dense TorchNative attention 保留 workspace。

等待模型加载完成，然后在终端 2 检查服务：

```bash
curl -f http://127.0.0.1:30000/health
```

该接口成功时可能没有响应正文，以退出码 0 为准。

## 3. 测试五组 Prefix

先进入与终端 1 相同的目录和环境：

```bash
cd /home/zhenbao/Pro/SparsePrefill/sglangSparsePrefill
conda activate sglang-prefill
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
```

每条命令都会测试该 prefix 对应的两个 suffix。对每个 suffix，客户端自动执行：

```text
flush
→ cold dense(prefix + suffix)
→ flush
→ dense warm(prefix, max_new_tokens=0)
→ sparse hit(prefix + suffix)
→ 检查 cached_tokens == prefix_length
→ 比较 cold/hit 的 token 与 logprob
→ flush
```

### 1K Prefix

```bash
python mytest/run_prefix_case.py mytest/datasets/prefix_1k.json \
  --base-url http://127.0.0.1:30000 \
  --logprob-atol 0.1
```

### 2K Prefix

```bash
python mytest/run_prefix_case.py mytest/datasets/prefix_2k.json \
  --base-url http://127.0.0.1:30000 \
  --logprob-atol 0.1
```

### 4K Prefix

```bash
python mytest/run_prefix_case.py mytest/datasets/prefix_4k.json \
  --base-url http://127.0.0.1:30000 \
  --logprob-atol 0.1
```

### 10K Prefix

```bash
python mytest/run_prefix_case.py mytest/datasets/prefix_10k.json \
  --base-url http://127.0.0.1:30000 \
  --logprob-atol 0.1
```

### 16K Prefix

```bash
python mytest/run_prefix_case.py mytest/datasets/prefix_16k.json \
  --base-url http://127.0.0.1:30000 \
  --logprob-atol 0.1
```

也可以按长度顺序一次执行全部测试：

```bash
for size in 1k 2k 4k 10k 16k; do
  python mytest/run_prefix_case.py "mytest/datasets/prefix_${size}.json" \
    --base-url http://127.0.0.1:30000 \
    --logprob-atol 0.1 || break
done
```

建议保持这个从短到长的顺序。16K cold dense/warm 使用 TorchNative SDPA，明显慢于短 case；若当前 PyTorch 回退到 math SDPA，也可能出现显存不足。该问题发生在 dense baseline，而不是 Radix prefix hit 本身。

成功时每个 suffix 会输出类似：

```text
[prefix_1k/suffix_1] PASS cached_tokens=1024 max_abs_logprob_diff=...
```

## 4. 只检查数据文件

不启动服务时，可以先验证数据长度与结构：

```bash
for size in 1k 2k 4k 10k 16k; do
  python mytest/run_prefix_case.py "mytest/datasets/prefix_${size}.json" --validate-only
done
```

## 5. 测试 fixed_chunk

停止服务，将启动命令中的两项改为：

```text
--sparse-policy fixed_chunk
--selection-unit-size 16
```

重新启动后复用第 3 节的测试命令。`ratio=1.0` 时 `token_h2o` 和 `fixed_chunk` 都应通过 dense 对齐。

## 6. 停止服务

在终端 1 按 `Ctrl+C`。
