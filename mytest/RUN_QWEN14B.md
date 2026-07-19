# Qwen2.5-14B Sparse Prefix 测试命令

本说明在单张 A800 上同时运行两个 Qwen2.5-14B-Instruct 服务：

- `127.0.0.1:30000`：`sparse_prefill`，prefix hit 进入 exact sparse Triton；
- `127.0.0.1:30001`：纯 dense Triton，提供相同 prefix-hit 执行分区的数值基线。

测试不再用 cold full-prefill 作为 sparse 的直接数值基线。dense 与 sparse 服务分别建立相同 prefix cache，再比较两次 prefix-hit 的 token 和 logprob。

## 1. 进入环境

每个终端都先执行：

```bash
cd /home/zhenbao/Pro/SparsePrefill/sglangSparsePrefill
conda activate sglang-prefill
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH=/data0/zhenbao/weight/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8
```

确认加载的是当前仓库：

```bash
python -c "import importlib.util; print(importlib.util.find_spec('sglang').origin)"
```

期望输出：

```text
/home/zhenbao/Pro/SparsePrefill/sglangSparsePrefill/python/sglang/__init__.py
```

## 2. 启动 sparse 服务

在终端 1 执行：

```bash
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port 30000 \
  --dtype bfloat16 \
  --tp-size 1 \
  --dp-size 1 \
  --pp-size 1 \
  --random-seed 0 \
  --max-total-tokens 20000 \
  --max-running-requests 1 \
  --attention-backend triton \
  --prefill-attention-backend sparse_prefill \
  --decode-attention-backend triton \
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

`sparse-ratio=1.0` 让 selector 为每个 suffix query 选择完整 prefix，但请求仍进入 exact sparse kernel。

## 3. 启动 dense 基线服务

在终端 2 执行：

```bash
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port 30001 \
  --dtype bfloat16 \
  --tp-size 1 \
  --dp-size 1 \
  --pp-size 1 \
  --random-seed 0 \
  --max-total-tokens 20000 \
  --max-running-requests 1 \
  --attention-backend triton \
  --page-size 1 \
  --chunked-prefill-size -1 \
  --cuda-graph-backend-prefill disabled \
  --cuda-graph-backend-decode disabled \
  --disable-overlap-schedule
```

等待两个服务完成加载：

```bash
curl -f http://127.0.0.1:30000/health
curl -f http://127.0.0.1:30001/health
```

两个服务都显式限制 `max-total-tokens=20000`。单张 80 GiB A800 可以同时容纳两份模型和最长 16K case 的 KV cache。

## 4. 测试五组 Prefix

在终端 3 进入第 1 节的环境。每个 suffix 的测试流程为：

```text
dense flush
→ dense warm(prefix, max_new_tokens=0)
→ dense prefix-hit(prefix + suffix)

sparse flush
→ sparse warm(prefix, max_new_tokens=0)
→ sparse prefix-hit(prefix + suffix)

→ 检查两边 cached_tokens == prefix_length
→ 比较 dense-hit 与 sparse-hit 的 output token 和 logprob
→ flush 两个服务
```

### 1K Prefix

```bash
python mytest/run_prefix_case.py mytest/datasets/prefix_1k.json \
  --dense-base-url http://127.0.0.1:30001 \
  --sparse-base-url http://127.0.0.1:30000 \
  --logprob-atol 0.1
```

### 2K Prefix

```bash
python mytest/run_prefix_case.py mytest/datasets/prefix_2k.json \
  --dense-base-url http://127.0.0.1:30001 \
  --sparse-base-url http://127.0.0.1:30000 \
  --logprob-atol 0.1
```

### 4K Prefix

```bash
python mytest/run_prefix_case.py mytest/datasets/prefix_4k.json \
  --dense-base-url http://127.0.0.1:30001 \
  --sparse-base-url http://127.0.0.1:30000 \
  --logprob-atol 0.1
```

### 10K Prefix

```bash
python mytest/run_prefix_case.py mytest/datasets/prefix_10k.json \
  --dense-base-url http://127.0.0.1:30001 \
  --sparse-base-url http://127.0.0.1:30000 \
  --logprob-atol 0.1
```

### 16K Prefix

```bash
python mytest/run_prefix_case.py mytest/datasets/prefix_16k.json \
  --dense-base-url http://127.0.0.1:30001 \
  --sparse-base-url http://127.0.0.1:30000 \
  --logprob-atol 0.1
```

也可以按长度顺序一次执行全部测试：

```bash
for size in 1k 2k 4k 10k 16k; do
  python mytest/run_prefix_case.py "mytest/datasets/prefix_${size}.json" \
    --dense-base-url http://127.0.0.1:30001 \
    --sparse-base-url http://127.0.0.1:30000 \
    --logprob-atol 0.1 || break
done
```

成功时输出类似：

```text
[prefix_1k/suffix_1] dense warm + prefix hit
[prefix_1k/suffix_1] sparse warm + prefix hit
[prefix_1k/suffix_1] PASS cached_tokens=1024 max_abs_logprob_diff=...
```

## 5. 只检查数据文件

不启动服务时执行：

```bash
for size in 1k 2k 4k 10k 16k; do
  python mytest/run_prefix_case.py "mytest/datasets/prefix_${size}.json" --validate-only
done
```

## 6. 测试 fixed_chunk

停止 sparse 服务，将以下参数改为：

```text
--sparse-policy fixed_chunk
--selection-unit-size 16
```

保持 `--sparse-ratio 1.0` 并重新启动 sparse 服务。dense 服务不变，复用第 4 节测试命令。

## 7. 停止服务

分别在终端 1 和终端 2 按 `Ctrl+C`。
