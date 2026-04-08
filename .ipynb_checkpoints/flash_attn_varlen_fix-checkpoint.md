# FlashAttention Varlen 混合 Prefill/Decode 修复说明

## 1. 目标

这次修复的目标有四个：

1. 继续使用 `flash_attn` 库。
2. 混合 `prefill/decode` 仍然统一走 `flash_attn_varlen_func`。
3. 不在 `prepare()` 中拆成两阶段准备。
4. 修正 `prepare`、warmup、CUDA Graph 三条路径的元数据语义和约束。

也就是说，核心不是“换 attention 实现”，而是把这一套 varlen attention 相关元数据的含义统一起来，并保证它们在不同执行路径下都合法。

---

## 2. 核心问题

最初的问题主要有三类。

### 2.1 `prepare()` 中的元数据没有严格对齐

混合 batch 场景下，下面这些变量必须同时对齐：

- `input_ids`
- `positions`
- `cu_seqlens_q`
- `cu_seqlens_k`
- `max_seqlen_q`
- `max_seqlen_k`
- `slot_mapping`
- `context_lens`
- `block_tables`

原来的 `prepare()` 里，`slot_mapping` 是按 decode/prefill 分支分别拼的，其中 prefill 分支又按 block 段来推导。这样在 chunk 跨 block 时很容易出现：

- token 个数和 `cu_seqlens_q[-1]` 不一致
- token 顺序和 `input_ids`、`positions` 不一致
- 逻辑 block index 和物理 block id 混用
- 最后一块 token 漏算或多算

一旦这些变量失配，`flash_attn_varlen_func` 虽然还能被调用，但实际语义就已经错了。

### 2.2 `is_decode` 切换过早

原来只要 `block_table` 分配满，就会提前把 `seq.is_decode = True`。

但这在语义上是不对的。

“已经分配好块”不等于“当前 prompt 的最后一个 prefill chunk 已经真正完成 forward”。如果切换过早，最后一个 prefill chunk 会被错误地当成 decode，只输入 1 个 token，后面的 logits、采样和输出都会变乱。

### 2.3 warmup 和 CUDA Graph 没有适配 varlen/paged-KV 约束

- warmup 构造出来的是裸 `Sequence`，没有 `block_table`
- 但 `prepare()` 却无条件按 `block_table` 构造 `slot_mapping`
- CUDA Graph capture / replay 只准备了 `slot_mapping/context_lens/block_tables`
- 但 `flash_attn_varlen_func` 实际还依赖 `cu_seqlens_q/cu_seqlens_k/max_seqlen_*`

所以：

- warmup 会直接因为空 `block_table` 报 `IndexError`
- graph 路径后续会因为 varlen 元数据不完整而失败

---

## 3. 对齐的核心

这部分是最关键的。

### 3.1 什么叫“对齐”

对 batch 中第 `i` 个序列，必须同时满足：

```text
q_len_i = cu_seqlens_q[i+1] - cu_seqlens_q[i]
k_len_i = cu_seqlens_k[i+1] - cu_seqlens_k[i]
```

并且还要满足：

```text
len(input_ids) == len(positions) == len(slot_mapping) == cu_seqlens_q[-1]
len(context_lens) == batch_size
len(cu_seqlens_q) == len(cu_seqlens_k) == batch_size + 1
```

如果这些关系不成立，`flash_attn_varlen_func` 的输入虽然类型可能没问题，但语义一定不对。

### 3.2 每个变量的语义

#### `input_ids`

本轮真正送入模型的 query token。

#### `positions`

这些 query token 的绝对位置。

#### `cu_seqlens_q`

每个序列本轮 query token 数的前缀和。

#### `cu_seqlens_k`

每个序列本轮可见 key 长度的前缀和。

#### `context_lens`

每个序列的真实上下文长度。

#### `slot_mapping`

当前 query token 对应写入 paged KV cache 的物理 slot。

#### `block_tables`

每个序列的逻辑 block 到物理 block 的映射。

### 3.3 本次修复后的核心策略

`prepare()` 不再把 decode 和 prefill 分成两个准备阶段，而是统一按“当前 chunk 中每一个 token”的顺序构造元数据。

也就是说：

- `input_ids` 的顺序
- `positions` 的顺序
- `slot_mapping` 的顺序

三者必须严格一一对应。

修复后的关键写法是：

```python
slot_mapping.extend(
    seq.block_table[token_idx // self.block_size] * self.block_size + token_idx % self.block_size
    for token_idx in range(chunk_start_idx, end)
)
```

这个写法的优点是：

1. 直接和 `input_ids.extend(seq.token_ids[chunk_start_idx:end])` 保持同序。
2. 直接和 `positions.extend(range(chunk_start_idx, end))` 保持同序。
3. 不需要额外区分 decode/prefill 的 token 填充方式。
4. chunk 跨 block 时也不会漏 token 或乱序。

---

## 4. `prepare()` 的具体修改

文件位置：

- [engine/model_runner.py](/gz-data/m_nano_vllm/engine/model_runner.py#L150)

### 4.1 修改前的问题

旧实现有两层问题：

1. `slot_mapping` 的 decode/prefill 两套拼法不一致。
2. warmup/no-cache 路径也会强行使用 `block_table`。

### 4.2 修改后的思路

新增了一个判定：

```python
use_paged_kv = all(
    len(seq.block_table) > (seq.chunk_start_idx + seq.chunk_size - 1) // self.block_size
    for seq in scheduledBatch
)
```

这个判定的含义是：

- 如果当前 batch 的每个序列，对应 chunk 覆盖到的 block 都已经分配好，就走 paged KV。
- 否则仍然走 `flash_attn_varlen_func`，但不使用 paged KV，不访问 `block_table`。

### 4.3 代码示例

```python
def prepare(self, scheduledBatch: list[Sequence]):
    input_ids = []
    positions = []
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    max_seqlen_q = 0
    max_seqlen_k = 0
    slot_mapping = []
    is_decodes = []
    context_lens = []

    use_paged_kv = all(
        len(seq.block_table) > (seq.chunk_start_idx + seq.chunk_size - 1) // self.block_size
        for seq in scheduledBatch
    )

    for seq in scheduledBatch:
        chunk_start_idx = seq.chunk_start_idx
        num_tokens = seq.chunk_size
        end = chunk_start_idx + num_tokens
        context_len = end if use_paged_kv else num_tokens

        input_ids.extend(seq.token_ids[chunk_start_idx:end])
        positions.extend(range(chunk_start_idx, end))

        if use_paged_kv:
            slot_mapping.extend(
                seq.block_table[token_idx // self.block_size] * self.block_size + token_idx % self.block_size
                for token_idx in range(chunk_start_idx, end)
            )
        else:
            slot_mapping.extend([-1] * num_tokens)

        cu_seqlens_q.append(cu_seqlens_q[-1] + num_tokens)
        cu_seqlens_k.append(cu_seqlens_k[-1] + context_len)
        max_seqlen_q = max(max_seqlen_q, num_tokens)
        max_seqlen_k = max(max_seqlen_k, context_len)
        context_lens.append(context_len)
        is_decodes.append(seq.is_decode)

    assert len(input_ids) == len(positions) == len(slot_mapping) == cu_seqlens_q[-1]
    assert len(cu_seqlens_q) == len(cu_seqlens_k) == len(context_lens) + 1

    block_tables = self.prepare_block_tables(scheduledBatch) if use_paged_kv else None
```

### 4.4 这里的关键点

#### `context_len = end if use_paged_kv else num_tokens`

- paged KV 模式下，key 的可见长度是整个上下文长度 `end`
- 无 cache 模式下，`k/v` 就是本轮输入本身，所以 key 长度只能是 `num_tokens`

这一步很关键。如果这里写错，`cu_seqlens_k` 和真实 `k/v` 张量长度会不匹配。

---

## 5. `is_decode` 切换时机的修正

文件位置：

- [engine/scheduler.py](/gz-data/m_nano_vllm/engine/scheduler.py#L108)
- [engine/block_manager.py](/gz-data/m_nano_vllm/engine/block_manager.py#L111)

### 5.1 原来的错误时机

以前在以下位置会提前切换到 decode：

- `BlockManager.allocate()`
- `Scheduler.schedule_running()`
- `Scheduler.schedule_waiting()`

判断依据是：

```python
if len(seq.block_table) == seq.num_blocks:
    seq.is_decode = True
```

这实际上只是“块分配完了”，不是“prefill 已经实际执行完”。

### 5.2 修正后的原则

`is_decode` 只能在本轮 forward 结束后切换。

也就是说，最后一个 prefill chunk 仍然要按 prefill 方式送入 attention；等这轮算完，才把状态切到 decode，以便下一轮开始按 decode 处理。

### 5.3 修改后的代码示例

```python
def postprocess(self, scheduledBatch: list[Sequence], token_ids: list[int]):
    for seq, token_id in zip(scheduledBatch, token_ids):
        chunk_size = seq.chunk_size
        finished_prefill = (not seq.is_decode) and (seq.num_computed_tokens + chunk_size == seq.num_tokens)
        seq.num_computed_tokens += chunk_size
        if finished_prefill:
            seq.is_decode = True
        if seq.is_decode:
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
```

### 5.4 这样改的好处

1. 最后一个 prefill chunk 不会被错误当成 decode。
2. 第一轮生成 token 仍然来自 prefill 最后一个位置的 logits。
3. 下一轮才是真正的 decode。
4. 混合 prefill/decode batch 里，每个序列的状态语义一致。

---

## 6. warmup 路径需要注意的事项

文件位置：

- [engine/model_runner.py](/gz-data/m_nano_vllm/engine/model_runner.py#L60)

warmup 是这次修复里非常重要的一条特殊路径。

### 6.1 warmup 的特点

warmup 构造的 `Sequence`：

- 只是为了初始化模型、预热 kernel、统计显存
- 没经过 scheduler
- 没经过 block manager
- 没有 `block_table`

所以它天然不能假设自己已经有 paged KV cache。

### 6.2 原来的问题

旧逻辑：

```python
num_seqs = min(max_num_batch_tokens // max_model_len, self.config.max_num_seqs)
seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
```

这里有两个风险：

1. `max_num_batch_tokens < max_model_len` 时，`num_seqs` 会变成 0。
2. 构造出来的 seq 没 `block_table`，但 `prepare()` 之前会默认按 paged KV 去访问它。

### 6.3 修复后的写法

```python
def warmup_model(self):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    max_num_batch_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
    warmup_len = min(max_num_batch_tokens, max_model_len)
    num_seqs = min(max(max_num_batch_tokens // warmup_len, 1), self.config.max_num_seqs)
    seqs = [Sequence([0] * warmup_len) for _ in range(num_seqs)]
    self.run(seqs)
```

### 6.4 warmup 的注意事项总结

#### 注意事项 1：不能假设有 cache

warmup 时必须允许 `block_tables=None`。

#### 注意事项 2：不能构造空 batch

`num_seqs` 至少要是 1。

#### 注意事项 3：warmup 长度不能强制等于 `max_model_len`

如果 `max_num_batched_tokens < max_model_len`，就应该退化成更短的 warmup 长度。

#### 注意事项 4：warmup 只需要让路径合法，不需要模拟真实调度状态

它的作用是预热，不是复刻调度器完整语义。

---

## 7. CUDA Graph 如何适配

文件位置：

- [engine/model_runner.py](/gz-data/m_nano_vllm/engine/model_runner.py#L205)
- [engine/model_runner.py](/gz-data/m_nano_vllm/engine/model_runner.py#L246)

### 7.1 为什么需要单独适配

`flash_attn_varlen_func` 依赖的不只是 `q/k/v` 张量本身，还依赖：

- `cu_seqlens_q`
- `cu_seqlens_k`
- `max_seqlen_q`
- `max_seqlen_k`
- `block_table`

原来的 graph capture 里，只给了：

- `slot_mapping`
- `context_lens`
- `block_tables`

这对 varlen attention 来说是不够的。

### 7.2 适配原则

CUDA Graph 只适合“形状稳定”的路径。

混合 prefill/decode 的 batch 不稳定，不能强行 graph 化。

因此这次的策略是：

- 混合 batch：eager path，仍然走 `flash_attn_varlen_func`
- 纯 decode batch：graph path，同样走 `flash_attn_varlen_func`

也就是说，attention kernel 还是一个，只是执行模式不同。

### 7.3 replay 只对纯 decode 生效

代码示例：

```python
context = get_context()
use_decode_graph = (
    not self.enforce_eager
    and input_ids.size(0) <= 512
    and context.is_decodes is not None
    and bool(context.is_decodes.all())
    and context.block_tables is not None
)

if not use_decode_graph:
    return self.model.compute_logits(self.model(input_ids, position))
```

这段逻辑的含义是：

- 如果不是纯 decode，就不要 replay graph
- 混合 batch 继续走 eager
- 避免 graph shape 和元数据语义失配

### 7.4 capture 时要准备哪些静态 buffer

这次补充了：

```python
input_ids = torch.zeros(max_bs, dtype=torch.int64)
positions = torch.zeros(max_bs, dtype=torch.int64)
slot_mapping = torch.arange(max_bs, dtype=torch.int32) * self.block_size
cu_seqlens_q = torch.arange(max_bs + 1, dtype=torch.int32)
cu_seqlens_k = torch.arange(max_bs + 1, dtype=torch.int32)

context_lens = torch.ones(config.max_num_seqs, dtype=torch.int32)
block_tables = torch.full((config.max_num_seqs, max_num_blocks), -1, dtype=torch.int32)
block_tables[:, 0] = torch.arange(config.max_num_seqs, dtype=torch.int32)
is_decodes = torch.ones(config.max_num_seqs, dtype=torch.bool)
```

### 7.5 capture 时设置 context

```python
set_context(
    1, config.max_model_len,
    cu_seqlens_q[:bs + 1], cu_seqlens_k[:bs + 1], context_lens[:bs],
    slot_mapping[:bs], block_tables[:bs], is_decodes[:bs]
)
```

这里有两个重点：

1. decode graph 的 `max_seqlen_q` 固定为 1。
2. graph capture 时用的是“合法占位元数据”，真正 replay 时再覆盖成真实值。

### 7.6 replay 前如何更新动态值

```python
graph_vars["slot_mapping"].fill_(-1)
graph_vars["slot_mapping"][:bs] = context.slot_mapping

graph_vars["context_lens"].zero_()
graph_vars["context_lens"][:bs] = context.context_lens

graph_vars["cu_seqlens_k"].fill_(context.cu_seqlens_k[-1])
graph_vars["cu_seqlens_k"][:bs + 1] = context.cu_seqlens_k

graph_vars["block_tables"].fill_(-1)
graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
```

这一步的本质是：

- graph 的 buffer 是静态分配的
- replay 前把这次 decode batch 的真实元数据原地拷进去
- 不重新分配、不改变形状，只改内容

### 7.7 CUDA Graph 适配的关键原则总结

#### 原则 1：只 graph 纯 decode

因为 decode 的 `q_len=1`，shape 最稳定。

#### 原则 2：capture 时必须给 varlen attention 提供完整合法元数据

不能只准备 `slot_mapping/block_tables`。

#### 原则 3：replay 前只做原地写入

不要重新创建张量。

#### 原则 4：graph 和 eager 共用同一个 attention kernel

两者都还是 `flash_attn_varlen_func`，只是调度方式不同。

---

## 8. 示例：混合 batch 的对齐关系

假设有两个序列：

- `seq0`：prefill chunk，从位置 `128` 开始，chunk 长度 `256`
- `seq1`：decode，当前只输入 1 个 token，位置是 `19`

那么：

```text
seq0:
  chunk_start_idx = 128
  num_tokens = 256
  end = 384

seq1:
  chunk_start_idx = 19
  num_tokens = 1
  end = 20
```

则应该得到：

```text
cu_seqlens_q = [0, 256, 257]
context_lens = [384, 20]
cu_seqlens_k = [0, 384, 404]
```

并且：

- `input_ids[0:256]` 是 `seq0` 的 chunk token
- `positions[0:256]` 是 `128..383`
- `slot_mapping[0:256]` 必须严格对应这 256 个 token
- `input_ids[256]` / `positions[256]` / `slot_mapping[256]` 对应 `seq1`

如果这里任何一组顺序不一致，attention 结果就会错。

---

## 9. 顺手修掉的其它问题

### 9.1 `SamplingParams` 复制 bug

文件位置：

- [engine/llm_engine.py](/gz-data/m_nano_vllm/engine/llm_engine.py#L91)

原来是：

```python
sampling_params = sampling_params * len(prompts)
```

这是错误的，因为 `SamplingParams` 不是可乘对象。

修复后：

```python
sampling_params = [sampling_params] * len(prompts)
```

### 9.2 `example.py` 重复初始化模型

文件位置：

- [example.py](/gz-data/m_nano_vllm/example.py#L7)

原来的示例脚本里会重复初始化 `LLM`，容易导致显存重复占用和后续误判。

现在只保留一条初始化路径。

---

## 10. 总结

这次修复的核心，不是换 attention 实现，而是统一 varlen attention 的元数据语义，并把这套语义贯穿到三条路径：

1. 普通 eager 路径
2. warmup 路径
3. CUDA Graph decode 路径

最终原则可以概括成三句话：

1. 混合 `prefill/decode` 仍然统一走 `flash_attn_varlen_func`。
2. `prepare()` 不拆两阶段，而是按 token 顺序统一构造所有元数据。
3. warmup 和 CUDA Graph 不是额外逻辑，它们也必须遵守同一套 varlen/paged-KV 元数据语义。

---

## 11. 当前代码参考位置

- `prepare()` 主逻辑：
  [engine/model_runner.py](/gz-data/m_nano_vllm/engine/model_runner.py#L150)

- warmup：
  [engine/model_runner.py](/gz-data/m_nano_vllm/engine/model_runner.py#L60)

- decode graph replay 判定：
  [engine/model_runner.py](/gz-data/m_nano_vllm/engine/model_runner.py#L205)

- CUDA Graph capture：
  [engine/model_runner.py](/gz-data/m_nano_vllm/engine/model_runner.py#L246)

- `is_decode` 切换时机：
  [engine/scheduler.py](/gz-data/m_nano_vllm/engine/scheduler.py#L108)

- 取消过早切换 decode：
  [engine/block_manager.py](/gz-data/m_nano_vllm/engine/block_manager.py#L111)
