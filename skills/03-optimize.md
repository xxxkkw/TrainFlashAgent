# Skill: Optimize (工程级优化)

**目标**：根据诊断结果，应用工程级别的优化，**优先解决 IO、数据长尾、资源分配问题**。

---

## 核心原则

**优化优先级（从高到低）：**

1. **数据管道优化** (IO/长尾) - 通常能带来 2-5x 提升
2. **资源分配优化** (batch_size, workers) - 通常能带来 20-50% 提升
3. **代码模式优化** (等价替换) - 通常能带来 10-30% 提升
4. **算子级优化** - **最后手段**，通常收益有限 (<10%)

---

## 场景 1: DataLoader 优化

**触发条件**：诊断报告显示 `DataLoader` 时间占比 > 40%

### 优化策略

#### A. 增加 worker 数量

```python
# 原代码
train_loader = DataLoader(dataset, batch_size=32, num_workers=0)

# 优化后
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,           # ← 增加到 4-8 (根据 CPU 核心数)
    prefetch_factor=4,       # ← 添加预取
    persistent_workers=True, # ← 保持 worker 存活
    pin_memory=True          # ← 启用 GPU 内存锁定
)
```

#### B. 调整 prefetch_factor

```python
# 如果 GPU 经常等待数据
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,
    prefetch_factor=8        # ← 增加到 4-8
)
```

#### C. 处理数据长尾

如果诊断显示 `max >> mean`，说明存在长尾样本：

```python
# 方案 1: 使用 collate_fn 进行动态 padding
def dynamic_collate_fn(batch):
    images, labels = zip(*batch)
    # 动态调整，避免过度 padding
    images = pad_sequence(images, batch_first=True, padding_value=0)
    return images, torch.stack(labels)

train_loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=dynamic_collate_fn
)
```

```python
# 方案 2: 使用 BatchSampler 进行长度排序
from torch.utils.data import BatchSampler

class LengthBasedBatchSampler:
    def __init__(self, dataset, batch_size, shuffle=True):
        # 按长度排序数据
        lengths = [len(item) for item in dataset]
        indices = sorted(range(len(dataset)), key=lambda i: lengths[i])
        self.indices = indices
        self.batch_size = batch_size
    
    def __iter__(self):
        batches = []
        batch = []
        for idx in self.indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                batches.append(batch)
                batch = []
        if batch:
            batches.append(batch)
        return iter(batches)
    
    def __len__(self):
        return len(self.indices) // self.batch_size

train_loader = DataLoader(
    dataset,
    batch_sampler=LengthBasedBatchSampler(dataset, batch_size=32)
)
```

---

## 场景 2: 资源分配优化

**触发条件**：GPU 利用率 < 70% 或 显存使用率 < 80%

### 优化策略

#### A. 增加 batch_size

```python
# 原代码
batch_size = 32

# 优化后 (逐步增加直到显存接近满载)
batch_size = 128  # 或根据显存调整
```

#### B. 启用梯度累积

```python
# 如果 batch_size 太大导致 OOM，使用梯度累积
accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps

for i, (inputs, labels) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### C. 启用混合精度训练

```python
# 原代码
loss = criterion(outputs, labels)
loss.backward()

# 优化后
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 场景 3: 代码模式优化

**触发条件**：诊断显示 Python 侧开销大

### 优化策略

#### A. 向量化替代循环

```python
# 低效
results = []
for item in data:
    results.append(process(item))
results = torch.stack(results)

# 高效 (如果可能)
results = torch.stack([process(item) for item in data])  # 至少减少 append 开销
# 或完全向量化
results = process_batch(data)
```

#### B. 减少 .item() 调用

```python
# 低效 (频繁 CPU-GPU 同步)
for i in range(len(losses)):
    total_loss += losses[i].item()

# 高效
total_loss = losses.sum().item()  # 或直接在 GPU 上计算
```

#### C. 使用 torch.no_grad()

```python
# 验证阶段
with torch.no_grad():  # ← 添加这个
    outputs = model(inputs)
    loss = criterion(outputs, labels)
```

---

## 验证优化效果

每次优化后，必须验证：

```bash
# 1. 测量优化前的吞吐量
python train.py --benchmark --steps 100 > baseline.txt

# 2. 应用优化

# 3. 测量优化后的吞吐量
python train.py --benchmark --steps 100 > optimized.txt

# 4. 对比
# 提取 [TRAINOPT_THROUGHPUT] 值并计算提升
```

### 验收标准

- [ ] 吞吐量提升 > 5%
- [ ] 模型 Loss 曲线与原版一致（误差 < 0.001）
- [ ] 无明显 OOM 或稳定性问题

---

## 优化日志模板

```
[Optimization Log]
Date: 2026-04-07
Skill: 03-optimize.md

Change Applied:
- File: train.py
- Modification: Increased num_workers from 0 to 8, added prefetch_factor=4

Baseline Throughput: 120 samples/sec
Optimized Throughput: 180 samples/sec
Improvement: +50%

Fidelity Check:
- Loss difference: 0.0002 (< 0.001 tolerance)
- Status: PASSED

Decision:
□ Commit to main
□ Rollback (if fidelity failed)
```
