# Skill: Diagnose (宏观诊断)

**目标**：使用自顶向下的方法识别训练瓶颈，**优先使用手动计时器**，而非直接调用 Profiler。

---

## 核心原则

**不要直接调用 torch.profiler 或 Nsight！**

这些工具会产生算子级别的细节（成千上万个 kernel），让 LLM 陷入微观分析，错过真正的工程瓶颈。

**正确的顺序：**
1. **Phase 1**: 手动计时器 → 识别宏观瓶颈（DataLoad/Forward/Backward）
2. **Phase 2**: 工程优化 → 解决 IO、数据长尾、同步阻塞
3. **Phase 3**: 仅当 Phase 2 收益趋近于零时，才调用 Profiler

---

## Step 1: 插入手动计时器

在训练循环的关键位置插入 `time.perf_counter()`。

### 目标位置

```python
# 在 train.py 中找到训练循环
def train_epoch(model, dataloader, optimizer):
    for batch in dataloader:           # ← 在这里插入计时器
        images, labels = batch
        outputs = model(images)        # ← Forward 开始
        loss = criterion(outputs, labels)
        loss.backward()                # ← Backward 开始
        optimizer.step()               # ← Step 结束
```

### 插入代码

```python
import time

def train_epoch(model, dataloader, optimizer):
    for batch_idx, batch in enumerate(dataloader):
        # === [TRAINOPT_TIMER] DataLoader ===
        start_dataloader = time.perf_counter()
        
        images, labels = batch
        
        end_dataloader = time.perf_counter()
        print(f"[TRAINOPT_TIMER] DataLoader: {end_dataloader - start_dataloader:.4f}s")
        # ====================================
        
        # === [TRAINOPT_TIMER] Forward ===
        start_forward = time.perf_counter()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        end_forward = time.perf_counter()
        print(f"[TRAINOPT_TIMER] Forward: {end_forward - start_forward:.4f}s")
        # ==================================
        
        # === [TRAINOPT_TIMER] Backward ===
        start_backward = time.perf_counter()
        
        loss.backward()
        optimizer.step()
        
        end_backward = time.perf_counter()
        print(f"[TRAINOPT_TIMER] Backward: {end_backward - start_backward:.4f}s")
        # ==================================
```

---

## Step 2: 运行并收集日志

```bash
# 运行至少 50-100 个 step
python train.py --steps 100 2>&1 | tee timer_logs.txt
```

### 预期输出

```
[TRAINOPT_TIMER] DataLoader: 0.5234s
[TRAINOPT_TIMER] Forward: 0.1234s
[TRAINOPT_TIMER] Backward: 0.2345s
[TRAINOPT_TIMER] DataLoader: 0.4521s
[TRAINOPT_TIMER] Forward: 0.1198s
...
```

---

## Step 3: 分析瓶颈

### 计算统计

```python
import re
import numpy as np

# 解析日志
pattern = r"\[TRAINOPT_TIMER\]\s+(\w+):\s+([0-9.]+)s"
matches = re.findall(pattern, open("timer_logs.txt").read())

# 按阶段分组
data = {}
for label, value in matches:
    if label not in data:
        data[label] = []
    data[label].append(float(value))

# 计算统计
for label, times in data.items():
    mean = np.mean(times)
    std = np.std(times)
    max_val = np.max(times)
    print(f"{label}: mean={mean:.4f}s, std={std:.4f}s, max={max_val:.4f}s")
    
    # 检测长尾
    if max_val > mean * 2:
        print(f"  ⚠️  WARNING: Long-tail detected! Max is {max_val/mean:.1f}x the mean")
```

### 输出示例

```
DataLoader: mean=0.4821s, std=0.1523s, max=1.2341s
  ⚠️  WARNING: Long-tail detected! Max is 2.6x the mean
Forward: mean=0.1198s, std=0.0012s, max=0.1234s
Backward: mean=0.2234s, std=0.0023s, max=0.2345s
```

---

## 诊断结论模板

根据分析，填写以下结论：

```
[Diagnostic Report]

Time Distribution:
- DataLoader: {X}% (mean={Y}s, std={Z}s)
- Forward: {A}% (mean={B}s)
- Backward: {C}% (mean={D}s)

Bottleneck Identified:
□ DataLoader is the primary bottleneck (>50% of total time)
□ Long-tail detected in {phase} (max > 2x mean)
□ Forward/Backward imbalance
□ Other: _______________

Recommendation:
→ Proceed to Skill: 03-optimize.md with focus on {bottleneck_type}
```

---

## 常见瓶颈模式

| 模式 | 特征 | 下一步 |
|------|------|--------|
| **IO 瓶颈** | DataLoader 时间 > 50% | 优化 `num_workers`, `prefetch_factor` |
| **数据长尾** | DataLoader std 大，max >> mean | 检查数据分布，优化预处理 |
| **GPU 利用率低** | Forward/Backward 时间异常短 | 检查 batch_size, 模型并行 |
| **同步阻塞** | 某阶段时间方差极大 | 检查 CPU-GPU 同步点 |
