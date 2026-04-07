# Skill: Verify (验证与回写)

**目标**：验证优化效果，确保模型精度不受影响，然后安全回写代码。

---

## 核心原则

**任何优化都必须经过验证才能回写！**

验证包括两个维度：
1. **性能验证**：吞吐量是否有提升
2. **保真度验证**：模型精度是否保持不变

---

## Step 1: 性能验证 (吞吐量)

### 基准测试

```python
# 在训练脚本中添加吞吐量监控
import time

class ThroughputMonitor:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.times = []
        self.start_time = time.perf_counter()
        self.samples_seen = 0
    
    def step(self, batch_size):
        self.samples_seen += batch_size
        self.times.append(time.perf_counter())
        
        if len(self.times) > self.window_size:
            elapsed = self.times[-1] - self.times[0]
            throughput = self.window_size * batch_size / elapsed
            print(f"[TRAINOPT_THROUGHPUT] {throughput:.2f} samples/sec")

# 在训练循环中使用
monitor = ThroughputMonitor(window_size=10)

for batch in dataloader:
    # ... training code ...
    monitor.step(batch_size=batch[0].shape[0])
```

### 对比结果

```bash
# 运行相同 step 数的对比
python train_baseline.py --steps 100 > baseline.txt
python train_optimized.py --steps 100 > optimized.txt

# 提取吞吐量
grep "TRAINOPT_THROUGHPUT" baseline.txt | awk '{sum+=$3} END {print "Baseline:", sum/NR}'
grep "TRAINOPT_THROUGHPUT" optimized.txt | awk '{sum+=$3} END {print "Optimized:", sum/NR}'
```

### 验收标准

吞吐量提升幅度视具体场景而定：

| 场景 | 预期提升 |
|------|---------|
| IO 瓶颈明显 (num_workers=0) | 2-5x |
| GPU 利用率低 | 20-50% |
| 已有较好优化 | 5-20% |

**关键是：用户根据实际需求判断是否接受。**

---

## Step 2: 保真度验证 (模型精度)

### 对比 Loss 曲线

```python
# 提取 Loss 日志
import re

def extract_losses(log_file):
    pattern = r"loss[:\s]+([0-9.]+)"
    matches = re.findall(pattern, open(log_file).read())
    return [float(m) for m in matches]

baseline_losses = extract_losses("baseline.txt")
optimized_losses = extract_losses("optimized.txt")

# 计算差异
import numpy as np

max_diff = max(abs(b - o) for b, o in zip(baseline_losses, optimized_losses))
mean_diff = np.mean([abs(b - o) for b, o in zip(baseline_losses, optimized_losses)])

print(f"Max Loss Difference: {max_diff:.6f}")
print(f"Mean Loss Difference: {mean_diff:.6f}")

# 验收
if mean_diff < 0.001:
    print("✓ Fidelity PASSED")
else:
    print("✗ Fidelity FAILED - Optimization may have affected model accuracy")
```

### 对比梯度分布 (可选，更严格)

```python
# 在训练中添加梯度监控
def get_gradient_norms(model):
    norms = []
    for param in model.parameters():
        if param.grad is not None:
            norms.append(param.grad.norm().item())
    return norms

# 对比基线和优化版的梯度
baseline_grads = get_gradient_norms(baseline_model)
optimized_grads = get_gradient_norms(optimized_model)

# 计算余弦相似度
from numpy.linalg import norm

cos_sim = np.dot(baseline_grads, optimized_grads) / (norm(baseline_grads) * norm(optimized_grads))
print(f"Gradient Cosine Similarity: {cos_sim:.6f}")

# 验收
if cos_sim > 0.999:
    print("✓ Gradient fidelity PASSED")
else:
    print("✗ Gradient fidelity FAILED")
```

---

## Step 3: 用户确认

在回写之前，生成报告并等待用户确认：

```
[Optimization Report]
=====================

Change: Increased num_workers from 0 to 8, prefetch_factor=4

Performance:
- Baseline Throughput: 120 samples/sec
- Optimized Throughput: 180 samples/sec
- Improvement: +50%

Fidelity:
- Max Loss Difference: 0.0002
- Mean Loss Difference: 0.0001
- Gradient Similarity: 0.9998
- Status: PASSED

Decision Required:
□ Approve - Merge to main
□ Reject - Rollback changes
□ More Testing - Run additional validation
```

---

## Step 4: 回写代码

**仅在用户确认批准后执行：**

```bash
# 方式 1: 手动复制
cp /tmp/trainflashagent_sandboxes/project/train.py /path/to/original/project/

# 方式 2: 使用 rsync (推荐)
rsync -av --exclude '__pycache__' \
      /tmp/trainflashagent_sandboxes/project/ \
      /path/to/original/project/

# 方式 3: git diff 审查后合并
cd /path/to/original/project/
git diff /tmp/trainflashagent_sandboxes/project/
# 审查差异后
git add train.py
git commit -m "perf: optimize DataLoader with num_workers=8, prefetch_factor=4"
```

---

## 回滚方案

如果优化后发现问题：

```bash
# 1. 立即停止训练
# 2. 恢复原代码
cd /path/to/original/project/
git checkout train.py  # 如果有 git 版本控制

# 或从备份恢复
cp /backup/train.py.backup /path/to/original/project/train.py
```

---

## 验证完成模板

```
[Verification Complete]
Date: 2026-04-07

Performance Gain: +50%
Fidelity Status: PASSED
Decision: APPROVED

Changes Merged:
- train.py: DataLoader optimization

Next Steps:
□ Monitor production training for stability
□ Document the optimization for future reference
□ Consider applying similar optimizations to other projects
```
