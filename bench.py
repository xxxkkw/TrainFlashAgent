import time
import requests
import concurrent.futures
import statistics
import datetime

# ================= 配置区域 =================
# 对应你 vllm serve 的启动参数
API_URL = "http://localhost:11065/v1/chat/completions"
MODEL_NAME = "gemma" 

# 测试参数
MAX_TOKENS = 128000       # 要求的输出长度
TIMEOUT = 120000          # 超时时间(秒)，长文本生成需要较长时间
# 测试的并发层级列表，覆盖 1-50
CONCURRENCY_LEVELS = [10] 
# ===========================================

def send_request(req_id):
    """
    发送单个请求并返回性能指标
    """
    # 构造 Prompt：要求模型写长文，以尽量填满 4096 token
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "请生成超长文本，并持续不断地输出，直到被最大 token 限制强行截断为止。不要总结，不要收尾，不要写结论，不要说即将结束，也不要主动停止。每一段都要继续展开，保持高密度细节，持续扩写。请写一部关于戴森球文明的超长硬科幻史诗，覆盖其物理基础、能源系统、栖居结构、社会治理、经济体系、教育制度、宗教信仰、语言演化、军事力量、AI 基础设施、交通网络、医疗技术、艺术文化、家庭结构、历史时代、科学突破、生态管理、阶层矛盾、外部威胁，以及一场席卷整个文明的大危机。使用丰富描写、具体技术细节、明确的人物、机构、地点、时间线和多层次世界观。每当写完一个重大事件，立刻继续扩展它带来的连锁后果、支线故事、政治反应、工程应对、军事升级和个人命运。永远不要真正结束，要不断引入新的复杂局势、新发现、新档案、新派系和新的战略困境，让文本始终保持超长、连续、无收束地延伸。"}
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
        "ignore_eos": True # 如果 vLLM 支持，可以开启此项强制生成直到 max_tokens
    }
    
    headers = {"Content-Type": "application/json"}
    
    start_time = time.time()
    result = {
        "id": req_id,
        "success": False,
        "tokens": 0,
        "duration": 0,
        "speed": 0,
        "error": None
    }

    try:
        # 发送请求
        response = requests.post(API_URL, json=payload, headers=headers, timeout=TIMEOUT)
        
        # 计算耗时
        duration = time.time() - start_time
        result["duration"] = duration
        
        if response.status_code == 200:
            data = response.json()
            # 获取生成的 token 数 (OpenAI 格式通常在 usage.completion_tokens)
            usage = data.get('usage', {})
            completion_tokens = usage.get('completion_tokens', 0)
            
            result["success"] = True
            result["tokens"] = completion_tokens
            # 单次请求的速度 (Tokens / Duration)
            result["speed"] = completion_tokens / duration if duration > 0 else 0
        else:
            result["error"] = f"Status {response.status_code}: {response.text[:100]}"
            
    except Exception as e:
        result["duration"] = time.time() - start_time
        result["error"] = str(e)
        
    return result

def run_benchmark(concurrency):
    """
    运行指定并发数的测试
    """
    print(f"\n" + "="*60)
    print(f"🚀 开始测试并发数: {concurrency}")
    print(f"   目标输出长度: {MAX_TOKENS}")
    print("="*60)
    
    start_bench = time.time()
    results = []
    
    # 使用线程池并发发送请求
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request, i) for i in range(concurrency)]
        
        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results.append(res)
            completed_count += 1
            
            # 打印简略进度
            status_icon = "✅" if res['success'] else "❌"
            if res['success']:
                print(f"   [{completed_count}/{concurrency}] {status_icon} 完成: {res['tokens']} tokens | 耗时: {res['duration']:.2f}s | 速度: {res['speed']:.2f} t/s")
            else:
                print(f"   [{completed_count}/{concurrency}] {status_icon} 失败: {res['error']}")

    total_bench_time = time.time() - start_bench
    
    # --- 统计与计算 ---
    successful_requests = [r for r in results if r['success']]
    
    if not successful_requests:
        print("\n⚠️  本轮测试全部失败，跳过统计。")
        return

    # 1. 总 Token 数
    total_tokens = sum(r['tokens'] for r in successful_requests)
    
    # 2. 系统总吞吐量 (Total Token Speed) = 总 Token / 总墙钟时间
    #    这代表了服务端在并发压力下每秒能处理多少 Token
    system_throughput = total_tokens / total_bench_time
    
    # 3. 平均单次请求速度 (Average Speed per Request)
    #    计算所有请求速度的平均值
    avg_speed_per_req = statistics.mean(r['speed'] for r in successful_requests)
    
    # 4. 平均延迟
    avg_latency = statistics.mean(r['duration'] for r in successful_requests)

    print(f"\n📊 并发 {concurrency} 测试结果:")
    print(f"   ✅ 成功请求:       {len(successful_requests)}/{concurrency}")
    print(f"   ⏱️  总耗时 (Wall):   {total_bench_time:.2f} s")
    print(f"   📦 总生成 Tokens:  {total_tokens}")
    print("-" * 30)
    print(f"   ⚡ 系统总吞吐速度: {system_throughput:.2f} tokens/s (并发总和)")
    print(f"   🏎️  单请求平均速度: {avg_speed_per_req:.2f} tokens/s")
    print(f"   🐢 平均响应延迟:   {avg_latency:.2f} s")

def main():
    print(f"开始性能压测 -> {API_URL}")
    print(f"模型: {MODEL_NAME}")
    while(1):
        for c in CONCURRENCY_LEVELS:
            run_benchmark(c)
            # 每轮测试后冷却几秒，避免请求积压影响下一轮
            print("\n... 冷却 5 秒 ...")
            time.sleep(5)

if __name__ == "__main__":
    main()
