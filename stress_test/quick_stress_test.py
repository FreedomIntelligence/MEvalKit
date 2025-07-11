#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEvalKit 快速压力测试工具

用于快速测试压力测试功能是否正常工作
"""

import subprocess
import time
import statistics
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class QuickStressTest:
    def __init__(self):
        self.print_lock = threading.Lock()
    
    def run_single_evaluation(self, task_id: str) -> dict:
        """运行单个评测任务"""
        start_time = time.time()
        
        try:
            # 构建运行命令
            cmd = [
                "python", "run.py",
                "--dataset", "CMMLUMed",
                "--model", "stressTest",
                "--workers", "1",
                "--question_limitation", "5"  # 快速测试，只评测5个问题
            ]
            
            # 运行评测
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30  # 30秒超时
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            success = result.returncode == 0
            
            with self.print_lock:
                status = "✓" if success else "✗"
                print(f"{status} {task_id}: {execution_time:.2f}秒")
            
            return {
                "task_id": task_id,
                "success": success,
                "execution_time": execution_time,
                "error": result.stderr if not success else None
            }
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            with self.print_lock:
                print(f"✗ {task_id} 出错: {str(e)}")
            
            return {
                "task_id": task_id,
                "success": False,
                "execution_time": execution_time,
                "error": str(e)
            }
    
    def test_concurrent_level(self, concurrent_level: int, num_tasks: int = 5):
        """测试特定并发级别"""
        print(f"\n🔄 测试并发数 {concurrent_level} (任务数: {num_tasks})")
        print("-" * 30)
        
        test_start_time = time.time()
        
        # 生成任务ID
        task_ids = [f"task_{i+1}" for i in range(num_tasks)]
        
        # 使用线程池执行并发任务
        task_results = []
        with ThreadPoolExecutor(max_workers=concurrent_level) as executor:
            future_to_task = {
                executor.submit(self.run_single_evaluation, task_id): task_id
                for task_id in task_ids
            }
            
            for future in as_completed(future_to_task):
                task_result = future.result()
                task_results.append(task_result)
        
        test_end_time = time.time()
        total_time = test_end_time - test_start_time
        
        # 统计结果
        successful_tasks = [r for r in task_results if r["success"]]
        if successful_tasks:
            execution_times = [r["execution_time"] for r in successful_tasks]
            avg_time = statistics.mean(execution_times)
            throughput = len(successful_tasks) / total_time
        else:
            avg_time = throughput = 0
        
        success_rate = len(successful_tasks) / len(task_results) * 100
        
        print(f"📊 结果: 成功 {len(successful_tasks)}/{len(task_results)} ({success_rate:.1f}%)")
        if successful_tasks:
            print(f"⏱️  平均用时: {avg_time:.2f}秒")
            print(f"📈 吞吐量: {throughput:.2f} 任务/秒")
        print(f"🕐 总耗时: {total_time:.1f}秒")
        
        return {
            "concurrent_level": concurrent_level,
            "success_rate": success_rate,
            "avg_time": avg_time,
            "throughput": throughput,
            "total_time": total_time
        }
    
    def run_quick_test(self):
        """运行快速测试"""
        print("=" * 50)
        print("MEvalKit 快速压力测试")
        print("=" * 50)
        
        # 测试不同并发级别
        concurrent_levels = [1, 4, 8]
        results = []
        
        for level in concurrent_levels:
            result = self.test_concurrent_level(level, num_tasks=3)
            results.append(result)
        
        # 打印汇总
        print("\n" + "=" * 50)
        print("📋 汇总结果")
        print("=" * 50)
        print(f"{'并发数':<8} {'成功率':<10} {'平均用时':<12} {'吞吐量':<12}")
        print("-" * 45)
        
        for result in results:
            print(f"{result['concurrent_level']:<8} {result['success_rate']:>6.1f}%   {result['avg_time']:>8.2f}s    {result['throughput']:>8.2f}/s")
        
        print("\n✅ 快速测试完成!")

def main():
    """主函数"""
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        quick_test = QuickStressTest()
        quick_test.run_quick_test()
    except KeyboardInterrupt:
        print("\n❌ 测试被用户中断")
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {str(e)}")

if __name__ == "__main__":
    main() 