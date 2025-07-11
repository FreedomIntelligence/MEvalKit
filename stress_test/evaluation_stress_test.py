#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEvalKit 评测服务压力测试工具

功能：
1. 测试不同并发数量下的评测性能 (1, 4, 8, 16, 32)
2. 统计平均每个评测用时
3. 生成详细的性能报告
"""

import subprocess
import time
import statistics
import json
import os
import argparse
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import threading

class EvaluationStressTest:
    def __init__(self, concurrent_levels: List[int] = None, 
                 tasks_per_level: int = 1, 
                 avg_task_time: float = 5.0):
        """
        初始化压力测试
        
        Args:
            concurrent_levels: 并发数量列表，默认为 [1, 4, 8, 16, 32]
            tasks_per_level: 每个并发级别要运行的任务数
            avg_task_time: 平均任务模拟时间（秒）
        """
        self.concurrent_levels = concurrent_levels or [1, 4, 8, 16, 32]
        self.tasks_per_level = tasks_per_level
        self.avg_task_time = avg_task_time
        self.results = {}
        self.detailed_results = {}
        
        # 创建结果保存目录
        self.results_dir = Path("stress_test_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 线程锁用于安全的输出
        self.print_lock = threading.Lock()
    
    def run_single_evaluation(self, task_id: str, concurrent_level: int) -> Dict[str, Any]:
        """
        运行单个评测任务
        
        Args:
            task_id: 任务ID
            concurrent_level: 当前并发级别
            
        Returns:
            任务执行结果字典
        """
        start_time = time.time()
        
        try:
            # 构建运行命令
            cmd = [
                "python", "run.py",
                "--dataset", "CMMLUMed",
                "--model", "stressTest",
                "--workers", "64",  # 单个任务使用1个worker
                "--question_limitation", "1000"  # 每个任务评测10个问题
            ]
            
            # 运行评测
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 60秒超时
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            success = result.returncode == 0
            
            # 安全地打印进度
            with self.print_lock:
                status = "✓" if success else "✗"
                print(f"{status} 完成任务 {task_id}: {execution_time:.2f}秒 (并发级别: {concurrent_level})")
            
            return {
                "task_id": task_id,
                "concurrent_level": concurrent_level,
                "success": success,
                "execution_time": execution_time,
                "start_time": start_time,
                "end_time": end_time,
                "error": result.stderr if not success else None
            }
            
        except subprocess.TimeoutExpired:
            end_time = time.time()
            execution_time = end_time - start_time
            
            with self.print_lock:
                print(f"✗ 任务 {task_id} 超时: {execution_time:.2f}秒 (并发级别: {concurrent_level})")
            
            return {
                "task_id": task_id,
                "concurrent_level": concurrent_level,
                "success": False,
                "execution_time": execution_time,
                "start_time": start_time,
                "end_time": end_time,
                "error": "Timeout"
            }
        
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            with self.print_lock:
                print(f"✗ 任务 {task_id} 出错: {str(e)} (并发级别: {concurrent_level})")
            
            return {
                "task_id": task_id,
                "concurrent_level": concurrent_level,
                "success": False,
                "execution_time": execution_time,
                "start_time": start_time,
                "end_time": end_time,
                "error": str(e)
            }
    
    def test_concurrent_level(self, concurrent_level: int) -> Dict[str, Any]:
        """
        测试特定并发级别的性能
        
        Args:
            concurrent_level: 并发数量
            
        Returns:
            测试结果字典
        """
        print(f"\n🔄 测试并发数: {concurrent_level}")
        print("=" * 40)
        print(f"开始运行 {self.tasks_per_level} 个任务，并发数: {concurrent_level}")
        
        # 记录测试开始时间
        test_start_time = time.time()
        
        # 生成任务ID列表
        task_ids = [f"task_{concurrent_level}_{i+1}" for i in range(self.tasks_per_level)]
        
        # 使用线程池执行并发任务
        task_results = []
        with ThreadPoolExecutor(max_workers=concurrent_level) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(self.run_single_evaluation, task_id, concurrent_level): task_id
                for task_id in task_ids
            }
            
            # 收集结果
            completed_count = 0
            for future in as_completed(future_to_task):
                task_result = future.result()
                task_results.append(task_result)
                completed_count += 1
                
                with self.print_lock:
                    print(f"进度: [{completed_count}/{self.tasks_per_level}]")
        
        # 记录测试结束时间
        test_end_time = time.time()
        total_test_time = test_end_time - test_start_time
        
        # 计算统计数据
        successful_tasks = [r for r in task_results if r["success"]]
        failed_tasks = [r for r in task_results if not r["success"]]
        
        if successful_tasks:
            execution_times = [r["execution_time"] for r in successful_tasks]
            avg_time = statistics.mean(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)
            std_dev = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
            throughput = len(successful_tasks) / total_test_time
        else:
            avg_time = min_time = max_time = std_dev = throughput = 0
        
        success_rate = len(successful_tasks) / len(task_results) * 100
        
        # 构建结果
        result = {
            "concurrent_level": concurrent_level,
            "total_tasks": len(task_results),
            "successful_tasks": len(successful_tasks),
            "failed_tasks": len(failed_tasks),
            "success_rate": success_rate,
            "avg_execution_time": avg_time,
            "min_execution_time": min_time,
            "max_execution_time": max_time,
            "std_deviation": std_dev,
            "throughput": throughput,
            "total_test_time": total_test_time,
            "task_results": task_results
        }
        
        # 打印结果
        print(f"\n📊 并发数 {concurrent_level} 测试结果:")
        print(f"  ✅ 成功任务: {len(successful_tasks)}/{len(task_results)}")
        print(f"  📈 成功率: {success_rate:.1f}%")
        if successful_tasks:
            print(f"  ⏱️  平均用时: {avg_time:.2f}秒")
            print(f"  📊 用时范围: {min_time:.2f}s - {max_time:.2f}s")
            print(f"  🎯 标准差: {std_dev:.2f}秒")
            print(f"  📈 吞吐量: {throughput:.2f} 任务/秒")
            print(f"  🕐 总耗时: {total_test_time:.1f}秒")
        
        if failed_tasks:
            print(f"  ❌ 失败任务: {len(failed_tasks)}")
            error_summary = {}
            for task in failed_tasks:
                error = task.get("error", "Unknown")
                error_summary[error] = error_summary.get(error, 0) + 1
            for error, count in error_summary.items():
                print(f"    - {error}: {count} 次")
        
        return result
    
    def run_stress_test(self) -> Dict[str, Any]:
        """
        运行完整的压力测试
        
        Returns:
            完整的测试结果
        """
        print("=" * 60)
        print("MEvalKit 评测服务压力测试")
        print("=" * 60)
        print("测试配置:")
        print(f"  - 并发数测试: {self.concurrent_levels}")
        print(f"  - 每个级别任务数: {self.tasks_per_level}")
        print(f"  - 平均任务时间: {self.avg_task_time}秒")
        print("=" * 60)
        
        test_start_time = time.time()
        
        # 运行各个并发级别的测试
        all_results = []
        for concurrent_level in self.concurrent_levels:
            result = self.test_concurrent_level(concurrent_level)
            all_results.append(result)
            self.results[concurrent_level] = result
            time.sleep(2)  # 测试间隔
        
        test_end_time = time.time()
        total_test_duration = test_end_time - test_start_time
        
        # 生成汇总报告
        self.print_summary_report(all_results, total_test_duration)
        
        # 保存结果
        self.save_results(all_results, total_test_duration)
        
        return {
            "test_config": {
                "concurrent_levels": self.concurrent_levels,
                "tasks_per_level": self.tasks_per_level,
                "avg_task_time": self.avg_task_time
            },
            "results": all_results,
            "total_duration": total_test_duration,
            "timestamp": datetime.now().isoformat()
        }
    
    def print_summary_report(self, results: List[Dict[str, Any]], total_duration: float):
        """打印汇总报告"""
        print("\n" + "=" * 80)
        print("📋 压力测试结果汇总")
        print("=" * 80)
        print(f"{'并发数':<8} {'成功率':<10} {'平均用时':<12} {'吞吐量':<12} {'标准差':<10} {'总时间':<10}")
        print("-" * 70)
        
        for result in results:
            concurrent = result["concurrent_level"]
            success_rate = result["success_rate"]
            avg_time = result["avg_execution_time"]
            throughput = result["throughput"]
            std_dev = result["std_deviation"]
            total_time = result["total_test_time"]
            
            print(f"{concurrent:<8} {success_rate:>6.1f}%   {avg_time:>8.2f}s    {throughput:>8.2f}/s   {std_dev:>6.2f}s   {total_time:>6.1f}s")
        
        print("-" * 70)
        
        # 性能分析
        print("\n📈 性能分析:")
        
        # 找到最佳吞吐量
        best_throughput = max(results, key=lambda x: x["throughput"])
        print(f"  🏆 最佳吞吐量: {best_throughput['concurrent_level']} 并发 ({best_throughput['throughput']:.2f} 任务/秒)")
        
        # 找到最稳定的配置（标准差最小）
        stable_results = [r for r in results if r["successful_tasks"] > 0]
        if stable_results:
            most_stable = min(stable_results, key=lambda x: x["std_deviation"])
            print(f"  🎯 最稳定配置: {most_stable['concurrent_level']} 并发 (标准差: {most_stable['std_deviation']:.2f}秒)")
        
        # 平均用时趋势
        if len(results) > 1:
            first_avg = results[0]["avg_execution_time"]
            last_avg = results[-1]["avg_execution_time"]
            trend = "增长" if last_avg > first_avg else "下降"
            change = abs(last_avg - first_avg)
            print(f"  📊 平均用时趋势: {trend} ({change:.2f}秒)")
        
        print(f"\n🕐 总测试时间: {total_duration:.1f}秒")
        print("=" * 80)
    
    def save_results(self, results: List[Dict[str, Any]], total_duration: float):
        """保存测试结果到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果
        detailed_file = self.results_dir / f"stress_test_detailed_{timestamp}.json"
        detailed_data = {
            "test_config": {
                "concurrent_levels": self.concurrent_levels,
                "tasks_per_level": self.tasks_per_level,
                "avg_task_time": self.avg_task_time
            },
            "results": results,
            "total_duration": total_duration,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False)
        
        # 保存汇总结果
        summary_file = self.results_dir / f"stress_test_summary_{timestamp}.json"
        summary_data = {
            "test_config": {
                "concurrent_levels": self.concurrent_levels,
                "tasks_per_level": self.tasks_per_level,
                "avg_task_time": self.avg_task_time
            },
            "summary": [
                {
                    "concurrent_level": r["concurrent_level"],
                    "success_rate": r["success_rate"],
                    "avg_execution_time": r["avg_execution_time"],
                    "throughput": r["throughput"],
                    "std_deviation": r["std_deviation"],
                    "total_test_time": r["total_test_time"]
                }
                for r in results
            ],
            "total_duration": total_duration,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 测试结果已保存:")
        print(f"  📄 详细结果: {detailed_file}")
        print(f"  📊 汇总结果: {summary_file}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="MEvalKit 评测服务压力测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python evaluation_stress_test.py                          # 默认配置测试
  python evaluation_stress_test.py --quick-test             # 快速测试
  python evaluation_stress_test.py --concurrent "1,4,8"     # 自定义并发数
  python evaluation_stress_test.py --tasks 30 --avg-time 3  # 自定义任务数和时间
        """
    )
    
    parser.add_argument(
        "--concurrent",
        type=str,
        default="1,4,8,16,32",
        help="并发数量列表，逗号分隔 (默认: 1,4,8,16,32)"
    )
    
    parser.add_argument(
        "--tasks",
        type=int,
        default=50,
        help="每个并发级别要运行的任务数 (默认: 50)"
    )
    
    parser.add_argument(
        "--avg-time",
        type=float,
        default=5.0,
        help="平均任务模拟时间，单位秒 (默认: 5.0)"
    )
    
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="快速测试模式 (较少任务数，较短时间)"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    # 解析并发数量列表
    try:
        concurrent_levels = [int(x.strip()) for x in args.concurrent.split(",")]
    except ValueError:
        print("❌ 错误: 并发数量格式无效，请使用逗号分隔的整数列表")
        return
    
    # 快速测试模式配置
    if args.quick_test:
        tasks_per_level = 10
        avg_task_time = 2.0
        concurrent_levels = [1, 4, 8]
        print("🚀 快速测试模式已启用")
    else:
        tasks_per_level = args.tasks
        avg_task_time = args.avg_time
    
    # 创建并运行压力测试
    stress_test = EvaluationStressTest(
        concurrent_levels=concurrent_levels,
        tasks_per_level=tasks_per_level,
        avg_task_time=avg_task_time
    )
    
    try:
        results = stress_test.run_stress_test()
        print("\n✅ 压力测试完成!")
        return results
    except KeyboardInterrupt:
        print("\n\n❌ 测试被用户中断")
        return None
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {str(e)}")
        return None


if __name__ == "__main__":
    main() 