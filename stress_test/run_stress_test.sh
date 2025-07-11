#!/bin/bash

# MEvalKit 压力测试运行脚本

echo "====================================================="
echo "           MEvalKit 压力测试运行脚本"
echo "====================================================="
echo ""

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ 错误: 未找到Python，请确保已安装Python"
    exit 1
fi

# 显示菜单
echo "请选择要运行的测试类型："
echo ""
echo "1) 快速验证测试 (推荐初次使用)"
echo "   - 测试并发数: 1, 4, 8"
echo "   - 每个级别 3 个任务"
echo "   - 大约需要 1-2 分钟"
echo ""
echo "2) 完整压力测试 (默认配置)"
echo "   - 测试并发数: 1, 4, 8, 16, 32"
echo "   - 每个级别 50 个任务"
echo "   - 大约需要 20-30 分钟"
echo ""
echo "3) 快速压力测试模式"
echo "   - 测试并发数: 1, 4, 8"
echo "   - 每个级别 10 个任务"
echo "   - 大约需要 3-5 分钟"
echo ""
echo "4) 自定义配置"
echo "   - 手动输入参数"
echo ""
echo "0) 退出"
echo ""

# 读取用户选择
read -p "请输入您的选择 (0-4): " choice

case $choice in
    1)
        echo ""
        echo "🚀 开始运行快速验证测试..."
        echo ""
        python quick_stress_test.py
        ;;
    2)
        echo ""
        echo "🚀 开始运行完整压力测试 (默认配置)..."
        echo "⚠️  警告: 此测试可能需要 20-30 分钟完成"
        echo ""
        read -p "确认要继续吗? (y/N): " confirm
        if [[ $confirm =~ ^[Yy]$ ]]; then
            python evaluation_stress_test.py
        else
            echo "测试已取消"
        fi
        ;;
    3)
        echo ""
        echo "🚀 开始运行快速压力测试模式..."
        echo ""
        python evaluation_stress_test.py --quick-test
        ;;
    4)
        echo ""
        echo "🔧 自定义配置模式"
        echo ""
        
        # 获取并发数量
        read -p "请输入并发数量 (例如: 1,4,8,16,32): " concurrent
        if [ -z "$concurrent" ]; then
            concurrent="1,4,8,16,32"
        fi
        
        # 获取任务数量
        read -p "请输入每个级别的任务数 (默认: 50): " tasks
        if [ -z "$tasks" ]; then
            tasks=50
        fi
        
        # 获取平均时间
        read -p "请输入平均任务时间(秒) (默认: 5.0): " avgtime
        if [ -z "$avgtime" ]; then
            avgtime=5.0
        fi
        
        echo ""
        echo "🚀 开始运行自定义压力测试..."
        echo "配置: 并发数=$concurrent, 任务数=$tasks, 平均时间=${avgtime}秒"
        echo ""
        
        python evaluation_stress_test.py --concurrent "$concurrent" --tasks "$tasks" --avg-time "$avgtime"
        ;;
    0)
        echo "退出程序"
        exit 0
        ;;
    *)
        echo "❌ 无效的选择，请输入 0-4 之间的数字"
        exit 1
        ;;
esac

echo ""
echo "====================================================="
echo "                 测试完成"

# 检查是否有结果文件
if [ -d "stress_test_results" ]; then
    result_count=$(ls -1 stress_test_results/*.json 2>/dev/null | wc -l)
    if [ $result_count -gt 0 ]; then
        echo "📊 测试结果已保存到 stress_test_results/ 目录"
        echo "最新结果文件:"
        ls -1t stress_test_results/*.json | head -2
    fi
fi

echo "=====================================================" 