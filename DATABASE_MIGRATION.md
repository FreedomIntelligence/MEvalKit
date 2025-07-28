# 数据库迁移说明

本项目已从文件系统存储方式升级为数据库存储方式，以提供更好的数据管理和查询性能。

## 主要改进

### 1. 存储方式升级
- **之前**: 评测结果保存在 `results/` 目录下的JSON文件中
- **现在**: 评测结果保存在SQLite数据库中，同时保留文件备份

### 2. 数据库结构
- **EvaluationResult表**: 存储评测结果和统计信息
- **EvaluationTask表**: 存储评测任务状态和进度

### 3. 兼容性保证
- 新的评测结果会同时保存到数据库和文件系统
- 如果数据库不可用，系统会自动回退到文件系统
- 现有的文件数据可以通过迁移脚本导入数据库

## 安装和设置

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 初始化数据库
数据库会在应用启动时自动创建。如果需要手动初始化：
```python
from src.database.models import db_manager
db_manager.create_tables()
```

### 3. 迁移现有数据（可选）
如果您有现有的评测结果文件，可以运行迁移脚本：
```bash
python migrate_to_database.py
```

## 数据库表结构

### EvaluationResult表
| 字段 | 类型 | 说明 |
|------|------|------|
| id | Integer | 主键 |
| business_id | String | 业务ID |
| user_id | String | 用户ID |
| dataset_name | String | 数据集名称 |
| model_name | String | 模型名称 |
| evaluation_mode | String | 评测模式(automatic/manual) |
| eval_type | String | 评测类型(llmjudge/textmcq/imagemcq) |
| total_questions | Integer | 总题目数 |
| valid_questions | Integer | 有效题目数 |
| valid_ratio | Float | 有效率 |
| raw_score | Float | 原始分数 |
| score | Float | 标准化分数 |
| result_data | JSON | 详细结果数据 |
| response_data | JSON | 模型响应数据 |
| created_at | DateTime | 创建时间 |
| updated_at | DateTime | 更新时间 |
| is_completed | Boolean | 是否完成 |

### EvaluationTask表
| 字段 | 类型 | 说明 |
|------|------|------|
| id | Integer | 主键 |
| task_id | String | 任务ID |
| business_id | String | 业务ID |
| user_id | String | 用户ID |
| dataset_name | String | 数据集名称 |
| model_name | String | 模型名称 |
| evaluation_mode | String | 评测模式 |
| eval_type | String | 评测类型 |
| status | String | 任务状态 |
| progress | Float | 进度百分比 |
| current_question | Integer | 当前题目 |
| total_questions | Integer | 总题目数 |
| question_limitation | Integer | 题目限制 |
| max_workers | Integer | 最大工作线程数 |
| error_message | Text | 错误信息 |
| created_at | DateTime | 创建时间 |
| started_at | DateTime | 开始时间 |
| completed_at | DateTime | 完成时间 |

## API变更

### 1. 获取用户评测记录
- **之前**: 从文件系统读取
- **现在**: 优先从数据库读取，失败时回退到文件系统

### 2. 排行榜数据
- **之前**: 从文件系统读取
- **现在**: 优先从数据库读取，失败时回退到文件系统

### 3. 任务状态检查
- **之前**: 从文件系统读取结果文件
- **现在**: 优先从数据库读取，失败时回退到文件系统

## 性能优化

### 1. 查询性能
- 数据库索引优化查询速度
- 减少文件I/O操作
- 支持复杂查询和统计

### 2. 并发处理
- 数据库事务保证数据一致性
- 支持多用户并发评测
- 任务状态实时更新

### 3. 数据管理
- 自动清理过期任务
- 数据备份和恢复
- 支持数据导出

## 故障排除

### 1. 数据库连接失败
如果数据库连接失败，系统会自动回退到文件系统模式，确保服务正常运行。

### 2. 数据不一致
如果发现数据库和文件系统数据不一致，可以：
1. 重新运行迁移脚本
2. 删除数据库文件重新初始化
3. 检查日志文件定位问题

### 3. 性能问题
如果遇到性能问题：
1. 检查数据库文件大小
2. 清理过期任务数据
3. 优化数据库索引

## 维护建议

### 1. 定期备份
建议定期备份数据库文件：
```bash
cp mevalkit.db mevalkit.db.backup.$(date +%Y%m%d)
```

### 2. 清理旧数据
定期清理过期的任务数据：
```python
from src.database.repository import task_repo
task_repo.cleanup_old_tasks(days=30)  # 清理30天前的任务
```

### 3. 监控数据库大小
监控数据库文件大小，避免过大影响性能。

## 迁移回滚

如果需要回滚到文件系统模式，可以：
1. 修改评测模块，移除数据库相关代码
2. 恢复原有的文件操作逻辑
3. 确保文件系统权限正确

## 技术支持

如果遇到问题，请：
1. 检查日志文件
2. 确认数据库文件权限
3. 验证依赖包版本
4. 联系技术支持团队 