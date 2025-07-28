# 数据库安全保护指南

## 🔐 概述

为MEvalKit项目添加了数据库密码保护和数据加密功能，确保评测数据的安全性。

## 🛡️ 安全特性

### 1. 用户认证系统
- **多用户支持**: 支持管理员和普通用户角色
- **密码哈希**: 使用SHA256哈希存储密码
- **角色管理**: 管理员和普通用户权限分离

### 2. 数据加密
- **AES加密**: 使用Fernet（AES-128-CBC）加密敏感数据
- **密钥派生**: 使用PBKDF2从密码生成加密密钥
- **盐值保护**: 每个数据库使用唯一的盐值

### 3. 安全存储
- **敏感数据加密**: 评测结果、答题详情等敏感数据加密存储
- **元数据保护**: 基本信息（分数、模型名等）保持可查询
- **密钥分离**: 加密密钥与数据分离存储

## 🚀 使用方法

### 方案一：完整安全数据库（推荐）

```bash
# 1. 设置安全数据库
python secure_database.py --setup

# 2. 迁移现有数据
python secure_database.py --migrate mevalkit.db

# 3. 添加用户
python secure_database.py --add-user user1 pass1

# 4. 查看统计
python secure_database.py --stats

# 5. 列出记录
python secure_database.py --list
```



## 📊 数据库结构

### 用户表 (users)
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### 加密评测结果表 (evaluation_results_secure)
```sql
CREATE TABLE evaluation_results_secure (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    business_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    evaluation_mode TEXT NOT NULL,
    eval_type TEXT NOT NULL,
    total_questions INTEGER,
    valid_questions INTEGER,
    valid_ratio REAL,
    raw_score REAL,
    score REAL,
    result_data_encrypted TEXT,      -- 加密的评测结果
    response_data_encrypted TEXT,    -- 加密的响应数据
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_completed BOOLEAN DEFAULT 0
);
```

## 🔑 默认账号

### 演示数据库
- **管理员**: admin / demo123
- **普通用户**: user / user123

### 生产环境
- 首次设置时创建管理员账号
- 密码通过交互式输入，不显示在屏幕上

## 🛠️ 集成到现有系统

### 1. 修改数据库连接

```python
# 原来的连接
from src.database.models import db_manager

# 安全连接
from secure_database import SecureDatabase

db = SecureDatabase()
if db.authenticate(username, password):
    # 使用安全数据库
    pass
```

### 2. 数据加密存储

```python
# 存储数据时加密
encrypted_result = db.encrypt_data(json.dumps(result_data))
encrypted_response = db.encrypt_data(json.dumps(response_data))

# 查询时解密
decrypted_result = json.loads(db.decrypt_data(encrypted_result))
```

### 3. 权限控制

```python
# 检查用户权限
if db.current_role == 'admin':
    # 管理员操作
    db.add_user('newuser', 'password')
else:
    # 普通用户操作
    pass
```

## 🔍 安全验证

### 1. 检查加密效果

```bash
# 查看原始数据库内容
sqlite3 demo_secure.db "SELECT business_id, result_data_encrypted FROM evaluation_results_secure;"
```

### 2. 验证认证

```bash
# 测试认证
python -c "
from demo_secure_db import DemoSecureDatabase
db = DemoSecureDatabase()
print(db.authenticate('admin', 'demo123'))
"
```

### 3. 数据完整性

```bash
# 验证数据完整性
python -c "
from demo_secure_db import DemoSecureDatabase
db = DemoSecureDatabase()
db.show_data('admin', 'demo123')
"
```

## ⚠️ 安全注意事项

### 1. 密码管理
- **强密码**: 使用包含大小写字母、数字、特殊字符的强密码
- **定期更换**: 定期更换管理员密码
- **安全存储**: 不要将密码存储在代码中

### 2. 密钥保护
- **配置文件**: 保护 `db_config.json` 配置文件
- **权限设置**: 设置适当的文件权限
- **备份安全**: 加密备份文件

### 3. 访问控制
- **最小权限**: 遵循最小权限原则
- **审计日志**: 记录数据库访问日志
- **网络隔离**: 限制数据库网络访问

## 🔧 故障排除

### 1. 认证失败
```bash
# 检查用户表
sqlite3 demo_secure.db "SELECT username, role FROM users;"

# 重置密码
python secure_database.py --reset-password admin
```

### 2. 解密失败
```bash
# 检查配置文件
cat db_config.json

# 验证密钥
python -c "
from secure_database import SecureDatabase
db = SecureDatabase()
print(db.load_config())
"
```

### 3. 数据迁移问题
```bash
# 检查源数据库
sqlite3 mevalkit.db ".tables"

# 手动迁移
python secure_database.py --migrate-step-by-step mevalkit.db
```

## 📈 性能考虑

### 1. 加密开销
- **CPU使用**: 加密/解密会增加CPU使用
- **存储空间**: 加密数据会增加约30%存储空间
- **查询性能**: 加密字段无法直接查询

### 2. 优化建议
- **选择性加密**: 只加密敏感数据
- **索引优化**: 在非加密字段上建立索引
- **缓存策略**: 缓存解密后的常用数据

## 🎯 最佳实践

### 1. 部署建议
- 使用HTTPS传输
- 定期备份加密数据库
- 监控异常访问

### 2. 开发建议
- 在开发环境使用演示数据库
- 在生产环境使用完整安全数据库
- 定期更新加密算法

### 3. 维护建议
- 定期检查用户权限
- 清理过期用户账号
- 更新安全补丁

## 📞 技术支持

如果遇到问题，请检查：
1. 密码是否正确
2. 配置文件是否存在
3. 数据库文件权限
4. 加密库是否正确安装

---

**注意**: 这个安全系统提供了基本的数据保护，但对于高安全要求的环境，建议考虑使用专业的数据库安全解决方案。 