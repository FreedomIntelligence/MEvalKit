#!/usr/bin/env python3
"""
安全数据库管理系统
支持密码保护和数据加密
"""

import sqlite3
import hashlib
import os
import json
import base64
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import getpass

class SecureDatabase:
    """安全数据库管理器"""
    
    def __init__(self, db_path="mevalkit_secure.db", config_file="db_config.json"):
        self.db_path = db_path
        self.config_file = config_file
        self.fernet = None
        self.is_authenticated = False
        
    def generate_key_from_password(self, password: str, salt: bytes = None) -> tuple:
        """从密码生成加密密钥"""
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key, salt
    
    def setup_database(self, admin_password: str):
        """设置安全数据库"""
        print("🔐 设置安全数据库...")
        
        # 生成加密密钥
        key, salt = self.generate_key_from_password(admin_password)
        self.fernet = Fernet(key)
        
        # 保存配置
        config = {
            "salt": base64.b64encode(salt).decode(),
            "db_path": self.db_path,
            "created_at": str(Path().absolute())
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # 创建数据库连接
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建用户表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建管理员用户
        admin_hash = hashlib.sha256(admin_password.encode()).hexdigest()
        cursor.execute("""
            INSERT OR REPLACE INTO users (username, password_hash, role)
            VALUES (?, ?, ?)
        """, ('admin', admin_hash, 'admin'))
        
        # 创建评测结果表（加密版本）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_results_secure (
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
                result_data_encrypted TEXT,
                response_data_encrypted TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_completed BOOLEAN DEFAULT 0
            )
        """)
        
        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_secure_business_id ON evaluation_results_secure (business_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_secure_user_id ON evaluation_results_secure (user_id)")
        
        conn.commit()
        conn.close()
        
        print("✅ 安全数据库设置完成")
        print(f"📁 数据库文件: {self.db_path}")
        print(f"📁 配置文件: {self.config_file}")
        print("🔑 管理员账号: admin")
        print("⚠️  请妥善保管密码！")
    
    def load_config(self):
        """加载配置"""
        if not Path(self.config_file).exists():
            return False
        
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        
        self.db_path = config.get('db_path', self.db_path)
        salt = base64.b64decode(config['salt'])
        return salt
    
    def authenticate(self, username: str, password: str) -> bool:
        """用户认证"""
        if not Path(self.db_path).exists():
            print("❌ 数据库不存在，请先设置数据库")
            return False
        
        # 加载配置
        salt = self.load_config()
        if salt is False:
            print("❌ 配置文件不存在")
            return False
        
        # 生成密钥
        key, _ = self.generate_key_from_password(password, salt)
        self.fernet = Fernet(key)
        
        # 验证用户
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        cursor.execute("""
            SELECT role FROM users 
            WHERE username = ? AND password_hash = ?
        """, (username, password_hash))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            self.is_authenticated = True
            self.current_user = username
            self.current_role = result[0]
            print(f"✅ 认证成功！欢迎 {username} ({result[0]})")
            return True
        else:
            print("❌ 用户名或密码错误")
            return False
    
    def add_user(self, username: str, password: str, role: str = 'user'):
        """添加用户"""
        if not self.is_authenticated or self.current_role != 'admin':
            print("❌ 需要管理员权限")
            return False
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        try:
            cursor.execute("""
                INSERT INTO users (username, password_hash, role)
                VALUES (?, ?, ?)
            """, (username, password_hash, role))
            conn.commit()
            print(f"✅ 用户 {username} 添加成功")
            return True
        except sqlite3.IntegrityError:
            print(f"❌ 用户 {username} 已存在")
            return False
        finally:
            conn.close()
    
    def encrypt_data(self, data: str) -> str:
        """加密数据"""
        if not self.fernet:
            raise Exception("未认证，无法加密数据")
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """解密数据"""
        if not self.fernet:
            raise Exception("未认证，无法解密数据")
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    def migrate_data(self, source_db_path: str):
        """迁移现有数据到安全数据库"""
        if not self.is_authenticated:
            print("❌ 请先认证")
            return False
        
        if not Path(source_db_path).exists():
            print(f"❌ 源数据库不存在: {source_db_path}")
            return False
        
        print(f"🔄 迁移数据从 {source_db_path} 到安全数据库...")
        
        # 连接源数据库
        source_conn = sqlite3.connect(source_db_path)
        source_cursor = source_conn.cursor()
        
        # 连接目标数据库
        target_conn = sqlite3.connect(self.db_path)
        target_cursor = target_conn.cursor()
        
        try:
            # 读取源数据
            source_cursor.execute("SELECT * FROM evaluation_results")
            rows = source_cursor.fetchall()
            
            print(f"📊 找到 {len(rows)} 条记录需要迁移")
            
            # 获取列名
            columns = [description[0] for description in source_cursor.description]
            
            for i, row in enumerate(rows):
                # 创建行数据字典
                row_data = dict(zip(columns, row))
                
                # 加密敏感数据
                result_data_encrypted = self.encrypt_data(json.dumps(row_data.get('result_data', [])))
                response_data_encrypted = self.encrypt_data(json.dumps(row_data.get('response_data', [])))
                
                # 插入到安全数据库
                target_cursor.execute("""
                    INSERT INTO evaluation_results_secure (
                        business_id, user_id, dataset_name, model_name,
                        evaluation_mode, eval_type, total_questions, valid_questions,
                        valid_ratio, raw_score, score, result_data_encrypted,
                        response_data_encrypted, created_at, updated_at, is_completed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row_data.get('business_id'), row_data.get('user_id'),
                    row_data.get('dataset_name'), row_data.get('model_name'),
                    row_data.get('evaluation_mode'), row_data.get('eval_type'),
                    row_data.get('total_questions'), row_data.get('valid_questions'),
                    row_data.get('valid_ratio'), row_data.get('raw_score'),
                    row_data.get('score'), result_data_encrypted,
                    response_data_encrypted, row_data.get('created_at'),
                    row_data.get('updated_at'), row_data.get('is_completed')
                ))
                
                if (i + 1) % 10 == 0:
                    print(f"📈 已迁移 {i + 1}/{len(rows)} 条记录")
            
            target_conn.commit()
            print(f"✅ 数据迁移完成！共迁移 {len(rows)} 条记录")
            
        except Exception as e:
            print(f"❌ 迁移失败: {str(e)}")
            target_conn.rollback()
        finally:
            source_conn.close()
            target_conn.close()
    
    def get_evaluation_results(self, limit: int = 10):
        """获取评测结果（解密后）"""
        if not self.is_authenticated:
            print("❌ 请先认证")
            return []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f"""
            SELECT business_id, user_id, dataset_name, model_name,
                   score, raw_score, created_at, is_completed
            FROM evaluation_results_secure
            ORDER BY created_at DESC
            LIMIT {limit}
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            results.append({
                'business_id': row[0],
                'user_id': row[1],
                'dataset_name': row[2],
                'model_name': row[3],
                'score': row[4],
                'raw_score': row[5],
                'created_at': row[6],
                'is_completed': row[7]
            })
        
        return results
    
    def show_stats(self):
        """显示统计信息"""
        if not self.is_authenticated:
            print("❌ 请先认证")
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        print("📈 安全数据库统计")
        print("=" * 40)
        
        # 用户统计
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        print(f"用户数量: {user_count}")
        
        # 评测记录统计
        cursor.execute("SELECT COUNT(*) FROM evaluation_results_secure")
        total_evaluations = cursor.fetchone()[0]
        print(f"总评测记录: {total_evaluations}")
        
        cursor.execute("SELECT COUNT(*) FROM evaluation_results_secure WHERE is_completed = 1")
        completed_evaluations = cursor.fetchone()[0]
        print(f"已完成评测: {completed_evaluations}")
        
        # 数据集统计
        cursor.execute("SELECT COUNT(DISTINCT dataset_name) FROM evaluation_results_secure")
        dataset_count = cursor.fetchone()[0]
        print(f"数据集种类: {dataset_count}")
        
        # 模型统计
        cursor.execute("SELECT COUNT(DISTINCT model_name) FROM evaluation_results_secure")
        model_count = cursor.fetchone()[0]
        print(f"模型种类: {model_count}")
        
        conn.close()

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="安全数据库管理系统")
    parser.add_argument("--setup", action="store_true", help="设置安全数据库")
    parser.add_argument("--migrate", help="迁移现有数据库")
    parser.add_argument("--add-user", nargs=2, metavar=('USERNAME', 'PASSWORD'), help="添加用户")
    parser.add_argument("--stats", action="store_true", help="显示统计信息")
    parser.add_argument("--list", action="store_true", help="列出评测记录")
    
    args = parser.parse_args()
    
    db = SecureDatabase()
    
    if args.setup:
        # 设置数据库
        password = getpass.getpass("请输入管理员密码: ")
        confirm_password = getpass.getpass("请确认密码: ")
        
        if password != confirm_password:
            print("❌ 密码不匹配")
            return
        
        db.setup_database(password)
        
    elif args.migrate:
        # 迁移数据
        username = input("用户名: ")
        password = getpass.getpass("密码: ")
        
        if db.authenticate(username, password):
            db.migrate_data(args.migrate)
        
    elif args.add_user:
        # 添加用户
        username = input("管理员用户名: ")
        password = getpass.getpass("管理员密码: ")
        
        if db.authenticate(username, password):
            db.add_user(args.add_user[0], args.add_user[1])
        
    elif args.stats:
        # 显示统计
        username = input("用户名: ")
        password = getpass.getpass("密码: ")
        
        if db.authenticate(username, password):
            db.show_stats()
        
    elif args.list:
        # 列出记录
        username = input("用户名: ")
        password = getpass.getpass("密码: ")
        
        if db.authenticate(username, password):
            results = db.get_evaluation_results(10)
            print("📊 评测记录:")
            print("=" * 80)
            for result in results:
                print(f"Business ID: {result['business_id']}")
                print(f"数据集: {result['dataset_name']}")
                print(f"模型: {result['model_name']}")
                print(f"分数: {result['score']}")
                print("-" * 40)
        
    else:
        print("安全数据库管理系统")
        print("=" * 30)
        print("用法:")
        print("  python secure_database.py --setup                    # 设置数据库")
        print("  python secure_database.py --migrate mevalkit.db      # 迁移数据")
        print("  python secure_database.py --add-user user1 pass1     # 添加用户")
        print("  python secure_database.py --stats                    # 显示统计")
        print("  python secure_database.py --list                     # 列出记录")

if __name__ == "__main__":
    main() 