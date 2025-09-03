#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seeed Wiki 爬虫守护进程
持续监控Wiki更新，支持后台运行
"""

import os
import sys
import time
import signal
import logging
from datetime import datetime
import subprocess
import json

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wiki_monitor.log'),
        logging.StreamHandler()
    ]
)

class WikiMonitorDaemon:
    def __init__(self):
        self.running = True
        self.pid_file = 'wiki_monitor.pid'
        self.log_file = 'wiki_monitor.log'
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """处理退出信号"""
        logging.info(f"收到信号 {signum}，正在停止监控...")
        self.running = False
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """清理资源"""
        try:
            if os.path.exists(self.pid_file):
                os.remove(self.pid_file)
            logging.info("清理完成")
        except Exception as e:
            logging.error(f"清理失败: {e}")
    
    def write_pid(self):
        """写入PID文件"""
        try:
            with open(self.pid_file, 'w') as f:
                f.write(str(os.getpid()))
            logging.info(f"PID文件已写入: {self.pid_file}")
        except Exception as e:
            logging.error(f"写入PID文件失败: {e}")
    
    def check_environment(self):
        """检查运行环境"""
        logging.info("🔍 检查运行环境...")
        
        # 检查Python脚本
        if not os.path.exists('scrape_with_embeddings.py'):
            logging.error("❌ 未找到 scrape_with_embeddings.py")
            return False
        
        # 检查Ollama服务
        try:
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                logging.error("❌ Ollama服务未运行")
                return False
            
            # 检查nomic-embed-text模型
            if 'nomic-embed-text' not in result.stdout:
                logging.warning("⚠️ nomic-embed-text模型未安装，正在安装...")
                subprocess.run(['ollama', 'pull', 'nomic-embed-text'], 
                             timeout=300)  # 5分钟超时
                logging.info("✅ 模型安装完成")
            else:
                logging.info("✅ Ollama环境和模型检查通过")
            
        except subprocess.TimeoutExpired:
            logging.error("❌ Ollama服务响应超时")
            return False
        except Exception as e:
            logging.error(f"❌ 检查Ollama服务失败: {e}")
            return False
        
        return True
    
    def run_quick_check(self):
        """运行快速检查"""
        try:
            logging.info("🔍 执行快速检查...")
            result = subprocess.run([
                'python3', 'scrape_with_embeddings.py', 
                '--mode', 'incremental'
            ], capture_output=True, text=True, timeout=300)  # 5分钟超时
            
            if result.returncode == 0:
                logging.info("✅ 快速检查完成")
                return True
            else:
                logging.error(f"❌ 快速检查失败: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logging.error("❌ 快速检查超时")
            return False
        except Exception as e:
            logging.error(f"❌ 快速检查异常: {e}")
            return False
    
    def run_full_update(self):
        """运行完整更新"""
        try:
            logging.info("🔄 执行完整更新...")
            result = subprocess.run([
                'python3', 'scrape_with_embeddings.py', 
                '--mode', 'incremental'
            ], capture_output=True, text=True, timeout=1800)  # 30分钟超时
            
            if result.returncode == 0:
                logging.info("✅ 完整更新完成")
                return True
            else:
                logging.error(f"❌ 完整更新失败: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logging.error("❌ 完整更新超时")
            return False
        except Exception as e:
            logging.error(f"❌ 完整更新异常: {e}")
            return False
    
    def should_run_full_update(self):
        """检查是否需要运行完整更新"""
        last_update_file = './data_base/last_update.json'
        
        if not os.path.exists(last_update_file):
            logging.info("📂 未找到更新记录，需要完整更新")
            return True
        
        try:
            with open(last_update_file, 'r') as f:
                data = json.load(f)
                last_update_str = data.get('last_update')
                
                if not last_update_str:
                    return True
                
                last_update = datetime.fromisoformat(last_update_str)
                time_diff = datetime.now() - last_update
                
                # 如果超过24小时，需要完整更新
                if time_diff.total_seconds() >= 24 * 3600:
                    logging.info(f"⏰ 距离上次更新已超过24小时，需要完整更新")
                    return True
                else:
                    logging.info(f"⏰ 距离上次更新 {time_diff.total_seconds()/3600:.1f} 小时，跳过完整更新")
                    return False
                    
        except Exception as e:
            logging.error(f"❌ 检查更新时间失败: {e}")
            return True
    
    def run(self):
        """运行守护进程"""
        logging.info("🚀 启动 Seeed Wiki 监控守护进程")
        
        # 检查环境
        if not self.check_environment():
            logging.error("❌ 环境检查失败，退出")
            return
        
        # 写入PID文件
        self.write_pid()
        
        # 设置定时任务
        last_full_update = datetime.now()
        check_count = 0
        
        logging.info("📊 监控任务设置:")
        logging.info("   - 每30分钟: 快速检查新页面")
        logging.info("   - 每天凌晨00:00: 完整数据库更新")
        logging.info("   - 按 Ctrl+C 停止监控")
        
        try:
            while self.running:
                current_time = datetime.now()
                
                # 检查是否需要完整更新（每天凌晨00:00）
                if (current_time.hour == 0 and current_time.minute == 0 and 
                    (current_time - last_full_update).total_seconds() > 3600):
                    logging.info("⏰ 执行每日完整更新...")
                    if self.run_full_update():
                        last_full_update = current_time
                        logging.info("✅ 每日更新完成")
                    else:
                        logging.error("❌ 每日更新失败")
                
                # 每30分钟执行快速检查
                if check_count % 30 == 0:
                    logging.info(f"🔍 执行第 {check_count//30 + 1} 次快速检查...")
                    self.run_quick_check()
                
                check_count += 1
                time.sleep(60)  # 每分钟检查一次
                
        except KeyboardInterrupt:
            logging.info("⚠️ 收到中断信号")
        except Exception as e:
            logging.error(f"❌ 监控进程异常: {e}")
        finally:
            self.cleanup()
            logging.info("🛑 监控进程已停止")

def main():
    """主函数"""
    daemon = WikiMonitorDaemon()
    daemon.run()

if __name__ == "__main__":
    main()
