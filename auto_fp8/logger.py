import logging
import os
from logging.handlers import RotatingFileHandler

# 创建一个logger对象
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# # 创建一个文件处理器,用于将日志写入到文件
# log_dir = os.path.join(os.getcwd(), 'logs')
# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)
# file_handler = RotatingFileHandler(os.path.join(log_dir, 'app.log'), maxBytes=10 * 1024 * 1024, backupCount=5)
# file_handler.setLevel(logging.INFO)

# 创建一个控制台处理器,用于将日志输出到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 定义日志格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 为logger对象添加文件处理器和控制台处理器
# logger.addHandler(file_handler)
logger.addHandler(console_handler)