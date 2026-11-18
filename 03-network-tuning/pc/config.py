# ============================================================
# PC 端配置文件
# 使用前请根据你的本机环境修改以下参数
# ============================================================

# --- 服务器配置 ---
# 监听地址：0.0.0.0 表示监听所有网络接口
SERVER_IP = "0.0.0.0"
# TCP 端口（需与 K230 端 config.py 中的 TCP_PORT 一致）
SERVER_PORT = 6001

# --- 滑条配置 ---
# 滑条数量
SLIDER_COUNT = 6
# 滑条名称前缀
SLIDER_NAME_PREFIX = "Channel"
# 滑条初始值 (0-255)
SLIDER_INITIAL_VALUE = 128

# --- 显示配置 ---
WINDOW_NAME = "H265 Stream"
# 初始窗口尺寸
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 600

# --- 二值化参数 ---
# 自适应阈值邻域大小（奇数）
BIN_BLOCK_SIZE = 11
# 常数减数
BIN_C = 2
