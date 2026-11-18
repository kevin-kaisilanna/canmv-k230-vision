import time, os, sys
import math
from media.sensor import * #导入sensor模块，使用摄像头相关接口
from media.display import * #导入display模块，使用display相关接口
from media.media import * #导入media模块，使用meida相关接口
from machine import UART, FPIOA

#2.4寸mipi屏分辨率定义
lcd_width = 640
lcd_height = 480

THRESHOLD = (0, 100)  # 黑白图像的灰度阈值
BINARY_VISIBLE = True # 使用二值化图像你可以看到什么是线性回归。
                        # 这可能降低 FPS（每秒帧数）.

sensor = Sensor(width=1280, height=960) #构建摄像头对象，将摄像头长宽设置为4:3
sensor.reset() #复位和初始化摄像头
sensor.set_framesize(width=640, height=480) #设置帧大小，默认通道0
sensor.set_pixformat(Sensor.GRAYSCALE) #设置输出图像格式，默认通道0

Display.init(Display.ST7701,  width=lcd_width, height=lcd_height, to_ide=True) #同时使用mipi屏和IDE缓冲区显示图像
#Display.init(Display.VIRT, sensor.width(), sensor.height()) #只使用IDE缓冲区显示图像

MediaManager.init() #初始化media资源管理器

sensor.run() #启动sensor

# ========== UART 初始化 ==========
UART_TX_PIN = 9
UART_RX_PIN = 10
UART_BAUD = 115200

fpioa = FPIOA()
fpioa.set_function(UART_TX_PIN, FPIOA.UART1_TXD)
fpioa.set_function(UART_RX_PIN, FPIOA.UART1_RXD)
uart = UART(UART.UART1, UART_BAUD)


# ========== 协议工具函数 ==========
def checksum(body):
    """计算 $ 与 * 之间字符串的校验和（两位大写十六进制）"""
    s = 0
    for ch in body:
        s = (s + ord(ch)) & 0xFF
    return "%02X" % s

def pack_line_frame(n, theta_x10, length, x1, y1, x2, y2):
    """
    组装巡线帧： $L,n,theta,length,x1,y1,x2,y2*CS\r\n
    n=1 有效，n=0 无效
    """
    body = "L,%d,%d,%d,%d,%d,%d,%d" % (n, theta_x10, length, x1, y1, x2, y2)
    return "$%s*%s\r\n" % (body, checksum(body))

clock = time.clock()

while True:
    clock.tick()

    #image.binary([THRESHOLD])将灰度值在THRESHOLD范围变成了白色
    img = sensor.snapshot().binary([THRESHOLD]) if BINARY_VISIBLE else sensor.snapshot()

    # 返回一个类似 find_lines() 和find_line_segments()的对象.
    # 有以下函数使用方法： x1(), y1(), x2(), y2(), length(),
    # theta() (rotation in degrees), rho(), and magnitude().
    #
    # magnitude() 代表线性回归的指令，其值为(0, INF]。
    # 0表示一个圆，INF数值越大，表示线性拟合的效果越好。

    line = img.get_regression([(255,255) if BINARY_VISIBLE else THRESHOLD])

    if (line):

        img.draw_line(line.line(), color = 127,thickness=4)

        x1, y1 = line.x1(), line.y1()
        x2, y2 = line.x2(), line.y2()
        dx = x2 - x1
        dy = y2 - y1
        length = int(math.sqrt(dx*dx + dy*dy))
        theta_x10 = int(line.theta() * 10)

        frame = pack_line_frame(1, theta_x10, length, x1, y1, x2, y2)
        uart.write(frame)
        print(line) #打印结果

    #显示图片，仅用于LCD居中方式显示
    Display.show_image(img, x=round((lcd_width-sensor.width())/2),y=round((lcd_height-sensor.height())/2))


    print("FPS %f, mag = %s" % (clock.fps(), str(line.magnitude()) if (line) else "N/A"))
