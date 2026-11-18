# CanMV K230 视觉工程

嘉楠 K230 / CanMV K230（亚博 Yahboom K230、01Studio K230 mini）板端 MicroPython 工程，用于车载平衡滚球系统的钢珠位置检测与无线图传。

## 这个工程做什么

- 摄像头裁出摆杆 ROI，用灰度双边缘对比度定位钢珠，三帧确认 + 速度预测 + 丢点保持，把位置送给主控。
- K230 自建 Wi-Fi 热点，浏览器直接看 MJPEG 实时画面、截图 JPG、录制 MP4（录制在浏览器完成，不占 SD 卡）。
- 另备 H.265-TCP 低延迟图传（配 PC 解码端）、RTSP 推流、触摸双历程启动器、PC 滑条在线调参、9 类图形分类推理。
- 平台特征：用固件的 `media.sensor` / `nncase_runtime` / `cv_lite`，脚本放 `/sdcard/`，`main.py` 上电自启。**与 OpenMV 系（`import sensor, image`）和 RK3576（CPython + OpenCV）不通用。**

## 部署：最快跑通（AP 网页图传 + 钢珠识别）

1. **烧固件**：给 K230 烧录 CanMV v1.7 / v1.8 MicroPython 固件；把开发板配套资料里的 `libs/`、`ybUtils/`、`ybMain/` 放到 SD 卡根目录 —— 这些是厂商库，本仓库不含。
2. **只拷一个文件**：把 `08-standalone-main/main.py` 复制成 SD 卡的 `/sdcard/main.py`。这是自包含的单文件整合版，上电自动运行。
3. **填密码**：在 `main.py` 顶部把 `AP_PASSWORD` 改成自己的密码（≥ 8 位）。`AP_SSID` 默认 `K230-CAM`，`HTTP_PORT` 默认 `8081`，可不改。
4. **上电**：串口打印 `[READY] Connect Wi-Fi: K230-CAM` 和 `[READY] Open browser: http://<IP>:8081/`。
5. **看画面**：电脑或手机连热点 `K230-CAM`，浏览器打开串口给出的地址，即可预览 / 截图 / 录制 MP4。
6. **调参**：识别框位置、阈值在 `main.py` 顶部常量区；画面不稳时先降 `TARGET_FPS`、`JPEG_QUALITY`。

### 换成其它工作方式

只替换 `/sdcard/` 里的文件即可：

| 想做什么 | 板端文件 | 怎么启动 |
| --- | --- | --- |
| 只要 AP 图传，不跑识别 | `01-wireless-stream/k230/ap_web_stream.py` | 另存为 `/sdcard/main.py` 后上电，或在 CanMV IDE 直接运行 |
| H.265 → PC 低延迟 | `01-wireless-stream/k230/main.py` + `stream_sender.py` | 填 `WIFI_SSID` / `WIFI_PASSWORD` / `PC_IP`；**先**在 PC 跑接收端（见下），**再**运行板端 `main.py` |
| RTSP → VLC | `01-wireless-stream/k230/rtsp_wireless_stream.py` | 运行后用 VLC 打开串口打印的 URL |
| 触摸双历程菜单 | `02-course-launcher/` 的 `main.py`、`ap_web_stream.py`、`grayscale_ball_tracker.py`、`launcher_control.py` 一起拷 | 运行 `main.py`，触屏按 1 / 2 切换历程，按板载 Key 回菜单 |
| PC 滑条实时调参 | 板端 `03-network-tuning/g1.py` + `config.py` + `mculib/` | 先改 `config.py` 的 `PC_IP`；PC 装 `opencv-python numpy av` 后跑 `pc/toolx2.py` |
| 图形分类推理 | `06-canmv-ide-workspace/cls_video.py` | 把 `05-classification/deploy_config.json` 和 `model/*.kmodel` 放到 `/sdcard/AI/`，在 IDE 运行；9 类标签见 `labels.txt` |

H.265 接收端（PC，端口要与板端 `PC_PORT = 6001` 一致）：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r 01-wireless-stream/pc/requirements.txt
python 01-wireless-stream/pc/receiver.py --port 6001
```

## 目录

| 目录 | 内容 |
| --- | --- |
| `01-wireless-stream/` | 三种图传（AP / H.265 / RTSP）、标定、PC 接收端 |
| `02-course-launcher/` | 触摸双历程启动器 |
| `03-network-tuning/` | 板端采集 + PC 滑条调参 |
| `04-ball-detection/` | 二值化轮廓 / 灰度双边缘两版钢珠识别 |
| `05-classification/` | 训练配置、9 类标签、数据集划分、自训练 kmodel |
| `06-canmv-ide-workspace/` | LVGL 菜单、`nncase_runtime` 推理、矩形识别 |
| `07-experiments/`、`08-standalone-main/`、`99-archive/` | 实验、**推荐单文件入口**、历史备份 |

## 第三方内容（均未上传）

| 类别 | 获取方式 |
| --- | --- |
| CanMV 固件模块 `media.*` / `nncase_runtime` / `ulab` | 烧录官方 CanMV 固件 |
| 亚博 / 01Studio 板级库 `ybUtils.*`、`libs.*` 与官方例程 | 开发板配套资料 |
| PC 端 `.venv`（PyAV / numpy / opencv，248 MB） | 用 `requirements.txt` 重建 |
| 训练平台导出的 C++ 部署包（Eigen / nlohmann-json / Clipper / K230 SDK） | 自行重新导出 |
| YOLO11n 预训练 kmodel、900 张训练图、演示视频 | 自行训练 / 保留在本地 |

`05-classification/model/*.kmodel` 为自行训练所得，随仓库保留。
真实 Wi-Fi 名称 / 密码、AP 密码与共享密钥已脱敏为 `YOUR_WIFI_SSID`、`YOUR_WIFI_PASSWORD`、`YOUR_AP_PASSWORD`、`YOUR_SHARED_SECRET`，**运行前必须填回**。

## 项目系列

同一竞赛项目按**硬件环境**拆分的独立仓库，各平台固件 API 与运行栈互不相通，代码不能混用。

| 硬件环境 / 运行栈 | 仓库 |
| --- | --- |
| 嘉楠 K230 · CanMV K230（MicroPython 固件） | **canmv-k230-vision** ← 本仓库 |
| 逐飞 OMV RT5 · OpenMV 兼容固件 | [omv-rt5-openmv-vision](https://github.com/kevin-kaisilanna/omv-rt5-openmv-vision) |
| 逐飞 OpenART Mini · 板级显示 / SPI | [openart-mini-display](https://github.com/kevin-kaisilanna/openart-mini-display) |
| KICKPI K7C · RK3576（Ubuntu + CPython/OpenCV） | [rk3576-k7c-vision](https://github.com/kevin-kaisilanna/rk3576-k7c-vision) |
| 竞赛设计报告（跨平台文档，无代码） | [rolling-ball-design-report](https://github.com/kevin-kaisilanna/rolling-ball-design-report) |

> 所有仓库均只收录自研代码；第三方开源库、厂商 SDK、平台生成代码与预训练模型未上传，来源见各仓库 README。

## 许可证

仅收录自研代码，尚未指定开源许可证（默认保留所有权利）；第三方组件权利归各自权利人。
