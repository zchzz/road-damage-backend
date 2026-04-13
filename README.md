# Road Damage Detection System

一个基于 Vue 3 + FastAPI + YOLO 的道路病害视频检测系统。

项目采用“前后端公网部署 + 本地 worker 真实检测”的结构：

- 前端部署到 Render
- 后端部署到 Render
- 本地电脑运行 worker，负责真正的视频检测与结果回传

这样可以把页面和任务管理放到公网，同时继续使用本地电脑的模型、显卡和权重文件完成检测。

---

## 一、项目架构

### 1. 前端
前端负责：

- 上传视频
- 设置检测参数（confidence / skip_frames / mode）
- 查看任务状态
- 实时查看任务进度
- 预览和下载结果文件

### 2. 后端
后端负责：

- 接收视频上传
- 写入任务队列
- 提供任务状态查询接口
- 提供结果查询接口
- 为 worker 提供 claim / download / progress / complete 接口
- 暴露上传文件和结果文件的静态访问地址

### 3. 本地 worker
本地 worker 负责：

- 轮询公网后端领取任务
- 下载原始视频到本地临时目录
- 调用本地模型和权重进行真实检测
- 生成结果视频、JSON 和 HTML 报告
- 将结果上传回公网后端

---

## 二、项目目录说明

### 后端核心目录

- `app/routes/upload.py`：上传接口
- `app/routes/task.py`：任务状态接口
- `app/routes/result.py`：结果接口
- `app/routes/worker.py`：worker 领取/上报接口
- `app/routes/ws.py`：WebSocket 任务状态推送
- `app/config.py`：配置与目录定义

### 算法相关目录

- `algorithm/weights/best.pt`：默认模型权重
- `algorithm/adapters/detector_adapter.py`：检测适配层
- `algorithm/source_project/road_damage_detector.py`：检测核心逻辑

### 数据目录

启动后会自动创建以下目录：

- `data/tasks`：任务元数据
- `data/uploads`：上传原始视频
- `data/outputs`：检测结果文件

---

## 三、接口说明

### 1. 上传任务
`POST /api/upload`

表单字段：

- `file`：视频文件
- `confidence`：置信度阈值，默认 `0.25`
- `skip_frames`：抽帧间隔，默认 `1`
- `mode`：任务模式，可选 `real` 或 `smoke`

返回：

- `task_id`
- `status`
- `message`
- `progress`
- `filename`
- `upload_url`
- `confidence`
- `skip_frames`
- `mode`

### 2. 查询任务状态
`GET /api/task/{task_id}`

返回：

- `status`
- `progress`
- `message`
- `current_frame`
- `total_frames`
- `result_ready`
- 任务参数与结果 URL

### 3. 查询结果
`GET /api/result/{task_id}`

返回：

- `summary`
- `detections`
- `output_video_url`
- `result_json_url`
- `report_url`
- 兼容字段：
  - `annotated_video_url`
  - `json_url`
  - `html_report_url`

### 4. WebSocket 推送
`WS /ws/{task_id}`

用于前端实时接收任务状态更新。

### 5. worker 相关接口

- `POST /api/worker/claim`
- `GET /api/worker/download/{task_id}`
- `POST /api/worker/progress/{task_id}`
- `POST /api/worker/complete/{task_id}`
- `POST /api/worker/fail/{task_id}`

worker 请求头需要带：

- `X-Worker-Secret: <WORKER_SHARED_SECRET>`

---

## 四、本地开发

### 1. 启动后端

进入后端项目目录：

pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
启动后默认地址：

http://127.0.0.1:8000
### 2. 启动前端

进入前端项目目录：

npm install
npm run dev

启动后默认地址：

http://127.0.0.1:5173
### 3. 启动本地 worker

确保当前目录是后端项目根目录，再运行：

python worker.py

也可以显式指定：

BACKEND_BASE_URL=http://127.0.0.1:8000
WORKER_MODE=real
WORKER_SHARED_SECRET=your-secret
python worker.py


## 五、Render 公网部署
方案说明

推荐采用以下部署方式：

前端：Render Static Site 或 Web Service

后端：Render Web Service

worker：仍运行在你自己的电脑上

这样部署后：

用户通过 Render 前端上传视频

Render 后端写入任务

你的本地 worker 从 Render 后端领取任务

本地 worker 用本地模型做真实检测

worker 把结果回传 Render 后端

前端查看结果

## 六、环境变量配置
### 1. 后端环境变量

见 .env.example

关键项：

CORS_ORIGINS

DEFAULT_TASK_MODE

DEFAULT_CONFIDENCE

DEFAULT_SKIP_FRAMES

WORKER_SHARED_SECRET

MODEL_PATH

PUBLIC_BACKEND_URL

### 2. 前端环境变量

见 .env.example

关键项：

VITE_API_BASE_URL

VITE_WS_BASE_URL

### 3. 本地 worker 环境变量

见 .env.example

关键项：

BACKEND_BASE_URL

WORKER_MODE

WORKER_SHARED_SECRET

MODEL_PATH

## 七、推荐环境变量示例
Render 后端
CORS_ORIGINS=https://your-frontend.onrender.com,http://localhost:5173,http://127.0.0.1:5173
DEFAULT_TASK_MODE=real
DEFAULT_CONFIDENCE=0.25
DEFAULT_SKIP_FRAMES=1
WORKER_SHARED_SECRET=replace-with-a-long-random-secret
MODEL_PATH=algorithm/weights/best.pt
PUBLIC_BACKEND_URL=https://your-backend.onrender.com
MAX_FILE_SIZE_MB=1024
APP_ENV=production
Render 前端
VITE_API_BASE_URL=https://your-backend.onrender.com/api
VITE_WS_BASE_URL=wss://your-backend.onrender.com
本地 worker
BACKEND_BASE_URL=https://your-backend.onrender.com
WORKER_MODE=real
WORKER_SHARED_SECRET=replace-with-a-long-random-secret
MODEL_PATH=algorithm/weights/best.pt
WORKER_POLL_INTERVAL=2
## 八、模型与检测说明
默认模型路径

默认模型权重路径为：

algorithm/weights/best.pt

也可以通过环境变量覆盖：

MODEL_PATH=algorithm/weights/best.pt
检测模式

real：真实检测，调用本地模型

smoke：烟雾测试，仅用于联调接口，不执行真实检测

生产环境建议使用：

DEFAULT_TASK_MODE=real
WORKER_MODE=real
## 九、结果文件说明

每个任务完成后，一般会生成三个文件：

标注视频

result.json

report.html

前端支持：

当前页预览视频

当前页预览 JSON

当前页 iframe 预览 HTML 报告

下载全部结果文件