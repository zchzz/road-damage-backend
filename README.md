# Road Damage Detection Backend

## 安装依赖
pip install -r requirements.txt

## 本地启动
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

## 目录说明
- uploads/ 上传原始视频
- outputs/ 输出结果文件
- data/tasks/ 任务元数据
- logs/ 日志目录

## 接口
- POST /api/upload
- GET /api/task/{task_id}
- GET /api/result/{task_id}
- WS /ws/{task_id}