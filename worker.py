#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import requests

REAL_DETECT_AVAILABLE = False
REAL_DETECT_IMPORT_ERROR = None

try:
    from algorithm.adapters.detector_adapter import detect_video
    REAL_DETECT_AVAILABLE = True
except Exception as e:
    REAL_DETECT_AVAILABLE = False
    REAL_DETECT_IMPORT_ERROR = repr(e)


CLAIM_TIMEOUT = (5, 120)
PROGRESS_TIMEOUT = (5, 120)
FAIL_TIMEOUT = (5, 120)
DOWNLOAD_TIMEOUT = (10, 1800)
SUCCESS_TIMEOUT = (10, 1800)

CHUNK_SIZE = 1024 * 1024


def log(*args):
    print("[worker]", *args, flush=True)


def build_worker_id():
    return f"local-worker-{socket.gethostname()}-{os.getpid()}"


def create_session(shared_secret: str | None = None) -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "Connection": "keep-alive",
        "User-Agent": "road-damage-local-worker/2.1",
    })
    if shared_secret:
        session.headers.update({
            "X-Worker-Secret": shared_secret
        })
    return session


def claim_task(session: requests.Session, base_url: str, worker_id: str):
    resp = session.post(
        f"{base_url}/api/worker/claim",
        data={"worker_id": worker_id},
        timeout=CLAIM_TIMEOUT,
    )
    resp.raise_for_status()
    payload = resp.json()
    log("claim响应:", payload)

    if isinstance(payload, dict) and "task" in payload:
        return payload["task"]

    if isinstance(payload, dict) and payload.get("task_id"):
        return payload

    return None


def download_task_file(session: requests.Session, base_url: str, task_id: str, save_path: Path):
    with session.get(
        f"{base_url}/api/worker/download/{task_id}",
        stream=True,
        timeout=DOWNLOAD_TIMEOUT,
    ) as resp:
        resp.raise_for_status()
        with save_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)


def report_progress(
    session: requests.Session,
    base_url: str,
    task_id: str,
    progress: int,
    message: str,
):
    resp = session.post(
        f"{base_url}/api/worker/progress/{task_id}",
        data={
            "progress": int(max(0, min(progress, 100))),
            "message": message,
        },
        timeout=PROGRESS_TIMEOUT,
    )
    resp.raise_for_status()


def report_failure(session: requests.Session, base_url: str, task_id: str, message: str):
    resp = session.post(
        f"{base_url}/api/worker/fail/{task_id}",
        data={"message": message},
        timeout=FAIL_TIMEOUT,
    )
    resp.raise_for_status()


def report_success(
    session: requests.Session,
    base_url: str,
    task_id: str,
    result_video_path: Path,
    result_json_path: Path | None = None,
    report_html_path: Path | None = None,
):
    files = {
        "result_video": (result_video_path.name, result_video_path.open("rb"), "video/mp4"),
    }
    opened = [files["result_video"][1]]

    try:
        if result_json_path and result_json_path.exists():
            files["result_json"] = (
                result_json_path.name,
                result_json_path.open("rb"),
                "application/json",
            )
            opened.append(files["result_json"][1])

        if report_html_path and report_html_path.exists():
            files["report_html"] = (
                report_html_path.name,
                report_html_path.open("rb"),
                "text/html; charset=utf-8",
            )
            opened.append(files["report_html"][1])

        resp = session.post(
            f"{base_url}/api/worker/complete/{task_id}",
            files=files,
            timeout=SUCCESS_TIMEOUT,
        )
        resp.raise_for_status()
    finally:
        for fp in opened:
            try:
                fp.close()
            except Exception:
                pass


def ensure_ffmpeg_available():
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception as e:
        raise RuntimeError(
            "未检测到 ffmpeg，无法将结果视频转码为浏览器兼容格式。"
            " 请先安装 ffmpeg 并确保命令行可直接调用 ffmpeg。"
        ) from e


def transcode_to_h264(input_path: Path, output_path: Path):
    """
    将 OpenCV 生成的视频转为更适合浏览器播放的 H.264 MP4
    """
    ensure_ffmpeg_available()

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(output_path),
    ]

    log("开始转码 H.264:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg 转码失败。\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"转码输出文件无效: {output_path}")

    log(f"转码完成: {output_path}")


def run_real_mode(
    input_video: Path,
    output_video: Path,
    report_html: Path,
    result_json: Path,
    task: dict,
):
    if not REAL_DETECT_AVAILABLE:
        raise RuntimeError(
            f"真实检测模式不可用：无法导入 detect_video。错误: {REAL_DETECT_IMPORT_ERROR}"
        )

    detect_video(
        video_path=str(input_video),
        output_video_path=str(output_video),
        confidence=float(task.get("confidence", 0.25)),
        skip_frames=int(task.get("skip_frames", 1)),
        report_path=str(report_html),
        result_json_path=str(result_json),
        progress_callback=None,
    )


def run_smoke_mode(
    input_video: Path,
    output_video: Path,
    report_html: Path,
    result_json: Path,
    task: dict,
):
    shutil.copy2(input_video, output_video)

    result_payload = {
        "summary": {
            "total_detections": 0,
            "video_duration": 0,
            "processed_frames": 0,
            "damage_density": 0,
            "damage_types": {},
            "smoke_test": True,
        },
        "raw_results": {
            "mode": "smoke",
            "note": "This is a smoke test result generated by worker.py",
        },
        "task": {
            "task_id": task["task_id"],
            "filename": task["filename"],
            "confidence": task.get("confidence"),
            "skip_frames": task.get("skip_frames"),
        },
    }

    with result_json.open("w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)

    report_html.write_text(
        "<html><body><h1>Smoke Test OK</h1></body></html>",
        encoding="utf-8",
    )


def resolve_task_mode(task: dict, default_mode: str) -> str:
    task_mode = str(task.get("mode") or "").strip().lower()
    if task_mode in {"smoke", "real"}:
        return task_mode
    return default_mode


def process_one_task(
    session: requests.Session,
    base_url: str,
    task: dict,
    default_mode: str,
):
    task_id = task["task_id"]
    filename = task["filename"]
    mode = resolve_task_mode(task, default_mode)

    with tempfile.TemporaryDirectory(prefix=f"rd-worker-{task_id}-") as tmpdir:
        tmpdir = Path(tmpdir)
        input_video = tmpdir / filename

        # 原始检测输出
        raw_output_video = tmpdir / "output_raw.mp4"
        # 最终上传给后端、供浏览器预览的输出
        output_video = tmpdir / "output.mp4"

        result_json = tmpdir / "result.json"
        report_html = tmpdir / "report.html"

        report_progress(session, base_url, task_id, 5, "已接单，开始下载原始视频")
        log(f"任务 {task_id}: 开始下载")
        download_task_file(session, base_url, task_id, input_video)

        report_progress(session, base_url, task_id, 20, f"原始视频下载完成，开始处理（mode={mode}）")
        log(f"任务 {task_id}: 开始处理 mode={mode}")

        if mode == "smoke":
            run_smoke_mode(input_video, output_video, report_html, result_json, task)
        else:
            run_real_mode(input_video, raw_output_video, report_html, result_json, task)

            if not raw_output_video.exists():
                raise RuntimeError(f"未生成原始输出视频: {raw_output_video}")

            report_progress(session, base_url, task_id, 85, "检测完成，正在转码结果视频")
            transcode_to_h264(raw_output_video, output_video)

        if not output_video.exists():
            raise RuntimeError(f"未生成最终输出视频: {output_video}")

        report_progress(session, base_url, task_id, 90, "处理完成，开始上传结果")
        log(f"任务 {task_id}: 开始上传结果")

        report_success(
            session=session,
            base_url=base_url,
            task_id=task_id,
            result_video_path=output_video,
            result_json_path=result_json,
            report_html_path=report_html,
        )

        log(f"任务完成: {task_id}")


def main():
    parser = argparse.ArgumentParser(description="Local worker for road-damage-backend")
    parser.add_argument(
        "--base-url",
        default=os.getenv("BACKEND_BASE_URL") or os.getenv("RENDER_BACKEND_URL") or "http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "real"],
        default=os.getenv("WORKER_MODE", "real"),
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=float(os.getenv("WORKER_POLL_INTERVAL", "2")),
    )
    parser.add_argument(
        "--shared-secret",
        default=os.getenv("WORKER_SHARED_SECRET", ""),
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    worker_id = build_worker_id()
    session = create_session(args.shared_secret)

    log(f"启动成功，worker_id={worker_id}, default_mode={args.mode}, base_url={base_url}")
    log(f"真实检测可用: {REAL_DETECT_AVAILABLE}")
    if not REAL_DETECT_AVAILABLE:
        log(f"真实检测导入失败: {REAL_DETECT_IMPORT_ERROR}")

    while True:
        current_task = None
        try:
            task = claim_task(session, base_url, worker_id)

            if not task:
                log("当前无待处理任务，继续轮询...")
                time.sleep(args.interval)
                continue

            current_task = task
            log(f"接到任务: {task['task_id']} ({task['filename']})")
            process_one_task(session, base_url, task, args.mode)

        except KeyboardInterrupt:
            log("收到退出信号，worker 已停止")
            break
        except Exception as e:
            log("worker 异常:", repr(e))
            if current_task and current_task.get("task_id"):
                try:
                    report_failure(session, base_url, current_task["task_id"], repr(e))
                    log(f"任务失败已上报: {current_task['task_id']}")
                except Exception as report_err:
                    log("上报失败状态也失败:", repr(report_err))
            time.sleep(max(args.interval, 3))

    try:
        session.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()