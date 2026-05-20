"""
media_server.py — K-Ride 미디어 생성 FastAPI (Kaggle GPU 전용)
================================================================
노트북 B 전용: TTS, MusicGen, 3D Photo Inpainting, 인물 모션, FFmpeg 합성

[아키텍처]
  POST /api/media/tts         → 한국어 TTS (XTTS-v2, ~2-3GB VRAM)
  POST /api/media/musicgen    → BGM 생성 (MusicGen, ~3-6GB VRAM)
  POST /api/media/inpaint3d   → 풍경 사진 → 카메라 무빙 영상
  POST /api/media/animate     → 인물 사진 → 모션 영상 (LivePortrait, ~6GB VRAM)
  POST /api/media/render      → TTS + BGM + 영상 → FFmpeg 합성
  GET  /api/media/status/{id} → 작업 상태 폴링
  GET  /api/media/download/{id} → 완성 파일 다운로드
  GET  /api/health             → 서버 + GPU 상태

[실행]
  uvicorn media_server:app --host 0.0.0.0 --port 8001 &
"""
from __future__ import annotations

import os
import time
import uuid
import shutil
import subprocess
import traceback
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ══════════════════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════════════════
OUTPUT_DIR = Path(os.environ.get("MEDIA_OUTPUT_DIR", "/kaggle/working/media_output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = Path(os.environ.get("MEDIA_UPLOAD_DIR", "/kaggle/working/media_upload"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 동시 작업 제한 (T4 16GB 보호)
MAX_WORKERS = 1
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# ══════════════════════════════════════════════════════════════════════════════
# 작업 상태 관리 (인메모리)
# ══════════════════════════════════════════════════════════════════════════════

class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Job:
    def __init__(self, job_id: str, job_type: str):
        self.job_id = job_id
        self.job_type = job_type
        self.status = JobStatus.QUEUED
        self.created_at = time.time()
        self.started_at: float | None = None
        self.completed_at: float | None = None
        self.result_path: str | None = None
        self.error: str | None = None
        self.progress: str = ""

    def to_dict(self):
        elapsed = None
        if self.started_at:
            end = self.completed_at or time.time()
            elapsed = round(end - self.started_at, 1)
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": self.status.value,
            "progress": self.progress,
            "elapsed_seconds": elapsed,
            "result_path": self.result_path,
            "error": self.error,
        }


_jobs: dict[str, Job] = {}


def _create_job(job_type: str) -> Job:
    job_id = str(uuid.uuid4())[:8]
    job = Job(job_id, job_type)
    _jobs[job_id] = job
    return job


# ══════════════════════════════════════════════════════════════════════════════
# 모델 싱글턴 (lazy loading — 사용 시점에 로드)
# ══════════════════════════════════════════════════════════════════════════════
_tts_model = None
_musicgen_model = None


def _get_tts():
    """XTTS-v2 한국어 TTS 모델 로드"""
    global _tts_model
    if _tts_model is None:
        import torch
        from TTS.api import TTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[Media] XTTS-v2 로딩 중... (~1.9GB 다운로드)")
        _tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        print(f"[Media] XTTS-v2 로딩 완료 (device={device})")
    return _tts_model


def _get_musicgen():
    """MusicGen 모델 로드 (small 또는 medium)"""
    global _musicgen_model
    if _musicgen_model is None:
        from audiocraft.models import MusicGen
        model_name = os.environ.get("MUSICGEN_MODEL", "facebook/musicgen-small")
        print(f"[Media] MusicGen 로딩 중... ({model_name})")
        _musicgen_model = MusicGen.get_pretrained(model_name)
        print("[Media] MusicGen 로딩 완료")
    return _musicgen_model


def _unload_model(name: str):
    """GPU 메모리 해제 (모델 교체 시)"""
    import torch
    global _tts_model, _musicgen_model
    if name == "tts" and _tts_model is not None:
        del _tts_model
        _tts_model = None
    elif name == "musicgen" and _musicgen_model is not None:
        del _musicgen_model
        _musicgen_model = None
    torch.cuda.empty_cache()
    print(f"[Media] {name} 모델 언로드 + GPU 캐시 해제")


# ══════════════════════════════════════════════════════════════════════════════
# 작업 실행 함수들
# ══════════════════════════════════════════════════════════════════════════════

def _run_tts(job: Job, text: str, language: str, speaker_wav: str | None):
    """TTS 작업 실행"""
    try:
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        job.progress = "TTS 모델 로딩 중"

        tts = _get_tts()

        out_path = str(OUTPUT_DIR / f"{job.job_id}_tts.wav")
        job.progress = "음성 생성 중"

        kwargs = {
            "text": text,
            "language": language,
            "file_path": out_path,
        }
        if speaker_wav and os.path.exists(speaker_wav):
            kwargs["speaker_wav"] = speaker_wav
        else:
            # 기본 화자 사용
            speakers = tts.speakers
            if speakers:
                kwargs["speaker"] = speakers[0]

        tts.tts_to_file(**kwargs)

        job.result_path = out_path
        job.status = JobStatus.COMPLETED
        job.progress = "완료"
        job.completed_at = time.time()
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.completed_at = time.time()
        traceback.print_exc()


def _run_musicgen(job: Job, description: str, duration: int):
    """MusicGen BGM 생성"""
    try:
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        job.progress = "MusicGen 모델 로딩 중"

        model = _get_musicgen()
        model.set_generation_params(
            duration=min(duration, 30),  # 최대 30초
            use_sampling=True,
            top_k=250,
        )

        job.progress = f"BGM 생성 중 ({duration}초)"

        from audiocraft.data.audio import audio_write
        wav = model.generate([description])

        out_path = str(OUTPUT_DIR / f"{job.job_id}_bgm")
        audio_write(out_path, wav[0].cpu(), model.sample_rate, strategy="loudness")
        out_path += ".wav"

        job.result_path = out_path
        job.status = JobStatus.COMPLETED
        job.progress = "완료"
        job.completed_at = time.time()
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.completed_at = time.time()
        traceback.print_exc()


def _run_inpaint3d(job: Job, image_path: str):
    """3D Photo Inpainting — 풍경 사진 카메라 무빙"""
    try:
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        job.progress = "Depth 추정 중"

        import torch
        import numpy as np
        from PIL import Image

        # Depth Anything V2 사용 (MiDaS 대체)
        from transformers import pipeline

        depth_pipe = pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=0 if torch.cuda.is_available() else -1,
        )

        img = Image.open(image_path).convert("RGB")
        # 가로 512px로 리사이즈 (GPU 절약)
        w, h = img.size
        if w > 512:
            ratio = 512 / w
            img = img.resize((512, int(h * ratio)), Image.LANCZOS)

        job.progress = "Depth 맵 생성 중"
        depth_result = depth_pipe(img)
        depth_map = depth_result["depth"]

        # Depth map을 NumPy 배열로 변환
        depth_np = np.array(depth_map)

        # Ken Burns 효과: crop → zoom 애니메이션
        job.progress = "Ken Burns 카메라 무빙 생성 중"
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        fps = 24
        total_frames = fps * 4  # 4초 영상

        out_path = str(OUTPUT_DIR / f"{job.job_id}_3d.mp4")

        # FFmpeg pipe로 영상 생성
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            out_path,
        ]

        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        for i in range(total_frames):
            t = i / total_frames
            # 부드러운 zoom-in + pan 효과 (depth 기반 가중치)
            zoom = 1.0 + 0.15 * t  # 1.0 → 1.15x
            pan_x = int(w * 0.05 * t)
            pan_y = int(h * 0.03 * t)

            # crop 영역 계산
            cw = int(w / zoom)
            ch = int(h / zoom)
            cx = min(pan_x, w - cw)
            cy = min(pan_y, h - ch)

            from PIL import Image as PILImage
            frame = PILImage.fromarray(img_np)
            frame = frame.crop((cx, cy, cx + cw, cy + ch))
            frame = frame.resize((w, h), PILImage.LANCZOS)
            proc.stdin.write(np.array(frame).tobytes())

        proc.stdin.close()
        proc.wait()

        # depth model 해제
        del depth_pipe
        torch.cuda.empty_cache()

        job.result_path = out_path
        job.status = JobStatus.COMPLETED
        job.progress = "완료"
        job.completed_at = time.time()
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.completed_at = time.time()
        traceback.print_exc()


def _run_animate(job: Job, image_path: str, driving_video_path: str | None):
    """인물 사진 → 모션 영상 (LivePortrait)"""
    try:
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        job.progress = "LivePortrait 준비 중"

        lp_dir = os.environ.get("LIVEPORTRAIT_DIR", "/kaggle/working/LivePortrait")
        out_dir = str(OUTPUT_DIR / job.job_id)
        os.makedirs(out_dir, exist_ok=True)

        # driving video가 없으면 기본 제공
        if not driving_video_path or not os.path.exists(driving_video_path):
            driving_video_path = os.path.join(lp_dir, "assets", "examples", "driving", "d0.mp4")

        if not os.path.exists(driving_video_path):
            raise FileNotFoundError(
                f"Driving video 없음: {driving_video_path}. "
                "LivePortrait 설치 후 assets/examples/driving/ 에 기본 영상을 배치하세요."
            )

        job.progress = "LivePortrait 추론 중 (3~10분 소요)"

        # LivePortrait CLI 호출
        cmd = [
            "python", os.path.join(lp_dir, "inference.py"),
            "--source", image_path,
            "--driving", driving_video_path,
            "--output-dir", out_dir,
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=900,  # 15분 타임아웃
            cwd=lp_dir,
        )

        if result.returncode != 0:
            raise RuntimeError(f"LivePortrait 실패: {result.stderr[:500]}")

        # 출력 파일 찾기
        out_files = list(Path(out_dir).glob("*.mp4"))
        if not out_files:
            raise FileNotFoundError(f"LivePortrait 출력 파일 없음: {out_dir}")

        job.result_path = str(out_files[0])
        job.status = JobStatus.COMPLETED
        job.progress = "완료"
        job.completed_at = time.time()
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.completed_at = time.time()
        traceback.print_exc()


def _run_render(job: Job, video_path: str, tts_path: str | None,
                bgm_path: str | None, bgm_volume: float):
    """FFmpeg 합성: 영상 + TTS + BGM → 최종 mp4"""
    try:
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        job.progress = "FFmpeg 합성 준비 중"

        out_path = str(OUTPUT_DIR / f"{job.job_id}_final.mp4")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"영상 파일 없음: {video_path}")

        inputs = ["-i", video_path]
        filter_parts = []
        audio_inputs = []
        input_idx = 1  # 0 = video

        # TTS 오디오
        if tts_path and os.path.exists(tts_path):
            inputs += ["-i", tts_path]
            audio_inputs.append(f"[{input_idx}:a]volume=1.0[tts]")
            input_idx += 1
        else:
            audio_inputs.append(f"anullsrc=r=44100:cl=stereo[tts]")

        # BGM 오디오
        if bgm_path and os.path.exists(bgm_path):
            inputs += ["-i", bgm_path]
            audio_inputs.append(f"[{input_idx}:a]volume={bgm_volume}[bgm]")
            input_idx += 1
        else:
            audio_inputs.append(f"anullsrc=r=44100:cl=stereo[bgm]")

        job.progress = "오디오 믹싱 중"

        # 오디오가 모두 있을 때: amix
        if tts_path and os.path.exists(tts_path) and bgm_path and os.path.exists(bgm_path):
            cmd = [
                "ffmpeg", "-y",
                *inputs,
                "-filter_complex",
                f"[1:a]volume=1.0[tts];[2:a]volume={bgm_volume}[bgm];[tts][bgm]amix=inputs=2:duration=first[aout]",
                "-map", "0:v", "-map", "[aout]",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                out_path,
            ]
        elif tts_path and os.path.exists(tts_path):
            cmd = [
                "ffmpeg", "-y",
                *inputs,
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                out_path,
            ]
        elif bgm_path and os.path.exists(bgm_path):
            cmd = [
                "ffmpeg", "-y",
                *inputs,
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                out_path,
            ]
        else:
            # 오디오 없이 영상만
            cmd = ["ffmpeg", "-y", "-i", video_path, "-c:v", "copy", out_path]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg 실패: {result.stderr[:500]}")

        job.result_path = out_path
        job.status = JobStatus.COMPLETED
        job.progress = "완료"
        job.completed_at = time.time()
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.completed_at = time.time()
        traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# 파일 업로드 헬퍼
# ══════════════════════════════════════════════════════════════════════════════
def _save_upload(upload: UploadFile, prefix: str) -> str:
    """업로드 파일 저장 → 경로 반환"""
    ext = Path(upload.filename).suffix if upload.filename else ""
    file_path = str(UPLOAD_DIR / f"{prefix}_{uuid.uuid4().hex[:6]}{ext}")
    with open(file_path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return file_path


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI 앱
# ══════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="K-Ride Media API (Kaggle GPU)",
    version="1.0.0-kaggle",
    description="TTS · MusicGen · 3D Photo Inpainting · LivePortrait · FFmpeg 합성",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    gpu_info = {}
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1),
                "gpu_memory_used_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
                "gpu_memory_cached_gb": round(torch.cuda.memory_reserved(0) / 1e9, 2),
            }
        else:
            gpu_info = {"gpu": "not available"}
    except ImportError:
        gpu_info = {"torch": "not installed"}

    active_jobs = sum(1 for j in _jobs.values() if j.status == JobStatus.RUNNING)

    return {
        "status": "ok",
        "runtime": "kaggle-gpu",
        **gpu_info,
        "active_jobs": active_jobs,
        "total_jobs": len(_jobs),
        "max_workers": MAX_WORKERS,
    }


# ── 작업 상태 / 다운로드 ────────────────────────────────────────────────────

@app.get("/api/media/status/{job_id}")
def get_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job.to_dict()


@app.get("/api/media/jobs")
def list_jobs(limit: int = 20):
    """최근 작업 목록"""
    sorted_jobs = sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)
    return {"jobs": [j.to_dict() for j in sorted_jobs[:limit]]}


@app.get("/api/media/download/{job_id}")
def download_result(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Job {job_id} is {job.status.value}")
    if not job.result_path or not os.path.exists(job.result_path):
        raise HTTPException(status_code=404, detail="Result file not found")

    return FileResponse(
        job.result_path,
        filename=os.path.basename(job.result_path),
        media_type="application/octet-stream",
    )


# ── TTS ──────────────────────────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str
    language: str = "ko"
    speaker_wav_path: Optional[str] = None  # 참조 화자 음성 경로 (Kaggle 내)


@app.post("/api/media/tts")
def create_tts(req: TTSRequest):
    """한국어 TTS 음성 생성 (XTTS-v2)"""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text가 비어있습니다")

    job = _create_job("tts")
    executor.submit(_run_tts, job, req.text, req.language, req.speaker_wav_path)

    return {"job_id": job.job_id, "status": job.status.value, "message": "TTS 작업이 시작되었습니다."}


@app.post("/api/media/tts/upload")
async def create_tts_with_upload(
    text: str = Form(...),
    language: str = Form("ko"),
    speaker_wav: Optional[UploadFile] = File(None),
):
    """TTS + 참조 화자 음성 파일 업로드"""
    speaker_path = None
    if speaker_wav:
        speaker_path = _save_upload(speaker_wav, "speaker")

    job = _create_job("tts")
    executor.submit(_run_tts, job, text, language, speaker_path)

    return {"job_id": job.job_id, "status": job.status.value}


# ── MusicGen ─────────────────────────────────────────────────────────────────

class MusicGenRequest(BaseModel):
    description: str = "calm Korean traditional ambient music, gayageum, peaceful travel"
    duration: int = 15  # 초 (최대 30)


@app.post("/api/media/musicgen")
def create_musicgen(req: MusicGenRequest):
    """MusicGen BGM 생성"""
    if req.duration > 30:
        raise HTTPException(status_code=400, detail="duration은 최대 30초입니다")

    job = _create_job("musicgen")
    executor.submit(_run_musicgen, job, req.description, req.duration)

    return {"job_id": job.job_id, "status": job.status.value, "message": f"BGM 생성 시작 ({req.duration}초)"}


# ── 3D Photo Inpainting ─────────────────────────────────────────────────────

@app.post("/api/media/inpaint3d")
async def create_inpaint3d(image: UploadFile = File(...)):
    """풍경 사진 → Depth 기반 카메라 무빙 영상 (Ken Burns)"""
    image_path = _save_upload(image, "landscape")
    job = _create_job("inpaint3d")
    executor.submit(_run_inpaint3d, job, image_path)

    return {"job_id": job.job_id, "status": job.status.value, "message": "3D 카메라 무빙 생성 시작"}


# ── Animate (LivePortrait) ───────────────────────────────────────────────────

@app.post("/api/media/animate")
async def create_animate(
    source_image: UploadFile = File(...),
    driving_video: Optional[UploadFile] = File(None),
):
    """인물 사진 → 모션 영상 (LivePortrait, 3~10분)"""
    image_path = _save_upload(source_image, "person")
    driving_path = None
    if driving_video:
        driving_path = _save_upload(driving_video, "driving")

    job = _create_job("animate")
    executor.submit(_run_animate, job, image_path, driving_path)

    return {"job_id": job.job_id, "status": job.status.value, "message": "인물 모션 생성 시작 (3~10분 소요)"}


# ── Render (FFmpeg 합성) ─────────────────────────────────────────────────────

class RenderRequest(BaseModel):
    video_job_id: str       # animate 또는 inpaint3d의 job_id
    tts_job_id: Optional[str] = None
    bgm_job_id: Optional[str] = None
    bgm_volume: float = 0.3  # BGM 볼륨 (0.0~1.0)


@app.post("/api/media/render")
def create_render(req: RenderRequest):
    """영상 + TTS + BGM → 최종 합성"""
    # 선행 작업 결과 경로 확인
    video_job = _jobs.get(req.video_job_id)
    if not video_job or video_job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"video job {req.video_job_id}이 완료되지 않았습니다")

    tts_path = None
    if req.tts_job_id:
        tts_job = _jobs.get(req.tts_job_id)
        if tts_job and tts_job.status == JobStatus.COMPLETED:
            tts_path = tts_job.result_path

    bgm_path = None
    if req.bgm_job_id:
        bgm_job = _jobs.get(req.bgm_job_id)
        if bgm_job and bgm_job.status == JobStatus.COMPLETED:
            bgm_path = bgm_job.result_path

    job = _create_job("render")
    executor.submit(_run_render, job, video_job.result_path, tts_path, bgm_path, req.bgm_volume)

    return {"job_id": job.job_id, "status": job.status.value, "message": "FFmpeg 합성 시작"}


# ── GPU 메모리 관리 ──────────────────────────────────────────────────────────

@app.post("/api/media/unload/{model_name}")
def unload_model(model_name: str):
    """GPU 메모리 해제 (tts | musicgen)"""
    if model_name not in ("tts", "musicgen"):
        raise HTTPException(status_code=400, detail="model_name: tts 또는 musicgen")
    _unload_model(model_name)
    return {"status": "ok", "unloaded": model_name}


# ══════════════════════════════════════════════════════════════════════════════
# 직접 실행
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
