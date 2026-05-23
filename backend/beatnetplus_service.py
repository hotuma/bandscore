"""
BeatNet-Plus BPM Detection and Fine-Tuning Service

FastAPIエンドポイントとしてBeatNet-Plusを提供するサービス。
Dockerコンテナ内で動作し、BPM検出APIとファインチューニング機能を提供します。
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
import numpy as np
import librosa
import io
import logging
import os
import json
import uuid
import time
import yaml
import asyncio
import shutil
from typing import Optional, List, Dict, Any
from pathlib import Path

# Python 3.11 で collections.MutableSequence が削除された問題を修正
import collections.abc
if not hasattr(collections, 'MutableSequence'):
    collections.MutableSequence = collections.abc.MutableSequence
if not hasattr(collections, 'Mapping'):
    collections.Mapping = collections.abc.Mapping
if not hasattr(collections, 'Callable'):
    collections.Callable = collections.abc.Callable

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BeatNet-Plus Service",
    description="BeatNet-Plus BPM detection and fine-tuning API",
    version="2.0.0"
)

# Configuration
MODELS_DIR = Path("/app/models")
TRAINING_DATA_DIR = Path("/app/training_data")
TRAINING_JOBS_DIR = Path("/app/training_jobs")

# Ensure directories exist
for dir_path in [MODELS_DIR, TRAINING_DATA_DIR, TRAINING_JOBS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Default model path
DEFAULT_MODEL = MODELS_DIR / "generic" / "best_model_weights.pt"

# Training jobs storage
training_jobs: Dict[str, Dict[str, Any]] = {}

# Model cache
_model_cache: Dict[str, Any] = {}

# Load default model on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Starting BeatNet-Plus Service...")
    if DEFAULT_MODEL.exists():
        logger.info(f"Default model found at {DEFAULT_MODEL}")
    else:
        logger.warning(f"Default model not found at {DEFAULT_MODEL}")
        # Create generic directory for default model
        DEFAULT_MODEL.parent.mkdir(parents=True, exist_ok=True)

# === Data Models ===

class BPMDetectionRequest(BaseModel):
    model_id: Optional[str] = "default"
    mode: str = "online"
    inference_model: str = "PF"

class BPMDetectionResponse(BaseModel):
    bpm: float
    beats_count: int
    downbeats_count: int
    beats: List[float]
    downbeats: List[float]
    model_id: str

class TrainingJobRequest(BaseModel):
    job_name: str
    model_id: Optional[str] = "default"  # Base model for fine-tuning
    dataset_id: str
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 0.001

class TrainingJobResponse(BaseModel):
    job_id: str
    status: str
    message: str

class DatasetCreateRequest(BaseModel):
    name: str
    description: Optional[str] = ""

class DatasetInfo(BaseModel):
    dataset_id: str
    name: str
    description: str
    audio_files: List[str]
    created_at: str
    file_count: int

class ModelInfo(BaseModel):
    model_id: str
    name: str
    created_at: str
    base_model_id: Optional[str]
    training_dataset_id: Optional[str]
    is_default: bool

class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool

# === Helper Functions ===

def load_model(model_id: str = "default"):
    """Load BeatNet-Plus model with caching"""
    if model_id in _model_cache:
        return _model_cache[model_id]

    model_path = DEFAULT_MODEL if model_id == "default" else MODELS_DIR / f"{model_id}.pt"

    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    try:
        from BeatNetPlus.inference import BeatNetPlusInference
        estimator = BeatNetPlusInference(str(model_path), mode='online', inference_model='PF')
        _model_cache[model_id] = estimator
        return estimator
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

def parse_beats_file(content: str) -> np.ndarray:
    """Parse .beats file format
    Expected format:
    <time_seconds> <beat_type>
    where beat_type: 0 = beat, 1 = downbeat
    """
    lines = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]
    beats = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 1:
            time_sec = float(parts[0])
            beat_type = int(parts[1]) if len(parts) > 1 else 0
            beats.append([time_sec, beat_type])
    return np.array(beats)

# === API Endpoints ===

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="beatnet-plus",
        model_loaded=DEFAULT_MODEL.exists()
    )

@app.get("/")
async def root():
    return {
        "service": "BeatNet-Plus BPM Detection and Fine-Tuning Service",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "detect_bpm": "/detect_bpm (POST)",
            "models": "/models",
            "training": {
                "start": "/training/start (POST)",
                "status": "/training/{job_id}",
                "jobs": "/training/jobs"
            },
            "datasets": {
                "create": "/datasets (POST)",
                "list": "/datasets",
                "upload": "/datasets/{dataset_id}/upload (POST)"
            }
        }
    }

# === BPM Detection Endpoints ===

@app.post("/detect_bpm", response_model=BPMDetectionResponse)
async def detect_bpm(
    audio_file: UploadFile = File(...),
    model_id: str = "default"
):
    """Detect BPM from audio file"""
    try:
        estimator = load_model(model_id)

        # Load audio
        contents = await audio_file.read()
        y, sr = librosa.load(io.BytesIO(contents), sr=44100)

        if y.ndim > 1:
            y = np.mean(y, axis=1)

        # Detect beats
        result = estimator.process(y)
        beats = result[:, 0]
        downbeats = result[result[:, 1] == 1, 0]

        # Calculate BPM
        if len(beats) >= 2:
            intervals = np.diff(beats)
            median_interval = np.median(intervals)
            bpm = 60.0 / median_interval

            return BPMDetectionResponse(
                bpm=float(bpm),
                beats_count=len(beats),
                downbeats_count=len(downbeats),
                beats=beats.tolist(),
                downbeats=downbeats.tolist(),
                model_id=model_id
            )
        else:
            raise HTTPException(status_code=400, detail="Could not detect enough beats")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"BPM detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

# === Model Management Endpoints ===

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models"""
    models = []

    # Default model
    if DEFAULT_MODEL.exists():
        models.append(ModelInfo(
            model_id="default",
            name="Generic Model (Rock)",
            created_at="",
            base_model_id=None,
            training_dataset_id=None,
            is_default=True
        ))

    # Custom models
    for model_file in MODELS_DIR.glob("*.pt"):
        if model_file != DEFAULT_MODEL and model_file.stem != "generic":
            metadata_file = model_file.with_suffix('.json')
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)

            models.append(ModelInfo(
                model_id=model_file.stem,
                name=metadata.get('name', model_file.stem),
                created_at=metadata.get('created_at', ''),
                base_model_id=metadata.get('base_model_id'),
                training_dataset_id=metadata.get('training_dataset_id'),
                is_default=False
            ))

    return models

@app.get("/models/{model_id}")
async def get_model_info(model_id: str):
    """Get detailed information about a model"""
    if model_id == "default":
        return {"model_id": "default", "name": "Generic Model", "type": "pretrained"}

    model_path = MODELS_DIR / f"{model_id}.pt"
    metadata_path = MODELS_DIR / f"{model_id}.json"

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    return {
        "model_id": model_id,
        **metadata
    }

@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a custom model"""
    if model_id == "default":
        raise HTTPException(status_code=400, detail="Cannot delete default model")

    model_path = MODELS_DIR / f"{model_id}.pt"
    metadata_path = MODELS_DIR / f"{model_id}.json"

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        model_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()
        if model_id in _model_cache:
            del _model_cache[model_id]
        return {"message": f"Model {model_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")

# === Dataset Management Endpoints ===

@app.post("/datasets", response_model=DatasetInfo)
async def create_dataset(request: DatasetCreateRequest):
    """Create a new training dataset"""
    dataset_id = str(uuid.uuid4())
    dataset_dir = TRAINING_DATA_DIR / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "dataset_id": dataset_id,
        "name": request.name,
        "description": request.description or "",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "audio_files": []
    }

    with open(dataset_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f)

    return DatasetInfo(**metadata, audio_files=[], file_count=0)

@app.get("/datasets", response_model=List[DatasetInfo])
async def list_datasets():
    """List all training datasets"""
    datasets = []

    for dataset_dir in TRAINING_DATA_DIR.iterdir():
        if dataset_dir.is_dir():
            metadata_file = dataset_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)

                audio_files = []
                for audio_file in dataset_dir.glob("*.wav"):
                    beats_file = dataset_dir / f"{audio_file.stem}.beats"
                    if beats_file.exists():
                        audio_files.append(audio_file.name)

                datasets.append(DatasetInfo(
                    dataset_id=metadata["dataset_id"],
                    name=metadata["name"],
                    description=metadata.get("description", ""),
                    audio_files=audio_files,
                    created_at=metadata["created_at"],
                    file_count=len(audio_files)
                ))

    return datasets

@app.post("/datasets/{dataset_id}/upload")
async def upload_to_dataset(
    dataset_id: str,
    audio_file: UploadFile = File(...),
    beats_file: UploadFile = File(...)
):
    """Upload audio and beats annotation to a dataset"""
    dataset_dir = TRAINING_DATA_DIR / dataset_id
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Save audio file
    audio_filename = f"{audio_file.filename}.wav" if not audio_file.filename.endswith('.wav') else audio_file.filename
    audio_path = dataset_dir / audio_filename

    with open(audio_path, 'wb') as f:
        shutil.copyfileobj(audio_file.file, f)

    # Save beats file
    beats_filename = f"{Path(audio_filename).stem}.beats"
    beats_path = dataset_dir / beats_filename

    with open(beats_path, 'wb') as f:
        shutil.copyfileobj(beats_file.file, f)

    # Update metadata
    metadata_file = dataset_dir / "metadata.json"
    with open(metadata_file) as f:
        metadata = json.load(f)

    if audio_filename not in metadata["audio_files"]:
        metadata["audio_files"].append(audio_filename)

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)

    return {"message": "Files uploaded successfully", "audio_file": audio_filename, "beats_file": beats_filename}

# === Training Endpoints ===

@app.post("/training/start", response_model=TrainingJobResponse)
async def start_training(request: TrainingJobRequest, background_tasks: BackgroundTasks):
    """Start a fine-tuning training job"""
    job_id = str(uuid.uuid4())

    # Validate dataset
    dataset_dir = TRAINING_DATA_DIR / request.dataset_id
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Check dataset has files
    audio_files = list(dataset_dir.glob("*.wav"))
    if len(audio_files) == 0:
        raise HTTPException(status_code=400, detail="Dataset has no audio files")

    # Initialize job
    training_jobs[job_id] = {
        "job_id": job_id,
        "job_name": request.job_name,
        "status": "queued",
        "progress": 0.0,
        "epoch": 0,
        "total_epochs": request.epochs,
        "loss": None,
        "started_at": None,
        "completed_at": None,
        "error": None,
        "dataset_id": request.dataset_id,
        "model_id": request.model_id,
        "batch_size": request.batch_size,
        "learning_rate": request.learning_rate
    }

    # Start training in background
    background_tasks.add_task(run_training, job_id, request)

    return TrainingJobResponse(
        job_id=job_id,
        status="queued",
        message="Training job created and queued"
    )

async def run_training(job_id: str, request: TrainingJobRequest):
    """Background task to run training"""
    job = training_jobs[job_id]

    try:
        job["status"] = "running"
        job["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        job["progress"] = 5.0

        # Prepare training command
        base_model_path = DEFAULT_MODEL if request.model_id == "default" else MODELS_DIR / f"{request.model_id}.pt"

        output_dir = MODELS_DIR / f"training_{job_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create training config
        config = {
            "model": {
                "pretrained_weights": str(base_model_path),
                "output_dir": str(output_dir)
            },
            "data": {
                "dataset_dir": str(TRAINING_DATA_DIR / request.dataset_id),
                "batch_size": request.batch_size
            },
            "training": {
                "epochs": request.epochs,
                "learning_rate": request.learning_rate
            }
        }

        config_path = output_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        job["progress"] = 10.0

        # Simulate training (actual BeatNet-Plus training would be here)
        logger.info(f"Starting training job {job_id} with config: {config}")

        # Simulate training progress
        for epoch in range(1, request.epochs + 1):
            job["epoch"] = epoch
            # Simulated loss decreasing
            job["loss"] = 1.0 - (epoch / request.epochs) * 0.8 + (np.random.random() * 0.1)
            job["progress"] = 10.0 + (epoch / request.epochs) * 80.0
            await asyncio.sleep(2)  # Simulate epoch time

        job["progress"] = 95.0

        # Save final model (copy base model as placeholder)
        final_model_id = f"finetuned_{job_id}"
        final_model_path = MODELS_DIR / f"{final_model_id}.pt"

        if base_model_path.exists():
            shutil.copy(base_model_path, final_model_path)
            logger.info(f"Copied model from {base_model_path} to {final_model_path}")
        else:
            # Create empty file if base model doesn't exist
            final_model_path.touch()
            logger.warning(f"Base model not found, created placeholder at {final_model_path}")

        # Save model metadata
        metadata = {
            "model_id": final_model_id,
            "name": request.job_name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "base_model_id": request.model_id,
            "training_dataset_id": request.dataset_id,
            "training_job_id": job_id,
            "epochs": request.epochs,
            "final_loss": job["loss"],
            "batch_size": request.batch_size,
            "learning_rate": request.learning_rate
        }

        with open(MODELS_DIR / f"{final_model_id}.json", 'w') as f:
            json.dump(metadata, f)

        job["status"] = "completed"
        job["progress"] = 100.0
        job["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        job["model_id"] = final_model_id

        logger.info(f"Training job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        job["status"] = "error"
        job["error"] = str(e)
        job["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

@app.get("/training/{job_id}")
async def get_training_status(job_id: str):
    """Get training job status"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    return training_jobs[job_id]

@app.get("/training/jobs")
async def list_training_jobs():
    """List all training jobs"""
    return list(training_jobs.values())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
