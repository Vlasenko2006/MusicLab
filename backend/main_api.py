#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MusicLab Backend API - FastAPI server for music generation

Handles:
- Upload two audio tracks (any format: mp3, m4a, wav, etc.)
- Convert to WAV 24kHz mono
- Extract 16-second segments
- Encode with EnCodec
- Run through trained model
- Return generated audio
"""

print("DEBUG: Script started - importing FastAPI...")
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Form
print("DEBUG: FastAPI imported")
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional, Dict
import os
import uuid
import hashlib
import logging
import shutil
from datetime import datetime, timedelta
import asyncio
import time
import torch
import soundfile as sf
import numpy as np
import subprocess
from pydub import AudioSegment
import librosa
import sys
import os

# Ensure ffmpeg is in PATH (important for subprocess calls)
os.environ["PATH"] += os.pathsep + "/usr/local/bin"
print(f"DEBUG: Added /usr/local/bin to PATH")

# Import your model (now in same directory)
print("DEBUG: Importing model_simple_transformer...")
from model_simple_transformer import SimpleTransformer
print("DEBUG: Importing encodec...")
import encodec
print("DEBUG: Importing music chatbot...")
from music_chatbot import get_chatbot_instance
from score_cache import get_cache
from pattern_optimizer import find_optimal_patterns as find_patterns
from audio_discriminator import AudioDiscriminator
print("DEBUG: Imports successful")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("DEBUG: Creating FastAPI app...")
app = FastAPI(
    title="MusicLab API",
    description="AI-powered music generation from two audio tracks",
    version="1.0.0"
)
print("DEBUG: FastAPI app created")

# CORS middleware for frontend access
print("DEBUG: Adding CORS middleware...")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("DEBUG: CORS middleware added")

# Job storage (use Redis in production)
jobs_db: Dict[str, dict] = {}

# Model globals (loaded once at startup)
MODEL = None
ENCODEC_MODEL = None
DISCRIMINATOR = None
DEVICE = None
MODEL_ARGS = None


def cleanup_old_cache_files(max_age_hours: int = 24):
    """Remove cache files older than specified hours"""
    cache_dir = 'cache'
    if not os.path.exists(cache_dir):
        return 0
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    removed_count = 0
    
    try:
        for item in os.listdir(cache_dir):
            if not item.startswith('music_'):
                continue
                
            item_path = os.path.join(cache_dir, item)
            if not os.path.isdir(item_path):
                continue
            
            # Check directory age
            dir_age = current_time - os.path.getmtime(item_path)
            if dir_age > max_age_seconds:
                logger.info(f"Removing old cache directory: {item} (age: {dir_age/3600:.1f} hours)")
                shutil.rmtree(item_path)
                removed_count += 1
                
                # Also remove from jobs_db if present
                job_id = item.replace('music_', '')
                if job_id in jobs_db:
                    del jobs_db[job_id]
    except Exception as e:
        logger.error(f"Error during cache cleanup: {e}")
    
    return removed_count


async def periodic_cache_cleanup():
    """Background task to clean up old cache files every hour"""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            removed = cleanup_old_cache_files(max_age_hours=24)
            if removed > 0:
                logger.info(f"Periodic cleanup: removed {removed} old cache directories")
        except Exception as e:
            logger.error(f"Error in periodic cleanup task: {e}")


def load_models():
    """Load model and EnCodec at startup"""
    global MODEL, ENCODEC_MODEL, DISCRIMINATOR, DEVICE, MODEL_ARGS
    
    checkpoint_path = 'checkpoints/best_model.pt'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"DEBUG: Loading model from {checkpoint_path} on {DEVICE}")
    print(f"DEBUG: Checkpoint path exists: {os.path.exists(checkpoint_path)}")
    
    # Load checkpoint
    print("DEBUG: Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    print(f"DEBUG: Checkpoint keys: {checkpoint.keys()}")
    MODEL_ARGS = checkpoint.get('args', {})
    print(f"DEBUG: Model args: {MODEL_ARGS}")
    
    # Create model
    print("DEBUG: Creating SimpleTransformer model...")
    MODEL = SimpleTransformer(
        encoding_dim=MODEL_ARGS.get('encoding_dim', 128),
        nhead=MODEL_ARGS.get('nhead', 8),
        num_layers=MODEL_ARGS.get('num_layers', 4),
        num_transformer_layers=MODEL_ARGS.get('num_transformer_layers', 1),
        dropout=MODEL_ARGS.get('dropout', 0.1),
        anti_cheating=MODEL_ARGS.get('anti_cheating', 0.0),
        use_creative_agent=MODEL_ARGS.get('use_creative_agent', False),
        use_compositional_agent=MODEL_ARGS.get('use_compositional_agent', False)
    ).to(DEVICE)
    print("DEBUG: Model created, loading state dict...")
    
    MODEL.load_state_dict(checkpoint['model_state_dict'], strict=False)
    MODEL.eval()
    print("DEBUG: Model state loaded and set to eval mode")
    
    logger.info(f"  Model loaded: {sum(p.numel() for p in MODEL.parameters()):,} parameters")
    logger.info(f"  Cascade stages: {MODEL_ARGS.get('num_transformer_layers', 1)}")
    
    # Load discriminator (music critic) for quality evaluation
    logger.info("Loading discriminator (music critic)...")
    print("DEBUG: Creating AudioDiscriminator...")
    DISCRIMINATOR = AudioDiscriminator(encoding_dim=128).to(DEVICE)
    if 'discriminator_state_dict' in checkpoint:
        print("DEBUG: Loading discriminator state from checkpoint...")
        DISCRIMINATOR.load_state_dict(checkpoint['discriminator_state_dict'])
        logger.info("  ✓ Discriminator loaded from checkpoint")
    else:
        logger.warning("  ⚠ No discriminator in checkpoint, using untrained critic")
    DISCRIMINATOR.eval()
    print("DEBUG: Discriminator set to eval mode")
    
    # Load EnCodec
    logger.info("Loading EnCodec model...")
    print("DEBUG: Creating EnCodec model (24kHz)...")
    ENCODEC_MODEL = encodec.EncodecModel.encodec_model_24khz()
    print("DEBUG: Setting bandwidth to 6.0...")
    ENCODEC_MODEL.set_target_bandwidth(6.0)
    print(f"DEBUG: Moving EnCodec to {DEVICE}...")
    ENCODEC_MODEL = ENCODEC_MODEL.to(DEVICE)
    ENCODEC_MODEL.eval()
    print("DEBUG: Setting EnCodec to eval mode and freezing parameters...")
    
    for param in ENCODEC_MODEL.parameters():
        param.requires_grad = False
    
    logger.info("✓ Models loaded successfully")
    print("DEBUG: All models loaded successfully!")


@app.on_event("startup")
async def startup_event():
    """Load models when server starts"""
    load_models()
    # Start background cleanup task (removes files older than 24 hours)
    asyncio.create_task(periodic_cache_cleanup())


def convert_to_wav(input_path: str, output_path: str, sample_rate: int = 24000) -> str:
    """
    Convert any audio format to WAV 24kHz mono using ffmpeg
    
    This is the most AWS-friendly approach:
    - ffmpeg is widely available on AWS (can be installed via yum/apt)
    - Handles all formats (mp3, m4a, flac, ogg, etc.)
    - Fast and reliable
    - Easy to containerize in Docker
    
    Args:
        input_path: Path to input audio file (any format)
        output_path: Path to output WAV file
        sample_rate: Target sample rate (default: 24000 Hz for EnCodec)
    
    Returns:
        Path to output WAV file
    """
    logger.info(f"Converting {input_path} to WAV...")
    print(f"DEBUG: convert_to_wav - input: {input_path}, output: {output_path}")
    print(f"DEBUG: Input exists: {os.path.exists(input_path)}")
    
    # Check if input and output are the same
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        print(f"DEBUG: Input and output are the same, skipping conversion")
        logger.info(f"  ⚠️  Skipping conversion (already WAV): {output_path}")
        return output_path
    
    try:
        # Use ffmpeg for conversion
        # -nostdin: Don't expect any input from stdin (prevents hanging)
        # -i: input file
        # -ar: audio sample rate
        # -ac: audio channels (1 = mono)
        # -y: overwrite output file
        print(f"DEBUG: Running ffmpeg...")
        result = subprocess.run([
            'ffmpeg',
            '-nostdin',  # CRITICAL: Prevent ffmpeg from reading stdin
            '-i', input_path,
            '-ar', str(sample_rate),
            '-ac', '1',  # mono
            '-y',  # overwrite
            output_path
        ], check=True, capture_output=True, timeout=30)  # Remove text=True to get bytes
        
        print(f"DEBUG: ffmpeg completed successfully")
        logger.info(f"  ✓ Converted to WAV: {output_path}")
        return output_path
        
    except subprocess.TimeoutExpired:
        print(f"DEBUG: ffmpeg timeout!")
        logger.error(f"  ✗ ffmpeg timeout after 30s")
        raise RuntimeError("Audio conversion timeout (30s)")
    except subprocess.CalledProcessError as e:
        # Decode stderr safely, ignoring encoding errors
        stderr_msg = e.stderr.decode('utf-8', errors='ignore') if isinstance(e.stderr, bytes) else str(e.stderr)
        print(f"DEBUG: ffmpeg failed: {stderr_msg[:200]}")  # First 200 chars
        logger.error(f"  ✗ ffmpeg conversion failed")
        raise RuntimeError(f"Audio conversion failed: {stderr_msg[:100]}")
    except FileNotFoundError:
        print(f"DEBUG: ffmpeg not found")
        logger.error("  ✗ ffmpeg not found. Install with: apt-get install ffmpeg")
        raise RuntimeError("ffmpeg not installed. Required for audio conversion.")
    except UnicodeDecodeError as e:
        print(f"DEBUG: Unicode decode error in ffmpeg output: {e}")
        logger.error(f"  ✗ ffmpeg output encoding error")
        raise RuntimeError("Audio file has encoding issues. Please try a different file.")


def extract_audio_segment(wav_path: str, start_sec: float = 0.0, duration_sec: float = 16.0):
    """
    Extract 16-second segment from WAV file
    
    Args:
        wav_path: Path to WAV file
        start_sec: Start time in seconds
        duration_sec: Duration in seconds (default: 16.0)
    
    Returns:
        audio_array: Numpy array [samples]
        sample_rate: Sample rate (Hz)
    """
    logger.info(f"Extracting segment from {wav_path}: {start_sec}s to {start_sec + duration_sec}s")
    
    # Load audio
    audio, sr = sf.read(wav_path)
    
    # Convert stereo to mono if needed
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    
    # Extract segment
    start_frame = int(start_sec * sr)
    num_frames = int(duration_sec * sr)
    segment = audio[start_frame:start_frame + num_frames]
    
    # Pad if segment is shorter than requested
    if len(segment) < num_frames:
        segment = np.pad(segment, (0, num_frames - len(segment)), mode='constant')
    
    logger.info(f"  ✓ Extracted segment: {segment.shape} samples at {sr} Hz")
    return segment, sr


def generate_music(job_id: str, track1_path: str, track2_path: str, 
                   start_time_1: float, end_time_1: float,
                   start_time_2: float, end_time_2: float):
    """
    Background task for music generation
    
    Pipeline:
    1. Convert both tracks to WAV 24kHz mono
    2. Extract 16-second segments from specified times
    3. Encode with EnCodec
    4. Run through model
    5. Decode output
    6. Save as WAV
    7. Cleanup temporary files (but keep output)
    """
    try:
        print(f"DEBUG: Starting generation for job {job_id}")
        print(f"DEBUG: track1={track1_path} ({start_time_1}s-{end_time_1}s), track2={track2_path} ({start_time_2}s-{end_time_2}s)")
        
        jobs_db[job_id]['status'] = 'running'
        jobs_db[job_id]['progress'] = 10
        jobs_db[job_id]['message'] = 'Converting audio to WAV...'
        
        cache_dir = f"cache/music_{job_id}"
        print(f"DEBUG: Cache directory: {cache_dir}")
        
        # Convert to WAV (use different names to avoid overwriting input)
        wav1_converted = os.path.join(cache_dir, 'track1_converted.wav')
        wav2_converted = os.path.join(cache_dir, 'track2_converted.wav')
        
        print(f"DEBUG: Converting track1 ({track1_path}) to {wav1_converted}")
        convert_to_wav(track1_path, wav1_converted, sample_rate=24000)
        print(f"DEBUG: Converting track2 ({track2_path}) to {wav2_converted}")
        convert_to_wav(track2_path, wav2_converted, sample_rate=24000)
        
        # Use converted files for processing
        wav1_path = wav1_converted
        wav2_path = wav2_converted
        
        jobs_db[job_id]['progress'] = 30
        jobs_db[job_id]['message'] = 'Extracting audio segments...'
        
        # Extract 16-second segments from specified start times
        duration = 16.0
        print(f"DEBUG: Extracting segment from track1 ({start_time_1}s - {start_time_1+duration}s)")
        audio1, sr1 = extract_audio_segment(wav1_path, start_time_1, duration)
        print(f"DEBUG: Track1 audio shape: {audio1.shape}, sr: {sr1}")
        print(f"DEBUG: Extracting segment from track2 ({start_time_2}s - {start_time_2+duration}s)")
        audio2, sr2 = extract_audio_segment(wav2_path, start_time_2, duration)
        print(f"DEBUG: Track2 audio shape: {audio2.shape}, sr: {sr2}")
        
        # Convert to tensors
        print(f"DEBUG: Converting to tensors...")
        audio1_tensor = torch.from_numpy(audio1).float().unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1, 1, samples]
        audio2_tensor = torch.from_numpy(audio2).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        print(f"DEBUG: Tensor shapes: audio1={audio1_tensor.shape}, audio2={audio2_tensor.shape}")
        
        jobs_db[job_id]['progress'] = 50
        jobs_db[job_id]['message'] = 'Encoding with EnCodec...'
        
        # Encode with EnCodec
        print(f"DEBUG: Encoding with EnCodec...")
        with torch.no_grad():
            encoded1 = ENCODEC_MODEL.encoder(audio1_tensor)  # [1, 128, T]
            encoded2 = ENCODEC_MODEL.encoder(audio2_tensor)  # [1, 128, T]
            print(f"DEBUG: Encoded shapes: enc1={encoded1.shape}, enc2={encoded2.shape}")
            
            logger.info(f"  Encoded shapes: {encoded1.shape}, {encoded2.shape}")
            
            jobs_db[job_id]['progress'] = 70
            jobs_db[job_id]['message'] = 'Generating music...'
            
            # Run through model
            num_stages = MODEL_ARGS.get('num_transformer_layers', 1)
            print(f"DEBUG: Running model with {num_stages} cascade stages")
            
            if num_stages > 1:
                # Cascade mode: pass both inputs
                print(f"DEBUG: Using CASCADE mode (2 inputs)")
                result = MODEL(encoded1, encoded2)
                encoded_output = result[0] if isinstance(result, tuple) else result
            else:
                # Single stage: only first input
                print(f"DEBUG: Using SINGLE STAGE mode (1 input)")
                result = MODEL(encoded1)
                encoded_output = result[0] if isinstance(result, tuple) else result
            
            print(f"DEBUG: Model output shape: {encoded_output.shape}")
            
            logger.info(f"  Model output shape: {encoded_output.shape}")
            
            jobs_db[job_id]['progress'] = 85
            jobs_db[job_id]['message'] = 'Decoding audio...'
            
            # Decode to audio
            print(f"DEBUG: Decoding output with EnCodec decoder...")
            predicted_audio = ENCODEC_MODEL.decoder(encoded_output)
            print(f"DEBUG: Decoded audio shape: {predicted_audio.shape}")
            predicted_audio = predicted_audio.squeeze().cpu().numpy().astype(np.float32)
            print(f"DEBUG: Final audio shape after squeeze: {predicted_audio.shape}")
        
        # Normalize and clip
        print(f"DEBUG: Normalizing audio...")
        predicted_audio = np.clip(predicted_audio, -1.0, 1.0)
        
        jobs_db[job_id]['progress'] = 95
        jobs_db[job_id]['message'] = 'Saving output...'
        
        # Save output
        output_path = os.path.join(cache_dir, 'generated.wav')
        print(f"DEBUG: Saving to {output_path}")
        sf.write(output_path, predicted_audio, 24000, subtype='PCM_16')
        
        logger.info(f"  ✓ Generated audio saved: {output_path}")
        print(f"DEBUG: Generation complete for job {job_id}")
        
        # Update job status
        jobs_db[job_id]['status'] = 'completed'
        jobs_db[job_id]['progress'] = 100
        jobs_db[job_id]['message'] = 'Music generation complete!'
        jobs_db[job_id]['output_path'] = output_path
        
        # Cleanup temporary files (keep output for download)
        for temp_file in [track1_path, track2_path, wav1_path, wav2_path]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    logger.info(f"  🗑️  Cleaned up: {temp_file}")
                except Exception as e:
                    logger.warning(f"  ⚠️  Could not delete {temp_file}: {e}")
        
    except Exception as e:
        logger.error(f"Generation failed for job {job_id}: {e}", exc_info=True)
        jobs_db[job_id]['status'] = 'failed'
        jobs_db[job_id]['message'] = f'Error: {str(e)}'
        
        # Cleanup on failure
        cache_dir = f"cache/music_{job_id}"
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "MusicLab API",
        "version": "1.0.0",
        "endpoints": {
            "generate": "/api/generate",
            "status": "/api/status/{job_id}",
            "download": "/api/download/{job_id}",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check for Docker/AWS"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "encodec_loaded": ENCODEC_MODEL is not None,
        "device": str(DEVICE),
        "timestamp": datetime.now().isoformat()
    }



@app.post("/api/evaluate-segments")
async def evaluate_segments_endpoint(
    track1: UploadFile = File(...),
    track2: UploadFile = File(...),
    start1: float = Form(...),
    start2: float = Form(...)
):
    """Evaluate quality score for specific segment positions"""
    try:
        # Generate hashes for caching
        track1_bytes = await track1.read()
        track2_bytes = await track2.read()
        track1_hash = hashlib.md5(track1_bytes).hexdigest()[:8]
        track2_hash = hashlib.md5(track2_bytes).hexdigest()[:8]
        
        # Check cache first
        cache = get_cache()
        cached = cache.get(track1_hash, track2_hash, start1, start2)
        if cached:
            logger.info(f"📊 Cache hit: start1={start1:.1f}s, start2={start2:.1f}s, score={cached['score']:.3f}")
            return {
                "status": "success",
                "score": float(cached['score']),
                "cached": True
            }
        
        # Save uploaded files with original extensions
        import tempfile
        import os
        
        # Create temp files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as f1:
            track1_path = f1.name
            f1.write(track1_bytes)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as f2:
            track2_path = f2.name
            f2.write(track2_bytes)
        
        # Convert to WAV 24kHz
        track1_wav = track1_path + '.wav'
        track2_wav = track2_path + '.wav'
        convert_to_wav(track1_path, track1_wav, sample_rate=24000)
        convert_to_wav(track2_path, track2_wav, sample_rate=24000)
        
        # Load audio segments using soundfile
        audio1, sr1 = sf.read(track1_wav)
        audio2, sr2 = sf.read(track2_wav)
        
        # Convert stereo to mono if needed
        if audio1.ndim == 2:
            audio1 = audio1.mean(axis=1)
        if audio2.ndim == 2:
            audio2 = audio2.mean(axis=1)
        
        # Extract 16s segments
        start1_frame = int(start1 * sr1)
        start2_frame = int(start2 * sr2)
        duration_frames = int(16.0 * sr1)
        
        y1 = audio1[start1_frame:start1_frame + duration_frames]
        y2 = audio2[start2_frame:start2_frame + duration_frames]
        
        # Pad if needed
        if len(y1) < duration_frames:
            y1 = np.pad(y1, (0, duration_frames - len(y1)))
        if len(y2) < duration_frames:
            y2 = np.pad(y2, (0, duration_frames - len(y2)))
        
        # Generate music from these segments (same as optimizer does)
        with torch.no_grad():
            t1 = torch.from_numpy(y1).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            t2 = torch.from_numpy(y2).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            # Encode
            enc1 = ENCODEC_MODEL.encoder(t1)
            enc2 = ENCODEC_MODEL.encoder(t2)
            
            # Generate (check cascade mode vs single stage)
            num_stages = MODEL_ARGS.get('num_transformer_layers', 1)
            if num_stages > 1:
                output = MODEL(enc1, enc2)  # Cascade: two inputs
            else:
                output = MODEL(enc1)  # Single stage: one input only
            if isinstance(output, tuple):
                output = output[0]
            
            # Decode to audio
            result = ENCODEC_MODEL.decoder(output)
            result_audio = result.squeeze().cpu().numpy()
        
        # Score using DISCRIMINATOR (music critic)
        with torch.no_grad():
            disc_logit = DISCRIMINATOR(output).squeeze().item()
            disc_prob = torch.sigmoid(torch.tensor(disc_logit)).item()
        
        # Old acoustic score (for comparison)
        rms = np.sqrt(np.mean(result_audio ** 2))
        std = np.std(result_audio)
        acoustic_score = rms * 10 + std * 5
        
        # Use discriminator probability as the quality score
        score = disc_prob
        
        # Cache the score
        cache.set(track1_hash, track2_hash, start1, start2, score)
        
        # Debug console output (table format)
        print(f"\n{'='*80}")
        print(f"🎵 MUSIC QUALITY EVALUATION")
        print(f"{'-'*80}")
        print(f"Track 1 segment: {start1:.1f}s - {start1+16.0:.1f}s")
        print(f"Track 2 segment: {start2:.1f}s - {start2+16.0:.1f}s")
        print(f"\n{'Metric':<30} {'Value':<20} {'Info'}")
        print(f"{'-'*80}")
        print(f"{'Discriminator Logit':<30} {disc_logit:<20.4f} {'Raw output'}")
        print(f"{'Discriminator Probability':<30} {disc_prob:<20.4f} {'Is this music?'}")
        print(f"{'Quality Score (0-1)':<30} {score:<20.4f} {'Used for ranking'}")
        print(f"{'Quality Percentage':<30} {score*100:<20.1f}% {'Display value'}")
        print(f"\n{'[Acoustic Metrics - Legacy]':<50}")
        print(f"{'RMS Energy':<30} {rms:<20.4f}")
        print(f"{'Std Deviation':<30} {std:<20.4f}")
        print(f"{'Acoustic Score':<30} {acoustic_score:<20.3f} {'Old formula'}")
        print(f"{'='*80}\n")
        
        logger.info(f"📊 Critic score: {score:.4f} ({score*100:.1f}%), logit={disc_logit:.3f}, acoustic={acoustic_score:.3f}")
        
        # Cleanup
        for path in [track1_path, track2_path, track1_wav, track2_wav]:
            try:
                os.remove(path)
            except:
                pass
        
        return {
            "status": "success",
            "score": float(score),
            "cached": False
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/generate")
async def generate_music_endpoint(
    track1: UploadFile = File(...),
    track2: UploadFile = File(...),
    start_time_1: float = Form(0.0),
    end_time_1: float = Form(16.0),
    start_time_2: float = Form(0.0),
    end_time_2: float = Form(16.0),
    background_tasks: BackgroundTasks = None
):
    """
    Upload two audio tracks and generate new music
    
    Args:
        track1: First audio track (any format: mp3, m4a, wav, etc.)
        track2: Second audio track (any format)
        start_time_1: Start time for track 1 in seconds (default: 0.0)
        end_time_1: End time for track 1 in seconds (default: 16.0)
        start_time_2: Start time for track 2 in seconds (default: 0.0)
        end_time_2: End time for track 2 in seconds (default: 16.0)
    
    Returns:
        job_id: Unique job identifier for status polling
    """
    # Don't clear cache here - let evaluation cache persist between calls
    # This ensures consistent scoring when using the same positions multiple times
    
    job_id = str(uuid.uuid4())
    logger.info(f"📨 New generation request - Job ID: {job_id}")
    logger.info(f"   Track 1: {track1.filename} ({start_time_1}s - {end_time_1}s)")
    logger.info(f"   Track 2: {track2.filename} ({start_time_2}s - {end_time_2}s)")
    
    # Initialize job
    jobs_db[job_id] = {
        'job_id': job_id,
        'status': 'queued',
        'progress': 0,
        'message': 'Upload received, queued for processing...',
        'created_at': datetime.now().isoformat()
    }
    
    # Create cache directory
    cache_dir = f"cache/music_{job_id}"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Save uploaded files
    track1_path = os.path.join(cache_dir, f"track1{os.path.splitext(track1.filename)[1]}")
    track2_path = os.path.join(cache_dir, f"track2{os.path.splitext(track2.filename)[1]}")
    
    with open(track1_path, 'wb') as f:
        shutil.copyfileobj(track1.file, f)
    
    with open(track2_path, 'wb') as f:
        shutil.copyfileobj(track2.file, f)
    
    logger.info(f"   ✓ Files saved to {cache_dir}")
    
    # Start background processing
    background_tasks.add_task(
        generate_music,
        job_id, track1_path, track2_path, 
        start_time_1, end_time_1, start_time_2, end_time_2
    )
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Generation started"
    }


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """
    Get generation status
    
    Returns:
        status: 'queued' | 'running' | 'completed' | 'failed'
        progress: 0-100
        message: Status message
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs_db[job_id]


@app.get("/api/download/{job_id}")
async def download_result(job_id: str):
    """
    Download generated audio file
    
    Returns:
        WAV file (24kHz, mono, PCM_16)
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_db[job_id]
    
    if job['status'] != 'completed':
        raise HTTPException(
            status_code=400, 
            detail=f"Generation not complete. Status: {job['status']}"
        )
    
    output_path = job.get('output_path')
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        output_path,
        media_type='audio/wav',
        filename='generated_music.wav'
    )


@app.delete("/api/cleanup/{job_id}")
async def cleanup_job(job_id: str):
    """
    Cleanup job files and remove from database
    Call this after downloading the result
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Remove cache directory
    cache_dir = f"cache/music_{job_id}"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        logger.info(f"🗑️  Cleaned up cache directory: {cache_dir}")
    
    # Remove from database
    del jobs_db[job_id]
    
    return {"message": "Job cleaned up successfully"}


# ========================================
# Chatbot Endpoints
# ========================================

@app.post("/api/clear-cache")
async def clear_score_cache():
    """
    Clear the score cache database
    
    Called when tracks are cleared to prevent cached scores from 
    previous tracks interfering with new optimization runs.
    
    Returns:
        Success message with number of entries cleared
    """
    try:
        cache = get_cache()
        num_entries = len(cache.cache)
        cache.clear()
        logger.info(f"[CACHE] Cleared {num_entries} cache entries")
        return {
            "status": "success",
            "message": f"Cleared {num_entries} cached scores",
            "cleared": num_entries
        }
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat_with_rita(
    session_id: str = Form(...),
    message: str = Form(...),
    language: str = Form("en")
):
    """
    Chat with Rita AI assistant
    
    Args:
        session_id: Unique session identifier (generated by frontend)
        message: User's message/question
        language: User's current language preference (en, ru, de, fr, es, pt, ar, zh)
        
    Returns:
        AI response with conversation context in the appropriate language
    """
    try:
        logger.info(f"[Rita] Session: {session_id}, Language: {language}, Message: {message[:50]}...")
        chatbot = get_chatbot_instance()
        
        # Don't inject language instructions - let Rita's system prompt handle language detection
        # Rita will respond in the same language the user writes in
        response = chatbot.chat(session_id, message)
        return response
    except Exception as e:
        logger.error(f"Chatbot error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/chat/{session_id}")
async def clear_chat_session(session_id: str):
    """
    Clear chat history for a session
    
    Automatically called when user closes browser/tab
    """
    try:
        chatbot = get_chatbot_instance()
        chatbot.clear_session(session_id)
        return {"message": f"Session {session_id} cleared"}
    except Exception as e:
        logger.error(f"Clear session error: {e}")
        return {"message": "Session cleared (or didn't exist)"}


@app.get("/api/chat/{session_id}/info")
async def get_chat_session_info(session_id: str):
    """
    Get information about chat session
    
    Returns message count and exchange count
    """
    try:
        chatbot = get_chatbot_instance()
        info = chatbot.get_session_info(session_id)
        return info
    except Exception as e:
        logger.error(f"Session info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/examples/{example_name}")
async def get_example_audio(example_name: str):
    """
    Serve example audio files for the Examples modal
    
    Allowed files: input.wav, target.wav, output.wav
    """
    # Validate example name
    allowed_files = {
        'input': 'input.wav',
        'target': '1_noisy_target.wav',
        'output': '1_predicted.wav'
    }
    
    if example_name not in allowed_files:
        raise HTTPException(status_code=404, detail="Example not found")
    
    # Construct path to music_samples folder (copied to parent dir in Docker)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    music_samples_dir = os.path.join(current_dir, '..', 'music_samples')
    file_path = os.path.join(music_samples_dir, allowed_files[example_name])
    
    # Check if file exists
    if not os.path.exists(file_path):
        logger.error(f"Example file not found: {file_path}")
        raise HTTPException(status_code=404, detail="Example file not found")
    
    # Return file with proper headers
    return FileResponse(
        file_path,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'inline; filename="{allowed_files[example_name]}"',
            "Access-Control-Allow-Origin": "*"
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # Create cache directory
    os.makedirs("cache", exist_ok=True)
    
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8001)


# ========================================
# Pattern Optimization Endpoint
# ========================================

@app.post("/api/find-optimal-patterns")
async def find_optimal_patterns_endpoint(
    track1: UploadFile = File(...),
    track2: UploadFile = File(...),
    method: str = Form('hybrid'),     # 'hybrid', 'grid', or 'bayesian'
    n_grid: int = Form(10),           # 10x10=100 grid evaluations
    n_bayesian: int = Form(40)        # 40 additional Bayesian evaluations
):
    """
    Find optimal 16-second patterns using hybrid (grid + Bayesian), grid only, or Bayesian only
    Hybrid: 10x10 grid (100 evals) + 40 Bayesian refinements = 140 total evaluations (~23 min)
    """
    job_id = str(uuid.uuid4())
    logger.info(f"🔍 Pattern optimization ({method}, grid={n_grid}, bayesian={n_bayesian}) - Job ID: {job_id}")
    
    cache_dir = f"cache/optimize_{job_id}"
    os.makedirs(cache_dir, exist_ok=True)
    
    track1_path = os.path.join(cache_dir, f"track1{os.path.splitext(track1.filename)[1]}")
    track2_path = os.path.join(cache_dir, f"track2{os.path.splitext(track2.filename)[1]}")
    
    # Read track bytes for hashing
    track1.file.seek(0)
    track1_bytes = await track1.read()
    track2.file.seek(0)
    track2_bytes = await track2.read()
    
    track1_hash = hashlib.md5(track1_bytes).hexdigest()[:8]
    track2_hash = hashlib.md5(track2_bytes).hexdigest()[:8]
    
    with open(track1_path, 'wb') as f:
        f.write(track1_bytes)
    with open(track2_path, 'wb') as f:
        f.write(track2_bytes)
    
    wav1 = os.path.join(cache_dir, "track1.wav")
    wav2 = os.path.join(cache_dir, "track2.wav")
    convert_to_wav(track1_path, wav1, sample_rate=24000)
    convert_to_wav(track2_path, wav2, sample_rate=24000)
    
    try:
        if method == 'hybrid':
            total_evals = n_grid * n_grid + n_bayesian
        elif method == 'grid':
            total_evals = n_grid * n_grid
        else:
            total_evals = n_bayesian
        logger.info(f"Running {method} optimization ({total_evals} evaluations)...")
        logger.info(f"Using score cache (track1_hash={track1_hash}, track2_hash={track2_hash})")
        num_stages = MODEL_ARGS.get('num_transformer_layers', 1)
        cache = get_cache()
        start1, start2, score, evaluation_history = find_patterns(
            wav1, wav2, MODEL, ENCODEC_MODEL, DEVICE, 
            discriminator=DISCRIMINATOR, 
            num_stages=num_stages,
            method=method,
            n_grid=n_grid,
            n_bayesian=n_bayesian,
            score_cache=cache,
            track1_hash=track1_hash,
            track2_hash=track2_hash
        )
        
        shutil.rmtree(cache_dir)
        
        logger.info(f"✅ Optimal: track1={start1:.1f}s, track2={start2:.1f}s, score={score:.3f}")
        
        # Convert evaluation history for JSON response
        all_results = [
            {
                "iteration": eval_data["iteration"],
                "start1": eval_data["start1"],
                "start2": eval_data["start2"],
                "score": eval_data["score"]
            }
            for eval_data in evaluation_history
        ]
        
        return {
            "status": "success",
            "start_time_1": start1,
            "end_time_1": start1 + 16.0,
            "start_time_2": start2,
            "end_time_2": start2 + 16.0,
            "score": score,
            "all_results": all_results
        }
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        raise HTTPException(status_code=500, detail=str(e))
