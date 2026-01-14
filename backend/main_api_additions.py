# Add to backend/main_api.py after the /api/generate endpoint

from pattern_optimizer import find_optimal_patterns

@app.post("/api/find-optimal-patterns")
async def find_optimal_patterns_endpoint(
    track1: UploadFile = File(...),
    track2: UploadFile = File(...)
):
    """
    Find optimal 16-second patterns in uploaded tracks using Bayesian optimization
    """
    job_id = str(uuid.uuid4())
    logger.info(f"🔍 Pattern optimization request - Job ID: {job_id}")
    
    # Save uploaded files
    cache_dir = f"cache/optimize_{job_id}"
    os.makedirs(cache_dir, exist_ok=True)
    
    track1_path = os.path.join(cache_dir, f"track1{os.path.splitext(track1.filename)[1]}")
    track2_path = os.path.join(cache_dir, f"track2{os.path.splitext(track2.filename)[1]}")
    
    with open(track1_path, 'wb') as f:
        shutil.copyfileobj(track1.file, f)
    with open(track2_path, 'wb') as f:
        shutil.copyfileobj(track2.file, f)
    
    # Convert to WAV
    wav1 = os.path.join(cache_dir, "track1.wav")
    wav2 = os.path.join(cache_dir, "track2.wav")
    convert_to_wav(track1_path, wav1, sample_rate=24000)
    convert_to_wav(track2_path, wav2, sample_rate=24000)
    
    try:
        logger.info(f"Running Bayesian optimization (9 grid + 40 Bayes iterations)...")
        start1, start2, score = find_optimal_patterns(
            wav1, wav2, MODEL, ENCODEC_MODEL, DEVICE
        )
        
        # Cleanup
        shutil.rmtree(cache_dir)
        
        logger.info(f"✅ Found optimal patterns: track1={start1:.1f}s, track2={start2:.1f}s, score={score:.3f}")
        
        return {
            "status": "success",
            "start_time_1": start1,
            "end_time_1": start1 + 16.0,
            "start_time_2": start2,
            "end_time_2": start2 + 16.0,
            "score": score
        }
    except Exception as e:
        logger.error(f"Pattern optimization error: {e}")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        raise HTTPException(status_code=500, detail=str(e))
