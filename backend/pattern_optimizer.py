import torch
import numpy as np
import soundfile as sf
from skopt import gp_minimize
from skopt.space import Real
import hashlib


def find_optimal_patterns(audio1_path, audio2_path, model, encodec_model, device='cpu', discriminator=None, num_stages=1, method='hybrid', n_grid=10, n_bayesian=40, score_cache=None, track1_hash=None, track2_hash=None):
    """
    Find optimal 16-second patterns using hybrid grid + Bayesian optimization
    Uses discriminator (music critic) for quality scoring if available
    
    Args:
        num_stages: Number of cascade stages (1=single stage, >1=cascade mode)
        method: 'hybrid' (default), 'grid', or 'bayesian'
        n_grid: Grid points per dimension (hybrid/grid mode)
        n_bayesian: Additional Bayesian evaluations (hybrid mode)
        score_cache: Optional cache instance for storing evaluation scores
        track1_hash: Optional hash of track1 for cache keys
        track2_hash: Optional hash of track2 for cache keys
    
    Returns: (best_start1, best_start2, best_score)
    """
    print(f"[OPTIMIZER] Loading audio files...")
    
    # Load audio
    audio1, sr1 = sf.read(audio1_path)
    audio2, sr2 = sf.read(audio2_path)
    
    # Convert stereo to mono
    if audio1.ndim == 2:
        audio1 = audio1.mean(axis=1)
    if audio2.ndim == 2:
        audio2 = audio2.mean(axis=1)
    
    duration1 = len(audio1) / sr1
    duration2 = len(audio2) / sr2
    
    print(f"[OPTIMIZER] Audio loaded - Track1: {duration1:.1f}s, Track2: {duration2:.1f}s")
    
    # Search space (time positions in seconds)
    search_space = [
        Real(0, max(0, duration1 - 16), name='start1'),
        Real(0, max(0, duration2 - 16), name='start2')
    ]
    
    iteration_counter = [0]
    evaluation_history = []  # Track all evaluations
    
    # Calculate total evaluations based on method
    if method == 'hybrid':
        total_evals = n_grid * n_grid + n_bayesian
    elif method == 'grid':
        total_evals = n_grid * n_grid
    else:  # bayesian
        total_evals = n_bayesian
    
    def evaluate_pair(params):
        iteration_counter[0] += 1
        start1, start2 = params
        
        # Check cache first
        if score_cache and track1_hash and track2_hash:
            cached = score_cache.get(track1_hash, track2_hash, start1, start2)
            if cached:
                score = cached['score']
                print(f"[EVAL {iteration_counter[0]}/{total_evals}] CACHED: track1={start1:.1f}s, track2={start2:.1f}s → {score:.4f} ({score*100:.1f}%)")
                evaluation_history.append({
                    'iteration': iteration_counter[0],
                    'start1': start1,
                    'start2': start2,
                    'score': score,
                    'logit': None,  # Not available for cached results
                    'cached': True
                })
                return -score  # Negative for minimization
        
        print(f"[EVAL {iteration_counter[0]}/{total_evals}] Testing: track1={start1:.1f}s, track2={start2:.1f}s")
        
        # Extract 16s segments
        idx1_start = int(start1 * sr1)
        idx1_end = idx1_start + int(16 * sr1)
        idx2_start = int(start2 * sr2)
        idx2_end = idx2_start + int(16 * sr2)
        
        seg1 = audio1[idx1_start:idx1_end]
        seg2 = audio2[idx2_start:idx2_end]
        
        # Encode
        with torch.no_grad():
            t1 = torch.from_numpy(seg1).float().unsqueeze(0).unsqueeze(0).to(device)
            t2 = torch.from_numpy(seg2).float().unsqueeze(0).unsqueeze(0).to(device)
            
            enc1 = encodec_model.encoder(t1)
            enc2 = encodec_model.encoder(t2)
            
            # Generate output (check cascade mode vs single stage)
            if num_stages > 1:
                output = model(enc1, enc2)  # Cascade: two inputs
            else:
                output = model(enc1)  # Single stage: one input only
            if isinstance(output, tuple):
                output = output[0]
            
            # Decode
            result = encodec_model.decoder(output)
            result_audio = result.squeeze().cpu().numpy()
        
        # Score using discriminator (music critic) if available
        if discriminator is not None:
            disc_logit = discriminator(output).squeeze().item()
            disc_prob = torch.sigmoid(torch.tensor(disc_logit)).item()
            score = disc_prob
            print(f"[EVAL {iteration_counter[0]}/{total_evals}] Critic score: {score:.4f} ({score*100:.1f}%), logit={disc_logit:.3f}")
            
            # Cache the result
            if score_cache and track1_hash and track2_hash:
                score_cache.set(track1_hash, track2_hash, start1, start2, score)
            
            evaluation_history.append({
                'iteration': iteration_counter[0],
                'start1': start1,
                'start2': start2,
                'score': score,
                'logit': disc_logit,
                'cached': False
            })
        else:
            # Fallback: RMS + spectral diversity (old method)
            rms = np.sqrt(np.mean(result_audio ** 2))
            std = np.std(result_audio)
            score = rms * 10 + std * 5
            print(f"[EVAL {iteration_counter[0]}/{total_evals}] Acoustic score: {score:.3f} (rms={rms:.4f}, std={std:.4f})")
            evaluation_history.append({
                'iteration': iteration_counter[0],
                'start1': start1,
                'start2': start2,
                'score': score,
                'cached': False
            })
        
        return -score  # Negative for minimization
    
    # Choose optimization method
    if method == 'hybrid':
        print(f"[OPTIMIZER] HYBRID optimization: {n_grid}x{n_grid} grid + {n_bayesian} Bayesian = {total_evals} evaluations...")
        
        # Phase 1: Grid search for broad coverage
        print(f"[OPTIMIZER] Phase 1/2: Grid search ({n_grid}x{n_grid} = {n_grid*n_grid} evaluations)...")
        max_start1 = max(0, duration1 - 16)
        max_start2 = max(0, duration2 - 16)
        grid1 = np.linspace(0, max_start1, n_grid)
        grid2 = np.linspace(0, max_start2, n_grid)
        
        best_score = -float('inf')
        best_start1 = 0
        best_start2 = 0
        
        # Collect all grid evaluations for Bayesian initialization
        grid_points = []
        grid_scores = []
        
        for s1 in grid1:
            for s2 in grid2:
                score_neg = evaluate_pair([s1, s2])
                score = -score_neg
                grid_points.append([s1, s2])
                grid_scores.append(score_neg)  # Keep negative for gp_minimize
                if score > best_score:
                    best_score = score
                    best_start1 = s1
                    best_start2 = s2
        
        print(f"[OPTIMIZER] Phase 2/2: Bayesian refinement ({n_bayesian} additional evaluations)...")
        
        # Phase 2: Bayesian optimization starting from grid results
        result = gp_minimize(
            evaluate_pair,
            search_space,
            n_calls=n_bayesian,
            n_initial_points=0,  # Use grid results as initial points
            x0=grid_points,      # Initial points from grid
            y0=grid_scores,      # Initial scores from grid
            random_state=42,
            verbose=False
        )
        
        # Update best if Bayesian found better
        if -result.fun > best_score:
            best_start1, best_start2 = result.x
            best_score = -result.fun
        
    elif method == 'grid':
        print(f"[OPTIMIZER] Starting grid search ({n_grid}x{n_grid} = {n_grid*n_grid} evaluations)...")
        
        # Create grid of positions
        max_start1 = max(0, duration1 - 16)
        max_start2 = max(0, duration2 - 16)
        grid1 = np.linspace(0, max_start1, n_grid)
        grid2 = np.linspace(0, max_start2, n_grid)
        
        best_score = -float('inf')
        best_start1 = 0
        best_start2 = 0
        
        # Evaluate all grid combinations
        for s1 in grid1:
            for s2 in grid2:
                score_neg = evaluate_pair([s1, s2])
                score = -score_neg
                if score > best_score:
                    best_score = score
                    best_start1 = s1
                    best_start2 = s2
        
    else:  # Bayesian optimization
        print(f"[OPTIMIZER] Starting Bayesian optimization ({n_bayesian} iterations)...")
        
        result = gp_minimize(
            evaluate_pair,
            search_space,
            n_calls=n_bayesian,
            n_initial_points=min(3, n_bayesian),
            random_state=42,
            verbose=False
        )
        
        best_start1, best_start2 = result.x
        best_score = -result.fun
    
    print(f"\n[OPTIMIZER] Complete! Best: track1={best_start1:.1f}s, track2={best_start2:.1f}s, score={best_score:.3f}")
    
    # Display summary table of all evaluations
    print(f"\n{'='*90}")
    print(f"📊 EVALUATION SUMMARY - All {total_evals} Evaluations (Sorted by Quality)")
    if method == 'hybrid':
        print(f"Method: HYBRID | Grid: {n_grid}x{n_grid} ({n_grid*n_grid}) + Bayesian: {n_bayesian}")
    elif method == 'grid':
        print(f"Method: GRID | Points: {n_grid}x{n_grid}")
    else:
        print(f"Method: BAYESIAN | Points: {n_bayesian}")
    print(f"{'='*90}")
    print(f"{'Rank':<6} {'Iter':<6} {'Track1':<12} {'Track2':<12} {'Probability':<15} {'Quality':<10} {'Status'}")
    print(f"{'-'*90}")
    
    # Sort by score (descending)
    sorted_evals = sorted(evaluation_history, key=lambda x: x['score'], reverse=True)
    
    for rank, eval_data in enumerate(sorted_evals, 1):
        # Check if this is the best (use score comparison to avoid floating point issues)
        is_best = (rank == 1)  # The first one after sorting is the best
        status = "⭐ SELECTED" if is_best else ""
        prob_str = f"{eval_data['score']:.4f}"
        quality_pct = f"{eval_data['score']*100:.1f}%"
        
        print(f"{rank:<6} #{eval_data['iteration']:<5} {eval_data['start1']:<12.1f} {eval_data['start2']:<12.1f} {prob_str:<15} {quality_pct:<10} {status}")
    
    print(f"{'='*90}\n")
    
    # Return best results AND all evaluation history for download
    return best_start1, best_start2, best_score, evaluation_history
