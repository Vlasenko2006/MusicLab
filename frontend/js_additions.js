// Add to app.js in MusicLab class

async findOptimalPatterns() {
    const btn = document.getElementById('find-optimal-btn');
    const progress = document.getElementById('optimize-progress');
    const progressFill = document.getElementById('optimize-progress-fill');
    const status = document.getElementById('optimize-status');
    
    // Disable button
    btn.disabled = true;
    btn.style.opacity = '0.6';
    progress.style.display = 'block';
    
    const API_URL = window.location.hostname === 'localhost' 
        ? 'http://localhost:8001' 
        : `http://${window.location.hostname}:8001`;
    
    try {
        // Simulate progress (Bayesian optimization takes ~30-60s)
        let fakeProgress = 0;
        const progressInterval = setInterval(() => {
            fakeProgress += 2;
            if (fakeProgress <= 95) {
                progressFill.style.width = fakeProgress + '%';
                status.textContent = `Analyzing patterns... ${fakeProgress}%`;
            }
        }, 800);
        
        // Create FormData
        const formData = new FormData();
        formData.append('track1', this.tracks[1].file);
        formData.append('track2', this.tracks[2].file);
        
        // Call API
        const response = await fetch(`${API_URL}/api/find-optimal-patterns`, {
            method: 'POST',
            body: formData
        });
        
        clearInterval(progressInterval);
        
        if (!response.ok) {
            throw new Error('Optimization failed');
        }
        
        const result = await response.json();
        
        // Update progress to 100%
        progressFill.style.width = '100%';
        status.textContent = '✅ Found optimal patterns!';
        
        // Update sliders to optimal positions
        this.tracks[1].startTime = result.start_time_1;
        this.tracks[1].endTime = result.end_time_1;
        this.tracks[2].startTime = result.start_time_2;
        this.tracks[2].endTime = result.end_time_2;
        
        // Update UI
        this.updateTimeDisplay(1);
        this.updateTimeDisplay(2);
        this.drawWaveform(1);
        this.drawWaveform(2);
        
        // Hide progress after 2s
        setTimeout(() => {
            progress.style.display = 'none';
            btn.disabled = false;
            btn.style.opacity = '1';
        }, 2000);
        
    } catch (error) {
        progressFill.style.width = '0%';
        status.textContent = '❌ Error: ' + error.message;
        btn.disabled = false;
        btn.style.opacity = '1';
        
        setTimeout(() => {
            progress.style.display = 'none';
        }, 3000);
    }
}

// Show button when both tracks loaded
checkOptimizeButtonReady() {
    const btn = document.getElementById('find-optimal-btn');
    if (this.tracks[1].loaded && this.tracks[2].loaded) {
        btn.style.display = 'inline-block';
    } else {
        btn.style.display = 'none';
    }
}

// Add to constructor or init:
document.getElementById('find-optimal-btn').addEventListener('click', () => {
    this.findOptimalPatterns();
});
