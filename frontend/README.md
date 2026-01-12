# MusicLab Frontend

Professional web interface for AI-powered music generation. Upload two audio tracks, select 16-second patterns, and generate unique music using our transformer model.

## 🎨 Features

### Current Implementation (Stage 1)
- ✅ **Dual Audio Upload**: Drag & drop or click to upload MP3, WAV, OGG files
- ✅ **Interactive Waveform Visualization**: Visual representation of audio tracks
- ✅ **16-Second Pattern Selection**: Precise slider controls to select interesting patterns
- ✅ **Audio Playback**: Built-in player with play/pause, time display, volume control
- ✅ **Pattern Preview**: Listen to selected sections before generation
- ✅ **Responsive Design**: Works on desktop, tablet, and mobile devices
- ✅ **Modern UI**: Dark theme with gradient accents, smooth animations

### Coming Soon (Stage 2)
- 🔄 **API Integration**: Connect to backend model for real music generation
- 🔄 **Download Generated Music**: Save results as WAV/MP3 files
- 🔄 **Generation History**: View and manage previous generations
- 🔄 **Advanced Settings**: Control noise levels, creativity parameters

## 📁 Files

```
frontend/
├── index.html      # Main HTML structure
├── styles.css      # Complete styling and animations
├── app.js          # JavaScript application logic
└── README.md       # This file
```

## 🚀 Quick Start

### Local Development

1. **Open in Browser**
   ```bash
   # Simply open index.html in your browser
   open frontend/index.html
   
   # Or use a local server (recommended):
   cd frontend
   python3 -m http.server 8000
   # Visit: http://localhost:8000
   ```

2. **Test the Interface**
   - Upload two audio files (minimum 16 seconds each)
   - Use the slider to select interesting 16-second patterns
   - Click "Preview Selection" to hear your choices
   - Click "Generate Music" to see the generation flow

## 🎯 How to Use

### Step 1: Upload Tracks
- Click "Upload" or drag & drop audio files into Track A and Track B areas
- Supported formats: MP3, WAV, OGG, M4A
- Minimum duration: 16 seconds

### Step 2: Select Patterns
Each track shows:
- **Waveform visualization** - Visual representation of audio
- **Time markers** - Track position indicators
- **Selection overlay** - Highlighted 16-second window (blue overlay)
- **Slider control** - Drag to move selection window
- **Time inputs** - Precise start/end time adjustment

### Step 3: Preview Selections
- Click "Preview Selection" to hear the 16-second pattern
- Adjust volume using the volume slider
- Fine-tune selection until satisfied

### Step 4: Generate
- Click "Generate Music" button when both tracks are ready
- Watch progress as AI processes your patterns
- Listen to and download the generated result

## 🎨 Design Features

### Color Scheme
- **Primary**: Indigo gradient (#6366f1 → #8b5cf6)
- **Accent**: Pink (#ec4899)
- **Background**: Dark (#0a0a0f)
- **Success**: Green (#10b981)

### Typography
- **Font**: Inter (Google Fonts)
- **Sizes**: Responsive from 0.75rem to 3rem
- **Weights**: 300-800 for different hierarchy levels

### Animations
- Smooth fade-in transitions
- Hover effects on interactive elements
- Loading spinners and progress bars
- Pulse animation for uploading status

## 🔧 Technical Details

### Audio Processing
- **Web Audio API**: Native browser audio processing
- **AudioContext**: 48kHz sample rate, stereo mixing
- **Waveform Rendering**: Canvas-based visualization (high DPI support)
- **Real-time Playback**: BufferSource nodes with gain control

### Selection System
- **16-second window**: Fixed duration, adjustable position
- **Precision controls**: Slider (0.1s steps) + manual input
- **Visual feedback**: Live overlay updates on waveform
- **Bounds checking**: Prevents invalid selections

### Browser Compatibility
- ✅ Chrome/Edge 90+ (recommended)
- ✅ Firefox 88+
- ✅ Safari 14+
- ⚠️ Mobile: iOS 14+, Android Chrome 90+

## 🔌 API Integration (Next Stage)

### Endpoint Structure (To Be Implemented)
```javascript
// Upload and process
POST /api/generate
Content-Type: multipart/form-data

{
  "track1": File,          // Selected 16s pattern from Track A
  "track1_start": 5.2,     // Start time in seconds
  "track1_end": 21.2,      // End time in seconds
  "track2": File,          // Selected 16s pattern from Track B
  "track2_start": 10.5,
  "track2_end": 26.5,
  "noise_level": 0.23      // Optional: Pure GAN noise fraction
}

Response:
{
  "status": "success",
  "audio_url": "/download/result_abc123.wav",
  "duration": 16.0,
  "job_id": "abc123"
}
```

### JavaScript Integration Points
Key functions ready for API connection:

1. **`generateMusic()`** (line ~615)
   - Currently: Simulates generation with progress bar
   - TODO: Replace with actual API call

2. **`downloadResult()`** (line ~756)
   - Currently: Shows alert
   - TODO: Download from API-provided URL

3. **File extraction** helpers needed:
   - Extract 16s segments from audio files
   - Convert to WAV format for model input
   - Handle base64 or Blob uploads

## 📊 Performance

### Optimizations
- **Canvas DPI scaling**: Sharp waveforms on Retina displays
- **Debounced slider updates**: Smooth interaction at 60fps
- **Lazy waveform rendering**: Only render visible tracks
- **Memory management**: Properly dispose AudioBufferSources

### File Size Limits
- Recommended: < 50MB per file
- Maximum: Browser-dependent (typically 100-500MB)
- Large files: May take longer to decode

## 🐛 Known Issues & Limitations

1. **Simulated Generation**: Currently shows progress bar without real processing
2. **Download Placeholder**: Download button shows alert instead of actual file
3. **No File Validation**: Trusts browser's audio decode (no format checks)
4. **Single Session**: No persistence across page reloads
5. **No Error Recovery**: Failed uploads require page refresh

## 🛠️ Development Roadmap

### Phase 1: Frontend Complete ✅
- [x] HTML structure
- [x] CSS styling
- [x] JavaScript functionality
- [x] Waveform visualization
- [x] Pattern selection
- [x] Audio playback

### Phase 2: Backend Integration (Next)
- [ ] API endpoint design
- [ ] File upload handling
- [ ] Model inference connection
- [ ] Result download mechanism
- [ ] Error handling & validation

### Phase 3: Enhanced Features
- [ ] User accounts & history
- [ ] Preset patterns library
- [ ] Advanced model parameters
- [ ] Multiple generation modes
- [ ] Social sharing

## 📝 Code Structure

### HTML (`index.html`)
- Semantic structure with header, main, footer
- Two identical track cards for symmetry
- Progressive disclosure (upload → player → generate → result)
- SVG icons for crisp visuals

### CSS (`styles.css`)
- CSS custom properties for theming
- Mobile-first responsive design
- BEM-like naming conventions
- Smooth transitions and animations

### JavaScript (`app.js`)
- Class-based architecture (`MusicLab`)
- Modular method organization
- Event-driven pattern
- Clean separation of concerns

## 🎓 Learning Resources

- [Web Audio API Guide](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)
- [Canvas Waveform Tutorial](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API)
- [File API Documentation](https://developer.mozilla.org/en-US/docs/Web/API/File_API)

## 📄 License

© 2025 MusicLab. Part of PowerMuse AI Music Generation project.

---

**Ready for Stage 2**: Backend API integration and model connection! 🚀
