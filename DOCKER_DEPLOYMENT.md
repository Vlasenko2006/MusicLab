# 🐳 Docker Deployment Guide - MusicLab

**Complete guide for dockerizing and deploying MusicLab to AWS**

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER BROWSER                            │
│                    http://YOUR_IP or Domain                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                FRONTEND (nginx:alpine - Port 80)                │
│                                                                 │
│  • Serves static HTML/CSS/JavaScript                           │
│  • Language selector with 8 languages                          │
│  • Real-time audio waveform visualization                      │
│  • Examples and About modals with images                       │
│  • Proxies /api/ requests to backend                           │
└────────────────────────┬────────────────────────────────────────┘
                         │ Proxy /api/ → http://backend:8001
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│          PYTHON BACKEND (FastAPI + Uvicorn - Port 8001)        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. Load SimpleTransformer Model (16.7M params)         │   │
│  │    • EnCodec audio encoder/decoder                     │   │
│  │    • 239MB checkpoint loading (~15s)                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                         │                                       │
│  ┌─────────────────────▼───────────────────────────────────┐   │
│  │ 2. Audio Processing Pipeline                           │   │
│  │    • Upload & validate audio files (Track A & B)       │   │
│  │    • Extract 16-second segments                        │   │
│  │    • EnCodec encoding (24kHz mono)                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                         │                                       │
│  ┌─────────────────────▼───────────────────────────────────┐   │
│  │ 3. AI Music Generation                                 │   │
│  │    • Transformer model inference                       │   │
│  │    • Pattern fusion from both tracks                   │   │
│  │    • Generate new 16-second audio                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                         │                                       │
│  ┌─────────────────────▼───────────────────────────────────┐   │
│  │ 4. MusicNote AI Chatbot (Groq API)                    │   │
│  │    • Llama 3.3 70B model                               │   │
│  │    • Session-based conversations                       │   │
│  │    • Music theory knowledge base                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                         │                                       │
│  ┌─────────────────────▼───────────────────────────────────┐   │
│  │ 5. Result Delivery                                     │   │
│  │    • Stream generated audio as WAV                     │   │
│  │    • Real-time progress updates                        │   │
│  │    • Download functionality                            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start - Local Development

### Prerequisites
- Docker Desktop installed and running
- 8GB RAM minimum
- 20GB free disk space (model checkpoints)
- Groq API key (https://console.groq.com)

### One-Command Deployment

```bash
# Clone repository
git clone https://github.com/Vlasenko2006/PowerMuse.git
cd PowerMuse

# Create environment file
cp .env.example .env
nano .env  # Add your GROQ_API_KEY

# Build and start all services
docker compose up -d --build

# Wait for services to start (~60 seconds for model loading)
docker compose logs -f

# Access application
open http://localhost        # Production port 80
open http://localhost:3000   # Alternative port 3000
```

### Verify Deployment

```bash
# Check container status
docker ps

# Test backend health
curl http://localhost:8001/health
# Expected: {"status":"healthy","model_loaded":true}

# Test frontend
curl -I http://localhost
# Expected: HTTP/1.1 200 OK

# View logs
docker compose logs -f backend
docker compose logs -f frontend
```

---

## 📦 Project Structure

```
PowerMuse/
├── frontend/                    # Frontend application
│   ├── index.html              # Main HTML with multilingual support
│   ├── app.js                  # JavaScript with translation loading
│   ├── styles.css              # Styling
│   ├── locales/                # Translation JSON files
│   │   ├── en.json            # English
│   │   ├── de.json            # German
│   │   ├── ru.json            # Russian
│   │   ├── fr.json            # French
│   │   ├── es.json            # Spanish
│   │   ├── pt.json            # Portuguese
│   │   ├── ar.json            # Arabic
│   │   └── zh.json            # Chinese
│   └── Images/                 # About modal images
│       ├── 1.png              # Upload interface
│       ├── 2.png              # Pattern selection
│       └── 3.png              # Results interface
├── backend/                     # Python backend
│   ├── main_api.py            # FastAPI application (10 endpoints)
│   ├── music_chatbot.py       # Groq-powered chatbot
│   └── requirements.txt       # Python dependencies
├── config/                      # Configuration files
│   └── config_key.yaml        # API keys (gitignored)
├── checkpoints/                 # Model checkpoints
│   └── checkpoint_epoch_15_simple_transformer.pt  # 239MB
├── docker-compose.yml          # Multi-container orchestration
├── Dockerfile.backend          # Backend container definition
├── Dockerfile.frontend         # Frontend container definition
├── nginx.conf                  # Nginx reverse proxy config
└── .env                        # Environment variables (gitignored)
```

---

## 🔧 Configuration Files

### 1. `.env` File

Create `.env` in project root:

```bash
# Groq API Configuration
GROQ_API_KEY=gsk_your_api_key_here

# Optional: Email configuration for notifications
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your@email.com
SENDER_PASSWORD=your_app_password
```

### 2. `config/config_key.yaml`

```yaml
# API Keys and Configuration
groq:
  api_key: "${GROQ_API_KEY}"  # Loaded from environment variable
  model: "llama-3.3-70b-versatile"
  
chatbot:
  max_history: 10
  temperature: 0.7
  
audio:
  sample_rate: 24000
  segment_duration: 16
  max_file_size: 100  # MB
```

---

## ☁️ AWS EC2 Deployment Guide

### Prerequisites
- AWS account (Free Tier eligible)
- SSH key pair created in AWS console
- Basic Linux command knowledge

### Step 1: Launch EC2 Instance

**Instance Configuration:**
```
Instance Type: t3.medium (2 vCPU, 4GB RAM)
AMI: Ubuntu 24.04 LTS
Storage: 30GB EBS (General Purpose SSD)
Region: Choose closest to your users
```

**Security Group (Inbound Rules):**
```
SSH (22)      → Your IP only (e.g., 203.0.113.0/32)
HTTP (80)     → 0.0.0.0/0 (public access)
HTTPS (443)   → 0.0.0.0/0 (if using SSL)
```

**⚠️ Security:** Do NOT expose port 8001 to internet!

### Step 2: Allocate Elastic IP

```bash
# In AWS Console:
1. EC2 → Elastic IPs → "Allocate Elastic IP address"
2. Select new IP → Actions → "Associate Elastic IP address"
3. Choose your EC2 instance
4. Note the IP (e.g., 52.14.89.123)
```

**Benefits:**
- Permanent IP address (survives reboots)
- Free while associated with running instance
- Can point custom domains to it

### Step 3: Setup Swap Memory (Critical!)

AI models require more RAM than t3.medium provides. Add swap:

```bash
# SSH into EC2
ssh -i your-key.pem ubuntu@YOUR_ELASTIC_IP

# Create 4GB swap file
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Verify swap is active
free -h
# Should show 4.0Gi swap

# Make swap permanent (survives reboots)
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

**Why needed:**
- Model loading requires ~2GB memory
- Docker builds need extra memory
- Prevents OOM (Out of Memory) crashes

### Step 4: Install Docker

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group (no sudo needed)
sudo usermod -aG docker ubuntu
exit  # Log out and log back in

# Verify installation
docker --version
docker compose version
```

### Step 5: Deploy Application

```bash
# Clone repository
git clone https://github.com/Vlasenko2006/PowerMuse.git
cd PowerMuse

# Create .env file
nano .env
```

**Add to `.env`:**
```bash
GROQ_API_KEY=gsk_your_actual_api_key_here
```

```bash
# Create config directory if not exists
mkdir -p config

# Build and start services
docker compose up -d --build

# Monitor startup (model takes ~60s to load)
docker compose logs -f backend

# Check health
curl http://localhost:8001/health
```

### Step 6: Verify Deployment

**From your local machine:**
```bash
# Test frontend
open http://YOUR_ELASTIC_IP

# Should see MusicLab interface with:
- Language selector (8 languages)
- Track A and Track B upload areas
- Examples and About buttons
```

### Step 7: Auto-Start on Reboot

Services restart automatically with `restart: unless-stopped` in docker-compose.yml.

For guaranteed startup, create systemd service:

```bash
sudo nano /etc/systemd/system/musiclab.service
```

```ini
[Unit]
Description=MusicLab Docker Compose
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ubuntu/PowerMuse
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
User=ubuntu

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable musiclab
sudo systemctl start musiclab
sudo systemctl status musiclab
```

---

## 🔍 Monitoring & Troubleshooting

### Check Container Status

```bash
# List running containers
docker ps

# Expected output:
CONTAINER ID   IMAGE                    STATUS
abc123def456   musiclab-backend         Up 5 minutes (healthy)
789ghi012jkl   musiclab-frontend        Up 5 minutes (healthy)
```

### View Logs

```bash
# Backend logs (model loading, API requests)
docker compose logs -f backend

# Frontend logs (nginx access/error logs)
docker compose logs -f frontend

# Last 100 lines from both
docker compose logs --tail=100
```

### Common Issues

#### Issue 1: Backend "Unhealthy" Status

**Symptoms:**
```bash
docker ps
# STATUS: Up 30 seconds (unhealthy)
```

**Solution:**
```bash
# Check logs for model loading
docker compose logs backend | grep -i "model"

# Model takes ~15s to load (239MB checkpoint)
# Wait 60s total before declaring failure

# If OOM errors, increase swap:
sudo swapoff /swapfile
sudo fallocate -l 8G /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### Issue 2: Frontend Shows 502 Bad Gateway

**Symptoms:** Frontend loads but API calls fail

**Solution:**
```bash
# Check backend is running
curl http://localhost:8001/health

# Check network connectivity
docker exec musiclab-frontend ping -c 3 backend

# Restart services
docker compose restart
```

#### Issue 3: Translations Not Loading

**Symptoms:** UI stays in English, other languages don't work

**Cause:** Opening `file://` instead of `http://`

**Solution:** Always access via `http://YOUR_IP` or `http://localhost`

#### Issue 4: Audio Upload Fails

**Symptoms:** "Upload failed" or timeout errors

**Check:**
```bash
# Verify uploads directory exists and has permissions
docker exec musiclab-backend ls -la /app/uploads

# Check nginx max upload size (should be 100M)
docker exec musiclab-frontend grep client_max_body_size /etc/nginx/conf.d/default.conf
```

---

## 📊 Performance Optimization

### Resource Limits

Current limits in `docker-compose.yml`:
- **Backend:** 2GB memory (1GB reserved)
- **Frontend:** 256MB memory (128MB reserved)

Adjust based on your instance:

```yaml
# docker-compose.yml
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 4G  # For larger instances
          cpus: '2.0'
```

### Model Caching

First API call loads model (~15s). Subsequent calls are instant.

To pre-load model on startup, add to `Dockerfile.backend`:

```dockerfile
# Add before CMD
RUN python -c "import torch; torch.load('checkpoints/checkpoint_epoch_15_simple_transformer.pt', map_location='cpu')"
```

⚠️ This increases build time significantly.

---

## 🔒 Security Best Practices

### 1. Environment Variables

**Never commit:**
- `.env` file
- `config/config_key.yaml`
- API keys

**Add to `.gitignore`:**
```
.env
config/config_key.yaml
*.pem
*.key
```

### 2. API Key Protection

```bash
# Use environment variable, not hardcoded
export GROQ_API_KEY=$(cat .env | grep GROQ_API_KEY | cut -d= -f2)
docker compose up -d
```

### 3. Network Isolation

Backend is not exposed to internet:
- Only accessible via nginx proxy at `/api/`
- Port 8001 only open to `musiclab-network`

### 4. HTTPS Setup (Production)

```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Get certificate (requires domain name)
sudo certbot --nginx -d yourdomain.com

# Auto-renew
sudo certbot renew --dry-run
```

Update `docker-compose.yml`:
```yaml
frontend:
  ports:
    - "80:80"
    - "443:443"
  volumes:
    - /etc/letsencrypt:/etc/letsencrypt:ro
```

---

## 🧪 Testing

### Health Checks

```bash
# Backend health
curl http://localhost:8001/health
# Expected: {"status":"healthy","model_loaded":true,"timestamp":"..."}

# Frontend health
curl -I http://localhost
# Expected: HTTP/1.1 200 OK

# Check all services
docker compose ps
# All should show "healthy" status
```

### Load Testing

```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Test frontend (100 requests, 10 concurrent)
ab -n 100 -c 10 http://localhost/

# Test backend API
ab -n 50 -c 5 http://localhost:8001/health
```

---

## 📈 Scaling

### Horizontal Scaling (Multiple Backend Instances)

```yaml
# docker-compose.yml
services:
  backend:
    deploy:
      replicas: 3  # Run 3 backend instances
```

Update nginx to load balance:

```nginx
# nginx.conf
upstream backend_servers {
    server backend:8001;
    server backend:8002;
    server backend:8003;
}

location /api/ {
    proxy_pass http://backend_servers;
    # ... rest of config
}
```

### Vertical Scaling

Upgrade EC2 instance:
- **t3.medium** (2 vCPU, 4GB) → **t3.large** (2 vCPU, 8GB)
- Or **t3.xlarge** (4 vCPU, 16GB) for high traffic

---

## 🛠️ Maintenance

### Update Application

```bash
# Pull latest changes
cd /home/ubuntu/PowerMuse
git pull origin main

# Rebuild and restart
docker compose down
docker compose up -d --build

# Verify
docker compose ps
```

### Backup Data

```bash
# Backup uploads
docker run --rm -v PowerMuse_uploads:/data -v $(pwd):/backup ubuntu tar czf /backup/uploads-backup.tar.gz /data

# Backup configuration
tar czf config-backup.tar.gz .env config/
```

### Cleanup Old Images

```bash
# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Full cleanup (BE CAREFUL!)
docker system prune -a --volumes
```

---

## 📞 Support

**Issues:** https://github.com/Vlasenko2006/PowerMuse/issues  
**Documentation:** See `README.md` and `HTML_SESSION_CHECKPOINT.md`  
**Model Details:** SimpleTransformer (16.7M parameters), EnCodec audio encoding

**Common URLs:**
- **Frontend:** http://YOUR_IP or http://localhost
- **Backend API Docs:** http://localhost:8001/docs (local only)
- **Backend Health:** http://localhost:8001/health

---

## ✅ Deployment Checklist

### Pre-Deployment
- [ ] Groq API key obtained
- [ ] `.env` file created with API key
- [ ] Docker installed and running
- [ ] Sufficient disk space (20GB+)
- [ ] Swap memory configured (4GB+)

### Deployment
- [ ] Git repository cloned
- [ ] Docker images built successfully
- [ ] Containers started and healthy
- [ ] Backend health check passes
- [ ] Frontend accessible via browser

### Post-Deployment
- [ ] All 8 languages work
- [ ] Examples modal displays correctly
- [ ] About modal shows 3 images
- [ ] Audio upload and processing works
- [ ] MusicNote chatbot responds
- [ ] Download functionality tested

### AWS Production
- [ ] Elastic IP allocated and associated
- [ ] Security group configured correctly
- [ ] Auto-start systemd service enabled
- [ ] Backups scheduled
- [ ] Monitoring configured

---

**Last Updated:** December 18, 2025  
**Version:** 2.0 - Docker + AWS Ready
