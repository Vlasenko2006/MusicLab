# 🚀 AWS EC2 Deployment Guide for MusicLab

Deploy MusicLab to AWS EC2 t3.micro (Free Tier) with Docker.

---

## Prerequisites

- AWS account with Free Tier
- Basic AWS knowledge (EC2, Security Groups)
- SSH key pair for EC2 access

---

## Step 1: Launch EC2 Instance

### Instance Configuration

1. **Go to EC2 Dashboard** → Launch Instance

2. **Choose settings:**
   - **Name**: `musiclab-production`
   - **AMI**: Ubuntu Server 24.04 LTS (Free Tier eligible)
   - **Instance Type**: `t3.micro` (1 vCPU, 1GB RAM)
   - **Key pair**: Select existing or create new SSH key
   - **Storage**: 20GB gp3 (default)

3. **Network Settings** - Configure Security Group:

```
Inbound Rules:
┌──────────┬────────┬────────────┬─────────────────────────────────┐
│ Type     │ Port   │ Source     │ Description                     │
├──────────┼────────┼────────────┼─────────────────────────────────┤
│ SSH      │ 22     │ My IP      │ SSH access (your IP only)       │
│ HTTP     │ 80     │ 0.0.0.0/0  │ Frontend public access          │
│ Custom   │ 8001   │ 0.0.0.0/0  │ Backend API (optional: restrict)│
└──────────┴────────┴────────────┴─────────────────────────────────┘

Outbound Rules:
- All traffic → 0.0.0.0/0 (default)
```

⚠️ **Security Note**: For production, restrict port 8001 to frontend IP only, but for testing you can allow public access.

4. **Launch Instance**

---

## Step 2: Allocate Elastic IP (Permanent Address)

Elastic IP ensures your instance keeps the same public IP after restarts.

```bash
# In AWS Console:
1. EC2 → Elastic IPs → "Allocate Elastic IP address"
2. Select new IP → Actions → "Associate Elastic IP address"
3. Choose your instance: musiclab-production
4. Note the IP address (e.g., 13.48.16.109)
```

**Benefits:**
- ✅ Permanent IP (survives instance stop/start)
- ✅ Free while associated with running instance
- ✅ Can point custom domain to it

---

## Step 3: Connect to Instance

```bash
# Replace with your key file and Elastic IP
ssh -i ~/.ssh/your-key.pem ubuntu@YOUR_ELASTIC_IP

# Example:
ssh -i ~/.ssh/musiclab-aws.pem ubuntu@13.48.16.109
```

---

## Step 4: Setup 4GB Swap Memory (CRITICAL)

**Why needed:**
- Docker build requires ~2GB memory (peaks during backend build)
- Model loading (PyTorch + EnCodec) needs ~1GB
- Total: Need 3GB, but we only have 1GB RAM → 4GB swap ensures safety

```bash
# Create 4GB swap file
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Verify swap is active
free -h
# Should show:
#               total        used        free      shared  buff/cache   available
# Mem:           951Mi       180Mi       600Mi       1.0Mi       170Mi       620Mi
# Swap:          4.0Gi          0B       4.0Gi

# Make swap permanent (survives reboots)
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Optimize swap usage (use swap only when really needed)
sudo sysctl vm.swappiness=10
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
```

---

## Step 5: Install Docker & Docker Compose

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add ubuntu user to docker group (no need for sudo docker)
sudo usermod -aG docker ubuntu

# Start Docker service
sudo systemctl enable docker
sudo systemctl start docker

# IMPORTANT: Log out and back in for group changes to take effect
exit

# SSH back in
ssh -i ~/.ssh/your-key.pem ubuntu@YOUR_ELASTIC_IP

# Verify Docker works without sudo
docker --version
# Docker version 27.x.x

docker compose version
# Docker Compose version v2.x.x
```

---

## Step 6: Clone Repository

```bash
# Install Git (if not installed)
sudo apt install -y git

# Clone MusicLab repository
git clone https://github.com/Vlasenko2006/PowerMuse.git
cd PowerMuse

# Verify files
ls -la
# Should see: docker-compose.yml, Dockerfile.backend, Dockerfile.frontend, backend/, frontend/
```

---

## Step 7: Configure Environment

```bash
# Create .env file with your Groq API key
nano .env
```

Paste this content (replace with your actual API key):
```env
GROQ_API_KEY=your_groq_api_key_here
BACKEND_PORT=8001
FRONTEND_PORT=80
```

Save and exit (Ctrl+X, then Y, then Enter)

```bash
# Create backend config with API key
nano backend/config/config_key.yaml
```

Paste this content (replace with your actual API key):
```yaml
groq:
  api_key: "your_groq_api_key_here"
  model: "llama-3.3-70b-versatile"

musiclab:
  system_prompt: |
    You are Rita (Рита in Russian), an expert music AI assistant for MusicLab.
    
    Your role:
    - ALWAYS respond in the language specified by the user's language preference
    - If user asks to speak in a specific language, switch to that language immediately
    - When responding in Russian, introduce yourself as "Рита" (Rita)
    - Help users understand how MusicLab works
    - Explain pattern selection and fusion techniques
    - Provide music theory insights
    - Answer technical questions about the AI model
    - Be friendly, knowledgeable, and encouraging
    
    MusicLab Overview:
    - Users upload two 16-second audio patterns
    - AI fuses them into a new unique composition
    - Uses transformer-based architecture with EnCodec
    - Supports various musical styles and combinations
    
    Key Features:
    - Pattern fusion preserves characteristics of both inputs
    - Sliding window for precise segment selection
    - Real-time preview and playback
    - Download results as WAV files
    
    Always be helpful, clear, and passionate about music creation!
```

Save and exit.

---

## Step 8: Build and Deploy

**Note:** Initial build takes ~15-20 minutes on t3.micro due to:
- Backend: 3.5GB image (PyTorch CPU, models, dependencies)
- Frontend: 88MB image (nginx + static files)

```bash
# Build and start containers
docker compose up -d --build

# Monitor build progress (optional)
docker compose logs -f

# Wait for completion - you'll see:
# ✔ Container musiclab-backend   Started
# ✔ Container musiclab-frontend  Started
```

**Expected build time:**
- Backend: ~12-15 minutes (downloading PyTorch, installing deps, copying model)
- Frontend: ~30 seconds
- Total: ~15-20 minutes

---

## Step 9: Verify Deployment

```bash
# Check containers are running
docker compose ps
# Should show both containers as "Up" and "healthy"

# Check backend health
curl http://localhost:8001/health | python3 -m json.tool
# Should return:
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "encodec_loaded": true,
#   "device": "cpu",
#   "timestamp": "2025-12-19T..."
# }

# Check frontend
curl http://localhost
# Should return HTML content
```

---

## Step 10: Access Your Application

Open your browser:

**Frontend:** `http://YOUR_ELASTIC_IP`
- Example: http://13.48.16.109

**Backend API:** `http://YOUR_ELASTIC_IP:8001`
- Health: http://13.48.16.109:8001/health
- Docs: http://13.48.16.109:8001/docs

---

## Management Commands

```bash
# View logs
docker compose logs -f                 # All containers
docker compose logs -f backend        # Backend only
docker compose logs -f frontend       # Frontend only

# Restart services
docker compose restart

# Stop services
docker compose stop

# Start services
docker compose start

# Rebuild after code changes
git pull origin main
docker compose down
docker compose up -d --build

# Check disk usage
df -h
docker system df

# Clean up unused images/containers (frees space)
docker system prune -f
```

---

## Monitoring & Troubleshooting

### Memory Usage

```bash
# Check system memory
free -h

# Check Docker container memory
docker stats

# If backend crashes with OOM:
# 1. Increase swap to 8GB:
sudo swapoff /swapfile
sudo fallocate -l 8G /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Container Issues

```bash
# If backend is unhealthy:
docker logs musiclab-backend --tail 100

# Common issues:
# - Model download timeout → wait longer (EnCodec downloads ~50MB)
# - OOM error → increase swap
# - Port already in use → docker compose down && docker compose up -d

# If frontend shows 502 Bad Gateway:
# - Backend not ready yet (wait 2-3 minutes after start)
# - Check backend health: curl http://localhost:8001/health
```

### Performance Tips

```bash
# Reduce memory usage in docker-compose.yml:
# backend:
#   deploy:
#     resources:
#       limits:
#         memory: 1.5G  # Instead of 2G

# Frontend can use less:
# frontend:
#   deploy:
#     resources:
#       limits:
#         memory: 128M  # Instead of 256M
```

---

## Security Hardening (Production)

```bash
# 1. Setup firewall
sudo ufw enable
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 8001/tcp  # Backend API

# 2. Auto-updates
sudo apt install unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades

# 3. Change default SSH port (optional)
sudo nano /etc/ssh/sshd_config
# Change: Port 22 → Port 2222
sudo systemctl restart sshd

# 4. Setup HTTPS with Let's Encrypt (if you have domain)
# See: https://certbot.eff.org/
```

---

## Cost Optimization

**Free Tier Usage (12 months):**
- ✅ t3.micro: 750 hours/month (1 instance 24/7)
- ✅ 30GB EBS storage
- ✅ Elastic IP (free while associated)
- ✅ 100GB outbound data/month

**After Free Tier (~$9-12/month):**
- t3.micro: ~$7.50/month
- 20GB EBS: ~$2/month
- Data transfer: ~$1-3/month

**Reduce costs:**
- Stop instance when not in use (Elastic IP stays free)
- Use t3.micro only (don't upgrade unless needed)
- Monitor CloudWatch for usage

---

## Updating the Application

```bash
# 1. SSH to instance
ssh -i ~/.ssh/your-key.pem ubuntu@YOUR_ELASTIC_IP

# 2. Navigate to project
cd PowerMuse

# 3. Pull latest changes
git pull origin main

# 4. Rebuild and restart
docker compose down
docker compose up -d --build

# 5. Verify
curl http://localhost:8001/health
```

---

## Backup & Recovery

```bash
# Backup configuration
cp .env .env.backup
cp backend/config/config_key.yaml backend/config/config_key.yaml.backup

# Backup uploads (if any)
tar -czf uploads_backup_$(date +%Y%m%d).tar.gz backend/uploads/

# Download backup to local machine
scp -i ~/.ssh/your-key.pem ubuntu@YOUR_ELASTIC_IP:~/PowerMuse/uploads_backup*.tar.gz ~/backups/
```

---

## Next Steps

1. ✅ **Test all features** - upload tracks, generate music, chatbot
2. 🌐 **Setup custom domain** (optional):
   - Point A record to Elastic IP
   - Setup nginx reverse proxy
   - Add SSL with Let's Encrypt
3. 📊 **Monitor performance**:
   - CloudWatch metrics
   - Docker stats
   - Application logs
4. 🔒 **Enhance security**:
   - Restrict port 8001 to localhost only
   - Setup HTTPS
   - Enable AWS CloudWatch alarms

---

## Support

- **GitHub**: https://github.com/Vlasenko2006/PowerMuse
- **Issues**: Report bugs via GitHub Issues
- **Email**: [Your support email]

---

**Deployed Successfully! 🎉**

Your MusicLab is now live at: `http://YOUR_ELASTIC_IP`
