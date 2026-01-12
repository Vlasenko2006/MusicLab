#!/bin/bash
# MusicLab AWS Deployment Script
# Instance: i-0ed8087de322eeff4

set -e

INSTANCE_ID="i-0ed8087de322eeff4"
SSH_KEY="$HOME/.ssh/sentiment-analysis-key.pem"

# Get current public IP dynamically
echo "🔍 Getting instance public IP..."
EC2_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

if [ "$EC2_IP" = "None" ] || [ -z "$EC2_IP" ]; then
    echo "❌ Instance has no public IP!"
    exit 1
fi

echo "🚀 Deploying MusicLab to AWS EC2"
echo "================================"
echo "Instance IP: $EC2_IP"
echo ""

# Test SSH connection
echo "1. Testing SSH connection..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 ubuntu@$EC2_IP "echo 'Connected!'" || {
    echo "❌ Cannot connect. Please check:"
    echo "   - Security group allows SSH from your IP"
    echo "   - Instance is running"
    exit 1
}

# Setup swap
echo ""
echo "2. Setting up 4GB swap..."
ssh -i "$SSH_KEY" ubuntu@$EC2_IP << 'ENDSSH'
    # Check if swap exists
    if [ -f /swapfile ]; then
        echo "Swap already exists"
    else
        echo "Creating 4GB swap file..."
        sudo fallocate -l 4G /swapfile
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile
        echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
        sudo sysctl vm.swappiness=10
        echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
    fi
    free -h | grep Swap
ENDSSH

# Install Docker
echo ""
echo "3. Installing Docker..."
ssh -i "$SSH_KEY" ubuntu@$EC2_IP << 'ENDSSH'
    if command -v docker &> /dev/null; then
        echo "Docker already installed: $(docker --version)"
    else
        echo "Installing Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker ubuntu
        sudo systemctl enable docker
        sudo systemctl start docker
        echo "Docker installed successfully"
    fi
ENDSSH

# Clone repository
echo ""
echo "4. Cloning MusicLab repository..."
ssh -i "$SSH_KEY" ubuntu@$EC2_IP << 'ENDSSH'
    if [ -d "PowerMuse" ]; then
        echo "Repository exists, pulling latest changes..."
        cd PowerMuse
        git pull origin main
    else
        echo "Cloning repository..."
        git clone https://github.com/Vlasenko2006/PowerMuse.git
        cd PowerMuse
    fi
ENDSSH

# Copy environment files and model checkpoint
echo ""
echo "5. Configuring environment and copying model checkpoint..."
scp -i "$SSH_KEY" .env ubuntu@$EC2_IP:~/PowerMuse/.env
scp -i "$SSH_KEY" backend/config/config_key.yaml ubuntu@$EC2_IP:~/PowerMuse/backend/config/config_key.yaml

# Copy model checkpoint (239MB - this will take ~30 seconds)
echo "   Copying model checkpoint (239MB)..."
ssh -i "$SSH_KEY" ubuntu@$EC2_IP "mkdir -p ~/PowerMuse/backend/checkpoints"
scp -i "$SSH_KEY" backend/checkpoints/best_model.pt ubuntu@$EC2_IP:~/PowerMuse/backend/checkpoints/best_model.pt

# Build and start containers
echo ""
echo "6. Building and starting Docker containers..."
echo "   ⏳ This will take 15-20 minutes (backend image is 3.5GB)..."
ssh -i "$SSH_KEY" ubuntu@$EC2_IP << 'ENDSSH'
    cd PowerMuse
    # Need to logout/login for docker group, so use sudo for first time
    sudo docker compose up -d --build
ENDSSH

# Wait for containers to be healthy
echo ""
echo "7. Waiting for containers to be ready..."
ssh -i "$SSH_KEY" ubuntu@$EC2_IP << 'ENDSSH'
    cd PowerMuse
    echo "Waiting for backend to be healthy (may take 2-3 minutes)..."
    for i in {1..60}; do
        if sudo docker compose ps | grep -q "healthy"; then
            echo "✅ Backend is healthy!"
            break
        fi
        echo -n "."
        sleep 5
    done
ENDSSH

# Verify deployment
echo ""
echo "8. Verifying deployment..."
ssh -i "$SSH_KEY" ubuntu@$EC2_IP << 'ENDSSH'
    cd PowerMuse
    echo ""
    echo "Container status:"
    sudo docker compose ps
    echo ""
    echo "Backend health check:"
    curl -s http://localhost:8001/health | python3 -m json.tool || echo "Backend not ready yet"
ENDSSH

echo ""
echo "================================"
echo "✅ Deployment complete!"
echo ""
echo "🌐 Access your application at:"
echo "   Frontend: http://$EC2_IP"
echo "   Backend:  http://$EC2_IP:8001"
echo "   API Docs: http://$EC2_IP:8001/docs"
echo ""
echo "📊 Monitor logs with:"
echo "   ssh -i $SSH_KEY ubuntu@$EC2_IP 'cd PowerMuse && sudo docker compose logs -f'"
echo ""
