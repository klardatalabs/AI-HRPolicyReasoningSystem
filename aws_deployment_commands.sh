# instantiate EC2 server (t3 micro), create and download a key pair locally for ssh (do this via console)

# ssh into server (use git bash if operating from windows) --> get this from ec2 console

# generate deploy key for git repo access
ssh-keygen -t ed25519 -C "klar-policy-app-ec2-deploy-key"

# main commands to be run on the server
~~~{"id":"51728","variant":"standard"}
#!/bin/bash
set -e

echo "=== Updating system ==="
sudo apt update -y
#sudo apt upgrade -y  # don't run!

echo "=== Installing basic tools ==="
sudo apt install -y git curl gnupg lsb-release ca-certificates

echo "=== Setting up Docker repository ==="
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

echo "=== Installing Docker & Docker Compose ==="
sudo apt update -y
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

echo "=== Adding ubuntu user to docker group ==="
sudo usermod -aG docker $USER

echo "=== Enabling Docker service on boot ==="
sudo systemctl enable docker
sudo systemctl start docker

echo "=== Cloning private GitHub repo ==="
# Replace this line with your actual repo URL
REPO_URL="git@github.com:Rishav273/rag_app.git"
TARGET_DIR="/home/ubuntu/tca"

if [ -d "$TARGET_DIR" ]; then
    echo "Directory already exists. Pulling latest changes..."
    cd $TARGET_DIR
    git pull
else
    echo "Cloning repository..."
    git clone $REPO_URL $TARGET_DIR
fi

echo "=== Creating required directories ==="
cd tca

echo "=== Running Docker Compose ==="
docker compose pull || true
docker compose -f docker-compose-prod.yml up -d --remove-orphans

echo "=== Deployment Complete ==="
echo "Visit your frontend at: http://YOUR_EC2_PUBLIC_IP:8501"
echo "SSH access & private services remain secure."
~~~
