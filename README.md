# Developer Setup Guide: BioMEdCLIP Training & crewAI Workflow

This project involves training a biomedical vision-language model (BioMEdCLIP) and orchestrating a Laboratory Information System (LIS) workflow using the CrewAI multi-agent framework. BioMEdCLIP (also called BiomedCLIP) is a multimodal biomedical foundation model pretrained on millions of scientific image-caption pairs. CrewAI is an open-source Python framework for multi-agent orchestration that automates collaborative AI workflows. The steps below cover setting up your local workstation (Windows) and a remote GPU-enabled VM (Linux/Ubuntu), including software installation, configuration of Google Drive via rclone, Jupyter Notebook setup, and required Python dependencies.

## Prerequisites

- **Windows machine** with OpenSSH (or PowerShell/WSL) for SSH access.  
- **Linux (Ubuntu) VM** with an NVIDIA GPU. You need the VM’s IP address, username, and a private SSH key.  
- **Google account** (if you will use Google Drive for data).  

Ensure OpenSSH is installed on Windows (you may use PowerShell or WSL). You should already have a `.pem` SSH key file (for example, downloaded from your cloud provider) and VM credentials. All sensitive values (key paths, usernames, IP addresses) will be replaced with placeholders below; do **not** commit real secrets or keys to Git.

## Windows (Local) Setup

1. **Set SSH key permissions.** On Windows, use `icacls` to restrict the private key file (equivalent to `chmod 400`). For example, in PowerShell or CMD run:
   ```powershell
   icacls "C:\path\to\your_key.pem" /reset
   icacls "C:\path\to\your_key.pem" /grant:r "%USERNAME%:(R)"
   icacls "C:\path\to\your_key.pem" /inheritance:r
   ```

2. **SSH into the VM.**
   ```powershell
   ssh -i "C:\path\to\your_key.pem" <VM_USERNAME>@<VM_IP_ADDRESS>
   ```

## Remote VM (Ubuntu) Setup

1. **Update and upgrade packages.**
   ```bash
   sudo apt update
   sudo apt upgrade -y
   ```

2. **Install Python and build tools.**
   ```bash
   sudo apt install -y python3-pip python3-dev build-essential
   ```

3. **Check for NVIDIA GPU.**
   ```bash
   lspci | grep -i nvidia
   ```

4. **Install NVIDIA drivers.**
   ```bash
   sudo apt install -y nvidia-driver-535
   sudo reboot
   ```

5. **Verify GPU driver.**
   ```bash
   nvidia-smi
   ```

6. **Create a Python virtual environment.**
   ```bash
   pip3 install virtualenv
   virtualenv venv
   source venv/bin/activate
   ```

7. **Install core Python packages.**
   ```bash
   pip install jupyter numpy pandas matplotlib scikit-learn
   ```

8. **Install PyTorch with CUDA.**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

9. **Optional: Additional libraries.**
   ```bash
   pip install open_clip_torch transformers opencv-python seaborn albumentations openpyxl
   ```

## Google Drive Integration (rclone)

1. **Install rclone.**
   ```bash
   curl https://rclone.org/install.sh | sudo bash
   ```

2. **Configure Google Drive remote.**
   ```bash
   rclone config
   ```

3. **Create a mount point.**
   ```bash
   mkdir -p ~/gdrive
   ```

4. **Mount the Drive.**
   ```bash
   rclone mount gdrive: ~/gdrive --daemon
   ```

5. **Automate mounting on reboot.**
   ```bash
   (crontab -l 2>/dev/null; echo "@reboot rclone mount gdrive: $HOME/gdrive --daemon") | crontab -
   ```

## Jupyter Notebook Access

1. **Start Jupyter on the VM.**
   ```bash
   jupyter notebook --no-browser --ip=0.0.0.0 --port=8888
   ```

2. **Set up SSH port forwarding.**
   ```powershell
   ssh -i "C:\path\to\your_key.pem" -L 8888:localhost:8888 <VM_USERNAME>@<VM_IP_ADDRESS>
   ```

3. **Open Jupyter in browser.**
   Navigate to [http://localhost:8888](http://localhost:8888).

## Environment Variables and `.env`

Create a `.env` file in your project (and add it to `.gitignore`) to store configuration such as model settings and API endpoints. For example:
```
PROVIDER=ollama
MODEL1=llama3.1:8b
MODEL2=all-minilm:latest
API_BASE=http://localhost:11434
SSH_KEY_PATH=C:/path/to/your_key.pem
SSH_USER=your_vm_username
SSH_HOST=your_vm_ip
```

## Python Dependencies (CrewAI)

`requirements.txt`:
```
crewai==0.159.0
openai==1.83.0
litellm==1.74.9
chromadb==1.0.20
onnxruntime==1.22.0
pydantic==2.9.2
tiktoken==0.8.0
```

Install with:
```bash
pip install -r requirements.txt
```

## Notes

- Never commit your `.pem` key or `.env` file to version control.
- Use `.gitignore` to exclude sensitive files.
- Access Google Drive under `~/gdrive` on the VM.

## The biomedical_graph.png shows how the BioMedCLIP model has been finetuned and used for the purpose of my classification.

## Also BioMedCLIP_Visual_Encoder.png shows the architecture of BioMedCLIP model. From this we can come to understand that what is the shape of input required by the model.
