python -m venv .env
.env\Scripts\activate.bat 
pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install gradio opencv-python scipy ftfy regex einops