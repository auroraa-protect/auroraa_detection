@echo off
echo ============================================================
echo  Auroraa Forensics -- GPU Environment Setup (RTX 2050)
echo ============================================================

set PIP=venv\Scripts\python.exe -m pip

echo.
echo [1/3] Installing core + ML dependencies...
%PIP% install -r requirements.txt

echo.
echo [2/3] Installing PyTorch with CUDA 12.1 support (RTX 2050)...
%PIP% install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo [3/3] Verifying CUDA availability...
venv\Scripts\python.exe -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo.
echo Done! Run notebooks with:
echo   venv\Scripts\jupyter notebook notebooks\training_pipeline.ipynb
pause
