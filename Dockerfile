FROM python:3.10.12

RUN pip install --no-cache-dir \
  torch torchvision pytorch-lightning \
  torchmetrics torchinfo opencv-python pyautogui