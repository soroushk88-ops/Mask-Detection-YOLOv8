# 1. Start from a small, official Python base image.
FROM python:3.11-slim

# 2. Install the system libraries that OpenCV (used by ultralytics) needs.
#    Without libgl1 you hit the classic "ImportError: libGL.so.1" at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Everything from here on happens inside /app in the container.
WORKDIR /app

# 4. Install the CPU-only build of PyTorch FIRST.
#    Our service only does CPU inference, so we don't want the huge (multi-GB)
#    CUDA/GPU build. This single choice keeps the image several gigabytes smaller.
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# 5. Copy ONLY requirements first, then install them.
#    Because this layer changes only when requirements.txt changes, Docker can
#    cache it. Later, when you edit your code, the install step is reused instead
#    of running again — builds become much faster.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Now copy the rest of the project into the image
#    (your app/ code and the models/best.pt weights).
COPY . .

# 7. Tell readers (and tools) that the service listens on port 8000.
EXPOSE 8000

# 8. The command that runs when the container starts.
#    host 0.0.0.0 makes the service reachable from OUTSIDE the container
#    (127.0.0.1 would only be reachable from inside it).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
