FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for GDAL/rasterio
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn python-multipart

# Copy application code
COPY src/ src/
COPY scripts/ scripts/
COPY static/ static/
COPY config/ config/

# Create checkpoint directories
RUN mkdir -p checkpoints/segmentation checkpoints/damage data/outputs

# Copy checkpoints if they exist (optional — can be mounted as volumes)
COPY checkpoints/ checkpoints/ 2>/dev/null || true

EXPOSE 8000

# Start the API server
CMD ["python3", "scripts/api.py"]
