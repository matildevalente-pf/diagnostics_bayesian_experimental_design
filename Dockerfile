# Runs the package without installing its system dependencies on the host.
#
#   docker build -t plasma-state-estimation-for-diagnostic-placement-optimization .
#   docker run --rm -v $(pwd):/workspace \
#       plasma-state-estimation-for-diagnostic-placement-optimization \
#       python relative/path/to/your_script.py
FROM python:3.12-slim

# libnetcdf is required by netcdf4, a dependency of desc-opt.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . /app
RUN pip install --no-cache-dir /app

WORKDIR /workspace
CMD ["python"]
