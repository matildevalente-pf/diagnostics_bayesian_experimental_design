# Runs the package at the versions the thesis results were computed with, without
# installing its system dependencies on the host. From the repository root:
#
#   docker build -t diagnostics-bayesian-experimental-design .
#   docker run --rm -it -v "$(pwd)":/workspace diagnostics-bayesian-experimental-design \
#       python relative/path/to/script.py
FROM python:3.12-slim

# libnetcdf is required by netcdf4, a dependency of desc-opt.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# In a layer of their own, so that a change to the code does not reinstall them.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Editable, so that the data, model and results paths resolve under /workspace: mounted
# over this checkout, downloads and results land on the host instead of in the image.
COPY . .
RUN pip install --no-cache-dir --no-deps -e .

CMD ["python"]
