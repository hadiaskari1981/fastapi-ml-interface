# ---------- Base image ------------------
FROM python:3.12

# ----------- Environment settings -------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ----- Set working directory -------------

WORKDIR /app

# ----- Install system dependencies -------
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ------------Install python dependencies --
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy application code ----

COPY src/main.py /app
# COPY model /app/model -> # if you are mounting /var/tmp/model as a volume using cp -R model /var/tmp/

# --- Expose port -----------------------

EXPOSE 8000

# ------- Run FastAPI ---------------------

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
