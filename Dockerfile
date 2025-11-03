# Use official lightweight Python 3.12 image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for building some Python packages)
RUN apt-get update && apt-get install -y \
    curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# Copy Poetry files first for dependency caching
COPY pyproject.toml poetry.lock* /app/

# Configure Poetry to install to system (no virtualenv)
RUN poetry config virtualenvs.create false

# Install only main dependencies (exclude dev)
RUN poetry install --no-root --only main --no-interaction --no-ansi

# Copy your full project
COPY . /app

# Expose Streamlit port
EXPOSE 8501

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    PYTHONUNBUFFERED=1

# Run Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
