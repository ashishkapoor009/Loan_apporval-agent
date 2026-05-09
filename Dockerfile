FROM python:3.11-slim
WORKDIR /app

# Install uv
RUN pip install uv

# Copy Python dependencies
COPY pyproject.toml .
# Setup uv environment
RUN uv venv && uv sync

# Copy backend source
COPY loan_agent.py .
COPY api.py .

# Copy frontend
COPY frontend/dist ./frontend/dist

# Expose port (Cloud Run sets PORT env var)
ENV PORT=8080
EXPOSE 8080

# Command to run the application using uvicorn
CMD ["uv", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
