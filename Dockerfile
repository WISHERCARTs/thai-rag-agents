# The live demo: the agents behind POST /ask, with the search index built into the image.
# The index is built in CI (python -m thai_agents build-index) and copied in; see .github/workflows/deploy.yml.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY thai_agents ./thai_agents
COPY index ./index
RUN useradd --create-home app && chown -R app /app
USER app

ENV THAIRAG_INDEX=/app/index PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn thai_agents.server:main_app --factory --host 0.0.0.0 --port ${PORT}"]
