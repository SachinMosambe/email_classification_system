FROM python:3.12-slim

# Upgrade system packages to mitigate vulnerabilities
RUN apt-get update && apt-get upgrade -y && apt-get clean

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]