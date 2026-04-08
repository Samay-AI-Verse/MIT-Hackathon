FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY requirements.txt /workspace/pharmasim/requirements.txt
RUN pip install --no-cache-dir -r /workspace/pharmasim/requirements.txt

COPY . /workspace/pharmasim

EXPOSE 7860

CMD ["uvicorn", "pharmasim.server:app", "--host", "0.0.0.0", "--port", "7860"]
