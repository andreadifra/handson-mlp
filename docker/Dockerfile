FROM debian:trixie-slim

WORKDIR /content

COPY ./pyproject.toml ./
COPY ./uv.lock ./
COPY ./apt.txt ./

ENV VIRTUAL_ENV=/opt/venvs/handson-mlp
ENV UV_PATH=/root/.local/bin
ENV PATH="$UV_PATH:$VIRTUAL_ENV/bin:$PATH"
ENV PYTHON_VERSION=3.12

# Choose the PyTorch variants, such as cpu or cu126
# See the list at https://pytorch.org/get-started/locally/ (select Linux)
ARG PT_VARIANT=cpu
ENV PT_VARIANT=${PT_VARIANT}

RUN apt-get update && \
    tr -d '\r' < apt.txt | xargs apt-get install -y --no-install-recommends && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    uv python install $PYTHON_VERSION && \
    uv venv --python $PYTHON_VERSION $VIRTUAL_ENV && \
    uv sync --active --frozen && \
    uv add --active swig && \
    uv add --active box2d-py && \
    uv remove --active swig

RUN \
    echo "PyTorch Variant = ${PT_VARIANT}" && \
    uv pip uninstall torch torchvision torchaudio && \
    UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 uv pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/${PT_VARIANT} && \
    apt-get purge -y build-essential cmake && \
    apt-get autoremove -y && \
    uv cache clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo 'Success'

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

