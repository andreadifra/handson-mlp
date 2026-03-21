# Hands-On Machine Learning in Docker

If you would like to run and tweak the book's notebooks in a Docker container, you're in the right place!

## Prerequisites

You must first [Install Docker](https://docs.docker.com/engine/installation/).

If you have an Nvidia GPU and you're on Linux (or [WSL on Windows](https://learn.microsoft.com/en-us/windows/wsl/install)), then you can use your GPU inside the Docker container. For this, you must ensure that you have the [latest drivers for your GPU](https://www.nvidia.com/drivers/), and install the [Nvidia Container Toolkit[(https://github.com/NVIDIA/nvidia-container-toolkit) so Docker can access the GPU.

Below, I will assume that your Docker setup is functional and supports your GPU (if any).

## Usage

You can either download one of the images that I built and uploaded to docker.io, or you can build your own (that's slower but you get a fully up-to-date Docker image).

### Download an existing image

Choose the appropriate image, depending on your platform.

```bash
# CPU only
docker pull ageron/handson-mlp:cpu

# Nvidia GPU with CUDA 12.6 compute platform
docker pull ageron/handson-mlp:cu126
```

### Build your own image

Assuming you have already downloaded the project locally in your home folder:

```bash
cd  # go to your home folder
cd handson-mlp
docker build -t ageron/handson-mlp:cpu -f docker/Dockerfile --build-arg PT_VARIANT=cpu .
```

If you want to build the Nvidia CUDA 12.6 variant, replace `cpu` with `cu126`. In the commands below, I'll use `cpu`, but you can replace `cpu` with the appropriate variant for your hardware.

If you only want to validate that the container definition still builds without waiting for the full PyTorch stack, use the smoke target:

```bash
docker build --target devcontainer-smoke -f docker/Dockerfile .
```

The smoke target installs the base environment but skips the PyTorch-dependent packages, so it is much faster than a full CPU or CUDA build.

> NOTE: You can see the list of available variants by visiting [PyTorch's download page](https://pytorch.org/get-started/locally/) and selecting Linux (even if you are on Windows or MacOS).

This will take quite a while and download gigabytes of data, but it is only required once.

Subsequent builds should be faster because the Dockerfile now lets BuildKit reuse uv's download cache across builds.

After the process is finished you have an `ageron/handson-mlp:cpu` image (or `:cu126`): it will be the base for your experiments. You can confirm that by running the following command:

```bash
docker images
```

This should output something like this:

```text
REPOSITORY            TAG         IMAGE ID            CREATED             SIZE
ageron/handson-mlp    cpu         3ebafebc604a        2 minutes ago       2.09GB
```

### Run the notebooks

Run the following commands to start the Jupyter server inside the container (which is named `homlp`):

```bash
cd  # go to the folder containing the handson-mlp directory
cd handson-mlp
docker run -it --rm --name homlp -p 8888:8888 -v "$PWD":/content ageron/handson-mlp:cpu
```

If you are using an Nvidia GPU, you must add `--gpus all` to the list of options.

Next, just point your browser to the URL printed on the screen (e.g., `http://127.0.0.1:8888/lab?token=ecaf97203325417dfe17e0824cb80e34cafcaa52e74485b0`), and you're ready to play with the book's code!

The project directory (in your home folder) is mapped to the `/content` directory inside the docker image. This means that the changes you make to the notebooks through the browser are saved in your project directory.

You can close the server by pressing `Ctrl-C` in the terminal window.

This will remove the container so you can start a new one later (but it will not remove the image or the notebooks, don't worry!).

## Use this repo as a VS Code devcontainer

If you prefer to work directly in VS Code instead of starting Jupyter Lab manually, this repository now includes two devcontainer configurations:

- `.devcontainer/cpu/devcontainer.json`: builds the local CPU image with `PT_VARIANT=cpu`
- `.devcontainer/gpu/devcontainer.json`: builds the local CUDA 12.6 image with `PT_VARIANT=cu126` and starts the container with `--gpus=all`

Both configurations:

- build directly from this repository's `docker/Dockerfile`
- build against `linux/amd64`, which is the correct target for standard Windows and WSL 2 Nvidia setups
- pin Python 3.12 and install `git` so VS Code source control and notebook kernels work in the container
- register a `Python (handson-mlp)` kernel for notebooks
- install a small set of useful VS Code extensions in the container, including Python, Jupyter, GitHub Copilot, and GitHub Copilot Chat
- keep the image's default Jupyter Lab command disabled, since VS Code connects directly to the container and runs notebooks itself

This is slower than reusing a prebuilt image the first time, but it ensures that Docker produces a native `linux/amd64` container for Windows and WSL 2 hosts instead of accidentally reusing an incompatible architecture.

### Open the devcontainer in VS Code

1. Open this repository in VS Code.
2. Make sure the local `Dev Containers` extension is installed.
3. Run `Dev Containers: Reopen in Container` from the Command Palette.
4. Select either `handson-mlp (CPU)` or `handson-mlp (GPU CUDA 12.6)`.
5. Wait for the initial container build to finish, then open any notebook and select the `Python (handson-mlp)` kernel if VS Code prompts you.

Use the CPU devcontainer on machines without an Nvidia GPU. Use the GPU devcontainer only on systems where `docker run --gpus all ...` already works, such as Docker Desktop with WSL 2 GPU support on Windows.

### Clean up stale Docker artifacts

Dev Containers creates BuildKit cache entries and `vsc-...` helper images as part of its normal build flow. If you rebuild often, these can accumulate and consume a lot of disk space even when they point at the same layers.

To remove reclaimable build cache:

```bash
docker buildx prune -f
```

To remove stopped containers:

```bash
docker container prune -f
```

To remove dangling images:

```bash
docker image prune -f
```

If you want a more aggressive cleanup of anything not currently used by a running container:

```bash
docker system prune -af
```

Run the aggressive cleanup only when you are okay with Docker rebuilding cached layers again later.

Have fun!

The manual Docker workflow above still works if you prefer to launch Jupyter Lab in the browser instead of using VS Code's notebook UI.
