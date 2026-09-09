# Environment setup for SLML 2026

![SLML logo](logo.png)

This guide walks you through creating a Python environment that will work for **all programming exercises in Weeks 2–13**.  
Follow every step in order; each step depends on the previous one.

---

## Prerequisites

Before you start, make sure you have:

- **Conda (Miniconda or Anaconda)** — the package and environment manager we use.  
  Download Miniconda (lightweight, recommended) from:  
  <https://docs.conda.io/en/latest/miniconda.html>  
  Choose the **Python 3.11** installer for your operating system.  
  *(If you already have Anaconda or Miniconda installed, skip this step.)*

- About **5–8 GB of free disk space** — for Python, all packages, and the MNIST dataset (Weeks 10–11).

- **Internet access** — required during setup and for the MNIST download step.

---

## Step 1 — Open a terminal

**Windows:**  
Open the **Anaconda Prompt** (search "Anaconda Prompt" in the Start menu).  
You should see a prompt like: `(base) C:\>`

**macOS and Linux:**  
Open a regular **Terminal**.  
You should see a prompt like: `(base) user@pc:~$`

The `(base)` prefix means conda is active.  
If you see no prefix, run `conda init` and then restart the terminal.

---

## Step 2 — Create the `slml` environment

The `environment.yml` file (in this folder) lists all required packages with pinned versions. Run:

```bash
conda env create -f environment.yml
```

This downloads and installs all packages — it can take **5–15 minutes** depending on your internet speed.  
If the solver seems stuck for more than 20 minutes, see the [Troubleshooting](#troubleshooting) section.

> **Note for macOS Apple Silicon (M1/M2/M3/M4):** After the conda step, run:
> ```bash
> conda activate slml
> pip install "torch>=2.0,<2.12" "torchvision>=0.15,<0.27"
> ```
> This installs the Metal (MPS) GPU build of PyTorch. The pip entries in `environment.yml` install the default build, so you need to re-run this to get the Apple Silicon-optimised version.

> **Note for Windows or Linux with an NVIDIA GPU:** After activating the environment, run:
> ```bash
> pip install "torch>=2.0,<2.12" "torchvision>=0.15,<0.27" --index-url https://download.pytorch.org/whl/cu121
> ```
> Replace `cu121` with the CUDA version that matches your GPU driver.  
> **GPU is not required for this course** — CPU is sufficient for all exercises.

> **Note for Linux or Windows, CPU only:**  
> To download a smaller PyTorch build (~250 MB instead of ~2 GB), run:
> ```bash
> pip install "torch>=2.0,<2.12" "torchvision>=0.15,<0.27" --index-url https://download.pytorch.org/whl/cpu
> ```

---

## Step 3 — Activate the environment

```bash
conda activate slml
```

Your prompt should now start with `(slml)`.  
You must activate the `slml` environment **every time** you open a new terminal before working on exercises.

---

## Step 4 — Register the Jupyter kernel

This step makes the `slml` environment visible as a kernel inside JupyterLab:

```bash
python -m ipykernel install --user --name slml --display-name "Python 3 (slml)"
```

You only need to do this once.

---

## Step 5 — Open JupyterLab

Make sure you are in the folder that contains the exercises, then run:

```bash
jupyter lab
```

Your browser will open JupyterLab automatically.  
If it does not, copy the `http://localhost:8888/...` URL from the terminal output.

> **Important:** When you open a notebook, check the kernel indicator in the top-right corner of the notebook.  
> It must show **"Python 3 (slml)"**.  
> If it shows a different kernel, click on it and select **"Python 3 (slml)"** from the list.

---

## Step 6 — Run the smoke test

Open `test.ipynb` and run **Kernel → Restart Kernel and Run All Cells**.

Each section prints **OK** when the check passes.  
The final cell prints a summary table.

If any check fails, see the [Troubleshooting](#troubleshooting) section below.

---

## Environment file (for reference)

The `environment.yml` in this folder contains:

```yaml
name: slml

channels:
  - conda-forge
  - nodefaults

dependencies:
  - python=3.11
  - numpy>=1.24,<2.0
  - pandas>=2.0
  - scipy>=1.10
  - scikit-learn>=1.2
  - matplotlib>=3.7,<3.10
  - seaborn>=0.12
  - pillow>=9.0
  - jupyterlab>=4.0
  - ipywidgets>=8.0
  - ipykernel>=6.0
  - ffmpeg>=6.0
  - nodejs
  - pip
  - pip:
    - torch>=2.0,<2.12
    - torchvision>=0.15,<0.27
```

---

## Alternative: pip / venv (advanced users)

If you prefer not to use conda, you can use a standard Python 3.11 virtual environment.  
You are responsible for installing ffmpeg separately (required for Weeks 4–5 animations).

```bash
# Create and activate a virtual environment
python3.11 -m venv slml-venv
source slml-venv/bin/activate      # macOS / Linux
# or: slml-venv\Scripts\activate   # Windows

# Install packages
pip install "numpy>=1.24,<2.0" "pandas>=2.0" "matplotlib>=3.7,<3.10" \
            "seaborn>=0.12" "scipy>=1.10" "scikit-learn>=1.2" \
            "pillow>=9.0" "jupyterlab>=4.0" "ipywidgets>=8.0" \
            "ipykernel>=6.0"

# Install PyTorch (CPU-only — sufficient for this course)
pip install "torch>=2.0,<2.12" "torchvision>=0.15,<0.27" --index-url https://download.pytorch.org/whl/cpu

# Register the kernel
python -m ipykernel install --user --name slml-venv --display-name "Python 3 (slml-venv)"
```

**ffmpeg** must be installed separately:
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt install ffmpeg`
- Windows: download from <https://ffmpeg.org/download.html> and add the `bin/` folder to your `PATH`

---

## Troubleshooting

### Wrong kernel selected

**Symptom:** `ModuleNotFoundError: No module named 'torch'` (or another package) when running a notebook.  
**Fix:** Check the kernel shown in the top-right of the notebook. If it is not `Python 3 (slml)`, click on it and switch. Re-run all cells from scratch (Kernel → Restart & Run All).

---

### Pytorch is not found when running Cell 7 and 8 with an Anaconda/Miniconda installation
Pytorch might not be found even though torch is installed and the right kernel is selected. 

On Windows machines in the Anaconda prompt:

```bash
conda activate slml
python -m pip uninstall -y torch torchvision torchaudio
conda install -c pytorch -c conda-forge pytorch torchvision cpuonly
python -m ipykernel install --user --name slml --display-name "Python 3 (slml)"
```

Close the opened Jupyter Lab tab and run Steps 5 and 6 again.

### Conda solver takes too long or fails

**Symptom:** `conda env create` runs for more than 20 minutes or exits with a `PackagesNotFoundError`.  
**Fix:**

1. Update conda itself first:
   ```bash
   conda update -n base conda
   ```
2. Try using `mamba` (a faster solver):
   ```bash
   conda install -n base mamba -c conda-forge
   mamba env create -f environment.yml
   ```
3. If the error mentions a channel conflict, delete any channel priority setting in `~/.condarc` and try again.

---

### Apple Silicon (M1/M2/M3/M4) — PyTorch MPS not available

**Symptom:** `torch.backends.mps.is_available()` returns `False` on an M-series Mac.  
**Fix:** The conda-forge and pytorch channels do not ship the Metal build. Re-install torch via pip:
```bash
conda activate slml
pip install --force-reinstall "torch>=2.0,<2.12" "torchvision>=0.15,<0.27"
```
Then re-run the smoke test. GPU is not required — all exercises run fine on CPU.

---

### MNIST download fails (Weeks 10–11 pre-fetch)

**Symptom:** The MNIST download cell in `test.ipynb` fails with a network error.  
**Causes and fixes:**

- **No internet access:** Set `SKIP_DOWNLOAD = True` at the top of the MNIST cell for now. Run it again when you have internet access before Week 10.
- **Corporate/university proxy:** Set the `HTTPS_PROXY` environment variable before launching JupyterLab:
  ```bash
  export HTTPS_PROXY=http://proxy.youruni.dk:8080  # macOS/Linux
  set HTTPS_PROXY=http://proxy.youruni.dk:8080      # Windows
  ```
- **Firewall blocks PyTorch mirrors:** Try downloading the four MNIST `.gz` files manually from <https://ossci-datasets.s3.amazonaws.com/mnist/> (the mirror torchvision itself uses; the four files are `train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`, `t10k-images-idx3-ubyte.gz`, `t10k-labels-idx1-ubyte.gz`) and placing them in `Exercises/data/MNIST/raw/`.

---

### seaborn / matplotlib deprecation warnings

The course notebooks have been updated to use `sns.set_theme()` and `plt.get_cmap()`.  
If you still see warnings about `sns.set()` or `plt.cm.get_cmap()`, make sure you are using the current exercise files. The `environment.yml` keeps `matplotlib<3.10` until newer matplotlib versions are separately validated for all weeks.

---

### "conda activate" not recognised (Windows)

Run `conda init powershell` (or `conda init cmd.exe`), then close and reopen the terminal.
