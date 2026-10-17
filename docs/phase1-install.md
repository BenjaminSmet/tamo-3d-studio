# Phase 1 install: TRELLIS.2 with Espalier-2 on Windows

TRELLIS.2 and the Espalier-2 patch only support Linux. On Windows they run inside
**WSL2**, a real Ubuntu that Windows runs next to your desktop. It uses your RTX 5070 Ti
directly through the normal Windows NVIDIA driver, so there is no dual boot and no
second driver.

Expect about 20 GB of downloads and builds, and around an hour the first time,
mostly spent compiling the 3D extensions.

## Why each piece is needed

| Piece | What it does |
| --- | --- |
| NVIDIA driver (Windows) | Lets anything use the GPU, including Ubuntu inside WSL2 |
| CUDA Toolkit 12.8 (inside Ubuntu) | The compiler (`nvcc`) that builds TRELLIS.2's 3D extensions for your card. The driver does not include it |
| uv | Installs Python 3.10 and the exact package versions Espalier-2 needs, in their own folder |
| Hugging Face account | TRELLIS.2 downloads a gated image encoder (DINOv3) on first run |

Your RTX 50-series card needs PyTorch 2.8 with CUDA 12.8 or newer. Espalier-2 pins
exactly those, so don't install PyTorch yourself.

## 1. Update the Windows driver

Install the latest NVIDIA Game Ready or Studio driver. Then in PowerShell:

```powershell
nvidia-smi
```

The top right must say `CUDA Version: 12.8` or higher. If it is lower, the driver is too old.

## 2. Install WSL2 with Ubuntu

In PowerShell as administrator:

```powershell
wsl --install -d Ubuntu-24.04
```

Restart when asked, then open **Ubuntu** from the Start menu and choose a username and password.

Give Ubuntu more memory than the default half of your RAM, because the export step
holds the texture in system memory. Create `C:\Users\<you>\.wslconfig` with:

```ini
[wsl2]
memory=24GB
swap=16GB
```

Then run `wsl --shutdown` in PowerShell and reopen Ubuntu.

## 3. Check the GPU from Ubuntu

From here on, every command runs in the Ubuntu window.

```bash
nvidia-smi
```

You should see the RTX 5070 Ti. **Never install an NVIDIA driver inside Ubuntu**: WSL2
uses the Windows one, and a Linux driver breaks it.

## 4. Install build tools and the CUDA Toolkit

```bash
sudo apt-get update
sudo apt-get install -y build-essential git curl wget

wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8

echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version
```

`nvcc --version` must report release 12.8. The `wsl-ubuntu` repository matters: it is
the toolkit without a driver.

## 5. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

## 6. Get access to the gated model

1. Create a free account at huggingface.co.
2. Open [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
   and request access. Approval can take a while.
3. Create a read token under Settings → Access Tokens, then log in:

   ```bash
   uvx --from huggingface_hub hf auth login
   ```

You do **not** need RMBG-2.0, which Espalier-2 uses for background removal by default.
Its license is non-commercial, so Tamo removes backgrounds with BiRefNet (MIT)
instead and runs Espalier-2 with `--no-rembg`.

## 7. Install Espalier-2

Keep it in your Ubuntu home folder, not under `/mnt/c`. Building and loading from the
Windows drive is much slower.

```bash
cd ~
git clone https://github.com/overyonder/espalier
cd espalier
./install.sh
```

It clones TRELLIS.2 and its extensions at pinned versions, applies the 16 GB patches,
and builds everything into `~/espalier/.venv`. It ends with
`Espalier-2 is installed.` If it stops earlier, copy the last 50 lines of the output into
the project thread.

Then add the one package the Tamo background step needs:

```bash
uv pip install --python ~/espalier/.venv einops
```

## 8. Get the Tamo repo into Ubuntu

```bash
cd ~
git clone https://github.com/BenjaminSmet/tamo-3d-studio
cd tamo-3d-studio
```

Your Windows files are visible from Ubuntu under `/mnt/c/...`, for example
`C:\Users\Ben\Pictures\anna.png` is `/mnt/c/Users/Ben/Pictures/anna.png`.
To open Ubuntu files from Windows Explorer, type `\\wsl$\Ubuntu-24.04\home\<you>` in the address bar.

## 9. First run

Close Unreal first, since it competes for the same video memory.

```bash
~/espalier/.venv/bin/python ai-engine/phase1/run_phase1.py \
    --image /mnt/c/Users/<you>/Pictures/<actor>-front.png --actor <actor>
```

The first run downloads the TRELLIS.2 and BiRefNet weights (several GB). Generation then
takes about 5 minutes at full resolution. This first run is geometry only. When it
succeeds, run it again with `--texture`.

## If something fails

| Symptom | Likely cause |
| --- | --- |
| `nvidia-smi` not found in Ubuntu | Windows driver too old, or WSL1 instead of WSL2 (`wsl -l -v` in PowerShell) |
| `nvcc` not found | Step 4 `PATH` lines not loaded; run `source ~/.bashrc` |
| `401` or `gated repo` errors | DINOv3 access not approved yet, or not logged in (step 6) |
| Out of memory | Unreal or a browser holding video memory, or the WSL memory limit from step 2 not applied |
| Build killed with no message | Ubuntu ran out of RAM; raise `memory` in `.wslconfig` or set `MAX_JOBS=4` before `./install.sh` |
