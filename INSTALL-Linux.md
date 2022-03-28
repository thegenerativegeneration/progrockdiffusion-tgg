# Software prerequisties

Ubuntu 20.04 or similar (A docker environment, VM, or Windows Subsystem for Linux should work provided it can access your GPU).

If using a GPU: CUDA 11.4+ (installation instructions can be found here: https://developer.nvidia.com/cuda-11-4-1-download-archive).

# Setup
## Update OS and Install git (nano is optional, for those who need a basic text editor)
```
sudo apt update
sudo apt upgrade -y
sudo apt install git nano
```

## Install Python
```
apt-get -y install python3
```

Install pip:
```
python3 -m ensurepip --upgrade
```

## Clone the prog rock diffusion repo
First, in your shell, go the directory above where you want progrockdiffusion to live. 
If you don't know how to change directories, you may want to stop here and find a tutorial on the basics of using a shell.

Once you're in the right directory:
```
git clone https://github.com/lowfuel/progrockdiffusion.git
cd progrockdiffusion
```
**Note: the "cd" command above is important, as the next steps will add additional libraries and data inside ProgRockDiffusion**

***From here on out, this is the directory you'll want to be in when you use the program.***

## Install the required libraries and tools
```
pip install -r requirements.base.txt
```

## Basic or GPU Accelerated PyTorch
You defnitely should install the GPU version if you have an NVIDIA card. It's almost 30x faster.
Otherwise, you can install the CPU version instead (required for MacOS)

### EITHER Install GPU accelerated PyTorch
```
pip install -r requirements.gpu.txt
```

### OR install the basic CPU version of PyTorch (warning - very slow!)
```
pip install -r requirements.cpu.txt
```

## Depending on your Linux platform, you may get an error about libGL.so.1
If you do, try installing these dependencies:
```
sudo apt-get install ffmpeg libsm6 libxext6 -y
```

## Finally:
```
sudo apt install imagemagick
```

## Test
To ensure everything is working, you can run progrockdiffusion with its default settings:
```
python3 prd.py
```
When it's done, check the images_out folder to see if the image was created!

**You can now return to the README file to learn how to use Progrockdiffusion**
