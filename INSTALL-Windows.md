# Software prerequisties
Windows 10 or 11 (If using a GPU, NVIDIA drivers installed)
Other versions may work but are untested

# Hardware Requirements:
If using an NVIDIA GPU, you can test that your environment is working properly by running:
```
nvidia-smi
```
If you get an error, stop here and troubleshoot how to get Nvidia drivers

# Setup
Open a PowerShell prompt. Powershell is important because it handles forward-slashes in path names, which Disco Diffusion uses in certain places.
You will want to use a Powershell prompt for progrockdiffusion.

## Install git
Install git from here: https://git-scm.com/download/win

## Install Python
Download and install Python 3.7: https://www.python.org/ftp/python/3.7.9/python-3.7.9-amd64.exe

## Install pip:
```
python3 -m ensurepip --upgrade
```

## Clone the prog rock diffusion repo
First, in your terminal, go the directory above where you want progrockdiffusion to live. 
If you don't know how to change directories, you may want to stop here and find a tutorial on the basics of using a terminal.

Once you're in the right directory:
```
git clone --recurse-submodules https://github.com/lowfuel/progrockdiffusion.git
cd progrockdiffusion
```
**Note: the "cd" command above is important, as the next steps will add additional libraries and data inside ProgRockDiffusion**

***From here on out, this is the directory you'll want to be in when you use the program.***

## Install the required libraries and tools
```
pip install -r requirements.base.txt
```

### EITHER Install GPU accelerated PyTorch
```
pip install -r requirements.gpu.txt
```

### OR install the basic CPU version of PyTorch (warning - very slow!)
```
pip install -r requirements.cpu.txt
```

## Test
To ensure everything is working, you can run progrockdiffusion with its default settings:
```
python3 prd.py
```
When it's done, check the images_out folder to see if the image was created!

**You can now return to the [README](README.md) file to learn how to use Progrockdiffusion**
