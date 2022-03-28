# progrockdiffusion
A command line version of [Disco Diffusion](https://github.com/alembics/disco-diffusion) for still image generation.
*Animations are not currently supported.*

In addition to Disco Diffusion's great features, Prog Rock has the following benefits:

- Uses your own hardware
- GPU and CPU modes are supported (note: CPU mode is quite slow)
- Works offline (once the required files are downloaded). No more colab disconnects!
- Can be invoked from a script or batch file for mass unattended batches
- Multiple partial settings files are supported, allowing you to layer settings
- Some variables can be randomized to help you explore the possible outputs of a prompt
- If you put \_artist\_ in your prompt, it will be replaced with a randomly selected artist

# Hardware requirements
GPU: An Nvidia GPU is *highly* recommended! The speed improvement is massive. 8gb is probably the minimum amount of GPU memory. This author has an RTX 3080 with 10gb and it runs fairly well, but some advanced features are not possible with "only" 10gb.

As for AMD cards, I have not had a chance to test one. If you're interested in helping me figure out if it's possible, drop me a note in the issues page.

CPU: For CPU mode, you will need at least 16gb of RAM.

STORAGE: you'll need at least 40gb of free disk space, depending on which models you enable.

# Software prerequisties
**[Linux]**

Ubuntu 20.04 or similar (A docker environment, VM, or Windows Subsystem for Linux should work provided it can access your GPU).

If using a GPU: CUDA 11.4+ (installation instructions can be found here: https://developer.nvidia.com/cuda-11-4-1-download-archive).

**[Windows]**

Windows 10 or 11 (If using a GPU, NVIDIA drivers installed)
Other versions may work but are untested

**[MacOS]**

Minimal testing has been done with the latest MacOS on an M1 Macbook Air.
PLEASE NOTE: GPU acceleration is not yet supported. It will work, but it will be *slow.*

## If using an NVIDIA GPU, test NVIDIA drivers
You can test that your environment is working properly by running:

```
nvidia-smi
```

The output should indicate a driver version, CUDA version, and so on. If you get an error, stop here and troubleshoot how to get Nvidia drivers, CUDA, and/or a connection to your GPU with the environment you're using.

# First time setup

**[Linux]**
```
sudo apt update
sudo apt upgrade -y
sudo apt install git nano
```

**[Windows]**

Open a Powershell Prompt (*Powershell* is important as it handles path names with forward-slashes)
Install git from here: https://git-scm.com/download/win

**[MacOS]**

Open a terminal and run this command:
```
git --version
```
It will either tell you the version if git is installed, or prompt you to install Xcode which includes git.

## Install Python

**[Linux]**
```
apt-get -y install python3
```
**[Windows]**
Download and install Python 3.7: https://www.python.org/ftp/python/3.7.9/python-3.7.9-amd64.exe

**[MacOS]**
Download and install Python 3.8: https://www.python.org/ftp/python/3.8.10/python-3.8.10-macos11.pkg

**[All Platforms]**
Install pip:
```
python -m ensurepip --upgrade
```
(note on linux you might need to use python3 instead of python)

## Clone the prog rock diffusion repo
First, in your terminal or Powershell prompt, go the directory where you want progrockdiffusion to live. 
If you don't know how to change directories, you may want to stop here and find a tutorial on the basics of using a commandline interface on your platform.

Once you're in the right directory:
```
git clone https://github.com/lowfuel/progrockdiffusion.git
cd progrockdiffusion
```
**Note: the "cd" command above is important, as the next steps will add additional libraries and data to ProgRockDiffusion**

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

## Install remaining libraries and tools
**[MacOS]**
```
export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1
pip install grpcio
```

**[Linux]** Depending on your Linux platform, you may get an error about libGL.so.1
If you do, try installing these dependencies:
```
sudo apt-get install ffmpeg libsm6 libxext6 -y
```
**[Linux]** Finally:
```
sudo apt install imagemagick
```

# Use

NOTE: On your first run it might appear to hang. Let it go for a good while, though, as it might just be downloading models.
Somtimes there is no feedback during the download process (why? Who knows)

CD to the directory where you installed ProgRockDiffusion. Now you're ready!

The simplest way to run it is:

[Linux]
```
python3 prd.py
```
[Windows and MacOS]
```
python prd.py
```
This will generate an image using the settings from "settings.json", which you could edit to adjust the defaults (or, better yet, make a copy of it and tell prd to use an alternative settings file using the command line arguments below).

**Note: On windows you'll type "python" instead of "python3" in the commands below.**

```
usage: python3 prd.py [-h] [-s SETTINGS] [-o OUTPUT] [-p PROMPT]

Generate images from text prompts.
By default, the supplied settings.json file will be used.
You can edit that, and/or use the options below to fine tune:

Optional arguments:
  -h, --help            show this help message and exit
  -s SETTINGS, --settings SETTINGS
                        A settings JSON file to use, best to put in quotes
                        Can be specified more than once to layer settings on top of one another
  -o OUTPUT, --output OUTPUT
                        What output directory to use within images_out
  -p PROMPT, --prompt PROMPT
                        Override the prompt
  -i, --ignoreseed
                        Use a random seed instead of what is in your settings file

  -c, --cpu CORES
                        Force CPU mode, and (optionally) specify how many threads to run.

  --cuda DEVICE-ID
                        Specify which CUDA device ID to use for rendering (default: 0).

  -g PERCENT, --geninit PERCENT:
                        Will save an image called geninit.png at PERCENT of overall steps, for use with --useinit

  -u, --useinit:
                        Forces use of geninit.png as an init_image starting at 20% of defined steps.

Usage examples:

To use the Default output directory and settings from settings.json:
 python3 prd.py

To use your own settings.json file (note that putting it in quotes can help parse errors):
 python3 prd.py -s "some_directory/mysettings.json"

Note that multiple settings files are allowed. They're parsed in order. The values present are applied over any previous value:
 python3 prd.py -s "some_directory/mysettings.json" -s "highres.json"

To quickly just override the output directory name and the prompt:
 python3 prd.py -p "A cool image of the author of this program" -o Coolguy

Multiple prompts with weight values are supported:
 python3 prd.py -p "A cool image of the author of this program" -p "Pale Blue Sky:.5"

You can ignore the seed coming from a settings file by adding -i, resulting in a new random seed

To force use of the CPU for image generation, add a -c or --cpu (warning: VERY slow):
 python3 prd.py -c

To specify which CUDA device to use (advanced) by device ID (default is 0):
 python3 prd.py --cuda 1

```
Simply edit the settings.json file provided, or BETTER YET copy it and make several that include your favorite settings.
*Note that multiple settings files can be specified in your command*, and they'll be loaded in order.
Settings.json is **always loaded**, and any specified after that are layered on top (they only need to contain the settings you want to tweak).
For example you could have a settings file that just contains a higher width, height, and more steps, for when you want to make a high-quality image.
Layer that on top of your regular settings and it will apply those values without changing anything else.

# Tips and Troubleshooting
## Get a random artist
In your prompt, if you use \_artist\_ instead of an artists name, an artist will be picked at random from artists.txt

## If you get an error about pandas needing a different verison of numpy, you can try:
```
pip install --force-reinstall numpy
```
## If you are getting "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized"
This seems to be because of a Pytorch compiling bug for Intel CPUs.
You can set an environment variable that will fix this, either on your machine (if you know how to do that), or by editing prd.py.
To do it by editing prd.py, find the line that says "import os" and add the following right below it:
```
os.environ['KMP_DUPLICATE_LIB_OK']='True'
```

## Switch between GPU and CPU modes
Let's assume you installed the GPU version. You can adjust these instructions if you did CPU first, of course.
Clone your existing conda environment:
```
conda create --name prdcpu --clone progrockdiffusion
conda activate prdcpu
```
Now install the CPU version of pytorch:
```
pip install torch==1.11.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
pip install torchvision==0.12.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
pip install torchaudio==0.11.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
```
All set! You can now switch between the two by simply doing:
```
conda activate progrockdiffusion
conda activate prdcpu
```

# Notes

- Currently Superres Sampling doesn't work, it will crash.
- Animations are untested but likely not working

# TODO

- Get Animations working
