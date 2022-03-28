# Software prerequisties
Only MacOS Big Sur has been tested. Other versions may work.
Please note that only CPU Mode is currently supported on MacOS. It will be very slow.

# Setup
## Install git (and Xcode, if required)
Open a terminal and run this command:
```
git --version
```
It will either tell you the version if git is installed, or prompt you to install Xcode which includes git.

## Install Python
Download and install Python 3.8: https://www.python.org/ftp/python/3.8.10/python-3.8.10-macos11.pkg

Install pip:
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

## Install the basic CPU version of PyTorch (warning - very slow!)
```
pip install -r requirements.cpu.txt
```

## Finally:
```
export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1
pip install grpcio
```

## Test
To ensure everything is working, you can run progrockdiffusion with its default settings:
```
python3 prd.py
```
When it's done, check the images_out folder to see if the image was created!

**You can now return to the [README](README.md) file to learn how to use Progrockdiffusion**
