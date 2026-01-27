# Installing Python 3.12.7 on OIST Cluster

## Option 1: Use Pre-built Module (Recommended)

First, check if Python 3.12 is already available:

```bash
# Check available Python versions
module avail python

# If Python 3.12 exists, load it
module load python/3.12

# Verify
python3 --version
```

If Python 3.12.x is available, **skip the installation** and just use the module!

## Option 2: Install from Source

If Python 3.12.7 is not available as a module, install it yourself:

### Step 1: Run Installation Script

```bash
# On a compute node (don't build on login node!)
srun -p short -t 1:00:00 --mem=8G -c 8 --pty bash

# Run the installation script
bash install_python_3.12.7.sh
```

**Installation takes ~15-30 minutes**

The script will:
- Download Python 3.12.7 source
- Build with optimizations
- Install to `$HOME/apps/python/3.12.7`
- Create an activation script
- Run tests and verify installation

### Step 2: Add to Your Environment

After installation completes, add to your `~/.bashrc`:

```bash
# Add this line to ~/.bashrc
source $HOME/apps/python/3.12.7/activate.sh
```

Or activate manually when needed:

```bash
source $HOME/apps/python/3.12.7/activate.sh
```

### Step 3: Create Virtual Environment for PG_005

```bash
# Activate Python 3.12.7
source $HOME/apps/python/3.12.7/activate.sh

# Go to your project
cd $HOME/PG_005

# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install your project dependencies
pip install -e .
```

## Using with Slurm Scripts

### If using module (Option 1):

```bash
#!/bin/bash
#SBATCH ...

module load python/3.12

cd $HOME/PG_005
source .venv/bin/activate
python batch_process.py
```

### If using custom installation (Option 2):

```bash
#!/bin/bash
#SBATCH ...

source $HOME/apps/python/3.12.7/activate.sh

cd $HOME/PG_005
source .venv/bin/activate
python batch_process.py
```

## Verification

After installation, verify everything works:

```bash
# Check Python version
python3 --version
# Should show: Python 3.12.7

# Check pip
pip3 --version

# Check it's your installation
which python3
# Should show: /home/your-username/apps/python/3.12.7/bin/python3
```

## Requirements for Building

The build script assumes these are available (usually are on OIST cluster):
- gcc/g++ compiler
- make
- wget
- Standard development libraries (zlib, openssl, etc.)

If the build fails due to missing dependencies, you may need to load additional modules:

```bash
module load gcc
module load openssl
# etc.
```

## Troubleshooting

### Build fails with missing libraries

Load required modules before building:
```bash
module load gcc/9.3.0
module load openssl
bash install_python_3.12.7.sh
```

### "No module named '_ctypes'"

Install libffi-dev before building (contact SCS if not available)

### Out of disk space

Clean up the build directory:
```bash
rm -rf $HOME/tmp/python-3.12.7-build
```

## Why Python 3.12.7?

Your project currently uses Python >= 3.11. Python 3.12.7 includes:
- Performance improvements
- Better error messages
- Improved type hints
- Compatibility with latest packages

However, **Python 3.11+ is sufficient** for your project. If Python 3.11 module is available, you can use that instead!

## Quick Decision Tree

```
Do you need Python 3.12 specifically?
├─ No → Use available module (module load python/3.11 or python/3.12)
└─ Yes → Check module avail python
    ├─ python/3.12 exists → Use module
    └─ Not available → Install from source (use script)
```

## Contact

If you encounter issues during installation, contact OIST SCS:
- Email: scs@oist.jp
- Documentation: https://groups.oist.jp/scs
