
 [![Minimum Python Version](https://img.shields.io/badge/Python-%3E=%203.9-blue)](https://www.python.org/downloads/) [![Maximum Python Version Tested](https://img.shields.io/badge/Python-%3C=%203.12-blueviolet)](https://www.python.org/downloads/) [![Supported Python Versions](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
 
 This project has been tested and is compatible with Python versions 3.9, 3.10, 3.11, and 3.12. While it might work on other versions, these are the officially supported and tested ones.

# Getting Started


Before you can run this project, you need to have python installed on your system

## Option 1: Setting up a Python Virtual Enviroment (venv)

This is the standard way to create an isolated Python enviroment.

**Steps:**

1. **Install pip (if you don't have it):**
  ```bash
   python -m ensurepip --default-pip
  ```
  or on some systems:
 ```bash
  sudo apt update
  sudo apt install python3-pip
  ```
2. **Create a virtual enviroment:**
```bash
   python -m venv venv
  ```
This command creatas a new directory named `venv` (you can choose a different name if you prefer) containing a copy of the Python interpreter and necessary supporting files.

3. **Activate the virtual enviroment:**
* **On macOS and Linux:**
```bash
  source venv/bin/activate
  ```
* **On Windows (command promt):**
```bash
  venv\Scripts\activate
  ```
* **On Windows (PowerShell):**
```bash
  .\venv\Scripts\Activate.ps1
  ```
Once the activated, you'll see `(venv)` at the beginning of your terminal promt.

4. **Install project dependencies:**
   Once the virtual enviroment is activated, you can install the required packages listed in the `requirements.txt` file:
  ```bash
  pip install  .
  ```

5. **macOS Users: Install OpenMP for XGBoost (Required)**
   
   On macOS, XGBoost requires the OpenMP library. Install it using Homebrew:
   ```bash
   brew install libomp
   ```
   
   If you don't have Homebrew installed, visit [https://brew.sh/](https://brew.sh/) for installation instructions.
   
   After installing OpenMP, you may need to reinstall XGBoost:
   ```bash
   pip install --force-reinstall xgboost
   ```

6. **Deactivate the virtual enviroment (when you are done):**
   ```bash
    deactivate
   ```
   This will return you to your base Python enviroment.

## Option 2: Setting up a Conda Enviroment

1. Create the environment from the `requirements.txt` file.  This can be done using anaconda, miniconda, miniforge, or any other environment manager.
```
conda create -n qbc python==3.12

```
* Note: if you receive the error `bash: conda: command not found...`, you need to install some form of anaconda to your development environment.
2. Activate the new environment:
```
conda activate qbc
pip install .
```

3. **macOS Users: Install OpenMP for XGBoost (Required)**
   
   On macOS, XGBoost requires the OpenMP library. Install it using Homebrew:
   ```bash
   brew install libomp
   ```
   
   If you don't have Homebrew installed, visit [https://brew.sh/](https://brew.sh/) for installation instructions.
   
   After installing OpenMP, you may need to reinstall XGBoost:
   ```bash
   pip install --force-reinstall xgboost
   ```

4. Verify that the new environment and packages were installed correctly:
```
conda env list
pip list
```
<!-- * Additional resources:
   * [Connect to computing cluster](http://ccc.pok.ibm.com:1313/gettingstarted/newusers/connecting/)
   * [Set up / install Anaconda on remote linux server](https://kengchichang.com/post/conda-linux/)
   * [Set up remote development environment using VSCode](https://code.visualstudio.com/docs/remote/ssh) -->

## Option 3: Using Galaxy (Cloud-Based, No Local Installation)

If you prefer not to install QBioCode on your local or personal machine, you can use [Galaxy](https://usegalaxy.org/), a free, web-based platform for data-intensive biomedical research.

```{admonition} Why Galaxy?
:class: tip
- **No installation required**: Run everything in your browser
- **Free computational resources**: Access to cloud computing
- **Jupyter notebook support**: Run QBioCode tutorials directly
- **Persistent workspace**: Your work is saved in the cloud
```

### Step 1: Register for a Galaxy Account

1. Go to [https://usegalaxy.org/](https://usegalaxy.org/)
2. Click **"Login or Register"** in the top menu
3. Select **"Register"** and fill in:
   - Email address
   - Password
   - Public name (username)
4. Click **"Create"** to complete registration
5. Verify your email address (check your inbox for confirmation link)

### Step 2: Launch a Qiskit Jupyter Notebook (Recommended - Pre-installed QBioCode)

1. **Log in** to your Galaxy account at [https://usegalaxy.org/](https://usegalaxy.org/)

2. From the left menu, select **"Interactive Tools"**

3. Search for **"Qiskit Jupyter notebook"**

4. Click on the Qiskit Jupyter tool to launch it

5. Configure the notebook environment:
   - **Allocate resources**: Default settings are usually sufficient
   - Click **"Execute"** or **"Run Tool"**

6. Wait for the notebook server to start (this may take 1-2 minutes)

7. Once ready, click the **link** to open your Jupyter environment in a new tab

```{admonition} Pre-installed QBioCode
:class: tip
The **Qiskit Jupyter notebook** in Galaxy comes with QBioCode pre-installed! You can start using it immediately without any installation steps. However, note that this may not be the latest version of the code.
```

### Step 3: Using QBioCode (Pre-installed Version)

Once your Qiskit Jupyter notebook server is running:

1. **Verify QBioCode is available**:
   
   Open a new notebook or terminal and run:
   ```python
   import qbiocode as qbc
   print(f"QBioCode version: {qbc.__version__}")
   ```

2. **Access the tutorials**:
   
   The tutorials should be available in the file browser. Navigate to the QBioCode tutorial directory and open any notebook:
   - `Artificial_data_generation/example_data_generation.ipynb`
   - `QProfiler/example_qprofiler.ipynb`
   - `QSage/qsage.ipynb`
   - `Quantum_Projection_Learning/QPL_example.ipynb`

### Step 3b: Install Latest Version (Optional)

If you need the latest version of QBioCode with the most recent features and bug fixes:

1. **Open a new terminal** in Jupyter:
   - Click **"File" → "New" → "Terminal"** (in JupyterLab)
   - Or use the **"New" → "Terminal"** button (in classic Jupyter)

2. **Install the latest QBioCode from GitHub**:

   ```bash
   # Clone the repository
   git clone https://github.com/IBM/QBioCode.git
   cd QBioCode
   
   # Install QBioCode (this will upgrade the pre-installed version)
   pip install --upgrade .
   ```

3. **Verify the updated installation**:

   ```bash
   python -c "import qbiocode; print(f'QBioCode version: {qbiocode.__version__}')"
   ```

```{note}
After upgrading, you may need to restart your Jupyter kernel for changes to take effect:
**Kernel → Restart Kernel**
```

### Step 4: Run QBioCode Tutorials

1. **Navigate to the tutorial directory**:

2. **Open a tutorial notebook**:
   - In the Jupyter file browser, navigate to `QBioCode/tutorial/`
   - Click on any tutorial notebook to open it:
     - `Artificial_data_generation/example_data_generation.ipynb`
     - `QProfiler/example_qprofiler.ipynb`
     - `QSage/qsage.ipynb`
     - `Quantum_Projection_Learning/QPL_example.ipynb`

3. **Run the tutorial**:
   - Execute cells sequentially using **Shift+Enter**
   - Follow the instructions in each notebook

### Tips for Using Galaxy

```{tip}
**Best Practices:**
- **Save your work frequently**: Use File → Save to preserve your progress
- **Download important results**: Export notebooks and data files to your local machine
- **Monitor resource usage**: Galaxy sessions have time limits; plan accordingly
- **Use version control**: Consider connecting to GitHub for better workflow management
```

```{warning}
**Important Limitations:**
- Galaxy sessions may timeout after inactivity (typically 1-2 hours)
- Computational resources are shared; quantum simulations may be slower
- Large datasets may require local installation for better performance
- IBM Quantum hardware access requires separate IBM Quantum account setup
```

### Troubleshooting Galaxy Installation

**Issue: pip install fails**
```bash
# Try upgrading pip first
pip install --upgrade pip
pip install .
```

**Issue: Import errors**
```bash
# Restart the kernel: Kernel → Restart Kernel
# Then re-import
import qbiocode
```

**Issue: XGBoost errors on macOS**
```bash
# Install OpenMP library
brew install libomp
pip install --force-reinstall xgboost
```

**Issue: Session timeout**
- Save your work regularly
- Download notebooks before closing
- Restart the interactive tool if needed

### Alternative: Google Colab

Another cloud-based option is [Google Colab](https://colab.research.google.com/):

```python
# In a Colab notebook cell:
!git clone https://github.com/IBM/QBioCode.git
%cd QBioCode
!pip install .
```

Then upload tutorial notebooks from the `tutorial/` directory.

---

<a name="running_qbiocode"></a>
