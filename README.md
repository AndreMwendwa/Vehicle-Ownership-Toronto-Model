# Two-Stage Hurdle Model of Vehicle Ownership in Toronto

## Description

These scripts provide the code used to carry out analysis of vehicle ownership patterns in the Greater Toronto Area using an ordered model.

## Setting up a Python Virtual Environment

A Python virtual environment is a self-contained directory tree that contains a Python installation for a particular version of Python, plus a number of additional packages.

### Prerequisites

- Python installed on your system. You can download and install Python from [python.org](https://www.python.org/downloads/). The original model was run using Python 3.9.6, but other versions should work as well. 

### Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/AndreMwendwa/Vehicle-Ownership-Toronto-Model.git
   cd Vehicle-Ownership-Toronto-Model
	```
	
2. **Create a virtual environment**
	```bash
	python3 -m venv venv
	
3. **Activate the Virtual Environment:**
  - On macOS and Linux:
	```bash
	source venv/bin/activate

 - On Windows (cmd.exe):
	```bash
	venv\Scripts\activate.bat
 
 - On Windows (PowerShell):
	```bash
	.\venv\Scripts\Activate.ps1
After activation, your shell prompt will change and prepend the name of the virtual environment ((venv) by default).
	
	
4. **Install Required Packages:**
	```bash
	pip install -r requirements.txt
	```


## Running Order of Code
-	Ordered Model Data Processing.ipynb
-	Min_distance_calculation.py
-	Neighbourhood_Characteristics.ipynb
-	Entropy_processing_DMTI_data.ipynb
-	Survey data (the main file)
-	Marginal Effects (post processing)

## Authors

[Mwendwa Kiko](https://www.linkedin.com/in/mwendwa-kiko/)
