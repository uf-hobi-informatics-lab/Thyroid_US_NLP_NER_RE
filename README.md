# Thyroid_US_NLP_NER_RE - NLP Model Package

This package provides a trained transformer model for analyzing thyroid ultrasound reports. It performs two key tasks:
- **Named Entity Recognition (NER)**: Identifies medical entities in text
- **Relation Extraction (RE)**: Finds relationships between identified entities

## Prerequisites

- Python 3.9.12
- CUDA-compatible GPU (recommended)
- Git

## Setup

1.  **Clone the github repo:** 
    ```bash
    git clone https://github.com/uf-hobi-informatics-lab/Thyroid_US_NLP_NER_RE.git
    ```
    This will create the project directory structure.

2.  **Navigate to the package root:**
    ```bash
    cd Thyroid_US_NLP_NER_RE
    ```
    **IMPORTANT:** All subsequent commands should be run from within this directory.

3.  **Create a Python virtual environment (optional):**
    ```bash
    conda create -n thyroid_us python=3.9.12
    conda activate thyroid_us
    ```

4. **Install `torch` with CUDA**
    ```bash 
    pip install torch==1.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
    ```

5.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

6. **Download the models:**

    Visit the dropbox folder: https://www.dropbox.com/scl/fo/o6a644wd3m61kdynvq0uv/AGp5T_sDCh0RSjKyx-sd1EY?rlkey=ry5voqckzu9woy3kmeqbn1bg8&st=50jsvgom&dl=0 and download the zipped folder. 
    Extract it and copy it in the ROOT_DIR.

## Project Structure:

The project strcuture should look like:


## Usage
Once you are in the `Thyroid_US_NLP_NER_RE` directory and your environment is set up, follow these steps:

### Step 1: Prepare Your Data 
Place your text files in the `input_text_files` folder (or specify a different folder in the config). 

### Step 2: Configure Settings 

Edit `run_config.yml`: 

    **raw_data_dir**: Path to your input text files (default: `input_text_files`)
    **root_dir**: Where outputs will be saved (default: `output`)
     
### Step 2: Execute the run.sh file:

This execute the pipeline running both the NER and RE pipelines.
Command line argument includes:
1. -c: the path to the config file (Currently it uses run_config.yml)
2. -e: name of the experiment (Currently set to `thyroid_nodule_us_pred_2025`)
3. -n: which gpu node to use. (Currently set to 0)
4. -size: specify the filtering size of the nodule in cm. If size is 0, it will not filter. If size is mentioned, it will filter the nodule whose dimension (any) is greater than or equal to the size. (currently set to 1)

```bash
run.sh -c run_config.yml -e thyroid_nodule_us_pred_2025 -n 0 -size 1
```     

### Step 4: View Results 

Check the output folder for: 

    thyroid_results_filtered.csv - Thyroid predictions
    lymph_results_filtered.csv - Lymph node predictions
     

## Troubleshooting 

###### Q: The script fails with "CUDA out of memory"
A: Try using a different GPU (-n 1, -n 2, etc.) or process fewer files at once. 

###### Q: Models not found error
A: Ensure you've downloaded and extracted the models to the correct directory. 

###### Q: Permission denied when running run.sh
A: Make the script executable: chmod +x run.sh 

###### Q: Dependencies installation fails
A: Ensure you're using Python 3.9.12 and have activated your virtual environment. 