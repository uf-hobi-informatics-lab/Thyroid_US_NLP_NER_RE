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
   
    *Using conda:*

    ```bash
    conda create -n thyroid_us python=3.9.12 -y
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
    The trained models are required to run the pipeline.
    1. Download the model archive from this [Dropbox link](https://www.dropbox.com/scl/fo/o6a644wd3m61kdynvq0uv/AGp5T_sDCh0RSjKyx-sd1EY?rlkey=ry5voqckzu9woy3kmeqbn1bg8&st=50jsvgom&dl=0).
    2. Unzip the downloaded folder.
    3. You should now have a directory named `models`. Place this entire `models` directory inside the project root (`Thyroid_US_NLP_NER_RE/`).

## Project Structure:

The project strcuture should look like:

```plainfile
├── ClinicalTransformerClassification
├── ClinicalTransformerNER
├── ClinicalTransformerRelationExtraction
├── encode_text.py
├── input_text_files
├── models
│   ├── GatorTron_NER_model
│   └── GatorTron_Rel_Extraction_model
├── NLPreprocessing
├── outputs
├── README.md
├── requirements.txt
├── run_config.yml
├── run_ner.py
├── run_post_processing.py
├── run_relation.py
├── run.sh
└── src_utils
    ├── aggregate_entities.py
    ├── convert_tsv.py
    ├── functions.py
    ├── __init__.py
    ├── mapping
    └── rule_based_system.py
```


## Usage
Once you are in the `Thyroid_US_NLP_NER_RE` directory and your environment is set up, follow these steps:

**Step 1: Prepare Input Data**

Place your raw ultrasound reports as plain text files (`.txt`) inside the `input_text_files` directory. You can use a different directory by specifying it in the configuration file. Ideally the name of the files should be note_id

### Step 2: Configure Settings 

Edit `run_config.yml`: 

- raw_data_dir: Confirm your path to your input text files (default: `input_text_files`)
- root_dir: confirm your path to where theoutputs will be saved (default: `output`)
     
### Step 3: Execute the run.sh file:

This execute the pipeline running both the NER and RE pipelines.
Command line argument includes:
1. -c: The path to the config file (default: run_config.yml)
2. -e: The name of the experiment (default: `thyroid_nodule_us_pred_2025`)
3. -n: The GPU device ID to use. (default: 0)
4. -s: Minimum size (cm) for nodule filtering. Nodules with any dimension greater than or equal to this value will be kept. Set to 0 to disable filtering (default: 1)

```bash
bash run.sh -c run_config.yml -e thyroid_nodule_us_pred_2025 -n 0 -s 1
```

This command runs the pipeline on GPU `0` and filters for nodules ≥ `1cm`.

### Step 4: View Results 

The processed output files will be generated in the `outputs` directory (or your specified `root_dir`). The primary results include:

- `thyroid_results_filtered.csv`: Structured extracted data for thyroid nodules.
- `lymph_results_filtered.csv`: Structured extracted data for lymph nodes.
     

## Troubleshooting 

###### Q: The script fails with "CUDA out of memory"?
A: Try using a different GPU (-n 1, -n 2, etc.) or process fewer files at once. 

###### Q: Models not found error?
A: Ensure you've downloaded and extracted the models to the correct directory. 

###### Q: Permission denied when running run.sh?
A: Make the script executable: chmod +x run.sh 

###### Q: Dependencies installation fails?
A: Ensure you're using Python 3.9.12 and have activated your virtual environment. 

## Contact:
Please contact us or post an issue if you have any questions.
    
    Aman Pathak (aman.pathak@ufl.edu)
    Yonghui Wu (yonghui.wu@ufl.edu)

## Related Projects

This work builds upon the following frameworks:  

[ClinicalTransformerNER](https://github.com/uf-hobi-informatics-lab/ClinicalTransformerNER)  
[ClinicalTransformerRelationExtraction](https://github.com/uf-hobi-informatics-lab/ClinicalTransformerRelationExtraction)  
[ClinicalTransformerClassification](https://github.com/uf-hobi-informatics-lab/ClinicalTransformerClassification.git)

## Citation:
If you use this work, please cite our paper:

Pathak, A., Yu, Z., Paredes, D., Monsour, E. P., Rocha, A. O., Brito, J. P., Ospina, N. S., & Wu, Y. (2024). Extracting Thyroid Nodules Characteristics from Ultrasound Reports Using Transformer-based Natural Language Processing Methods. AMIA ... Annual Symposium proceedings. AMIA Symposium, 2023, 1193–1200.
https://pmc.ncbi.nlm.nih.gov/articles/PMC10785862/