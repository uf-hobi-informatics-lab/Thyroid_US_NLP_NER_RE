# Thyroid US - NLP Model Package

This package contains a trained NLP transformer model for Clinical Named Entity Recognition and Clicnial Relation Extraction, along with scripts to run predictions on new data.

## Setup

1.  **Clone the github repo:** Unzip the `2025_packaging.zip` file to a location of your choice. Let's assume you extract it to `/home/yourfriend/projects/`. This will create a directory structure like:
    `/home/yourfriend/projects/2025_packaging/`

2.  **Navigate to the package root:**
    ```bash
    cd /home/yourfriend/projects/2025_packaging
    ```
    **IMPORTANT:** All subsequent commands should be run from within this directory.

3.  **Create a Python virtual environment (optional):**

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Make sure you create a `requirements.txt` file at the root of `2025_packaging` that lists all your Python dependencies.)*

5. **Download the models:**:
    Visit the dropbox folder: https://www.dropbox.com/scl/fo/o6a644wd3m61kdynvq0uv/AGp5T_sDCh0RSjKyx-sd1EY?rlkey=ry5voqckzu9woy3kmeqbn1bg8&st=50jsvgom&dl=0 and download the zipped folder. Extract it and copy it in the ROOT_DIR.

## Usage

Once you are in the `2025_packaging` directory and your environment is set up, follow these steps:

### Step 1: Configure the run_config.yml file

This step sets up the raw input file folder, output folder and other path. 

1. **raw_data_dir:** This should be the **absolute file path** to your raw text files directory. Currently set to `input_text_files`
2. **root_dir:** This is the **absolute file path**  where the outputs, logs and intermediate files are generated and stored. These will generate the .ann containing the predictions of both NER and RE. Currently set to `output`

### Step 2: Execute the run.sh file:

This execute the pipeline running both the NER and RE pipelines.
Command line argument includes:
1. -c: the path to the config file
2. -e: name of the experiment
3. -n: which gpu node to use. 

```bash
run.sh -c run_config.yml -e thyroid_nodule_us_pred_2025 -n 0

```     
      
### Step 3: Check the predictions

If step 2 executes correctly, the output folder should create two folders `brat` and `brat_re`.


**Input:** `.ann` files from `output/ann_files_NER_BIO/`
**Output:** `test.tsv` in `output/brat_file/` (or whatever path you specify)

```bash

python generate_test_tsv.py \
    --input_BIO_FORMATTED_dir output/ann_files_NER_BIO_formatted_output/ \
    --mapping_file src_utils/mapping/all_comb.pkl \
    --output_tsv_path output/brat_file/test.tsv
```