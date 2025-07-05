from src_utils.functions import summarize_thyroid_ann, summarize_lymph_ann, rearrange_thyroid_df, rearrange_lymph_df
from src_utils.rule_based_system import filtering_size

import unicodedata, os,sys, pickle
import argparse, torch
import cProfile, yaml, copy
from encode_text import preprocessing
from pathlib import Path
from NLPreprocessing.annotation2BIO import read_annotation_brat
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial


def multiprocess(partial_func, ann_files, worker=8):
    result = []
    with ThreadPool(worker) as p:
        result.extend(p.map(partial_func, ann_files))

    results_final = []    
    for answer in result:
        if isinstance(answer, list): 
            results_final.extend(answer)  
        else:
            results_final.extend(list(answer))
    results_final = pd.DataFrame(results_final)
    return results_final



def process_thyroid_nodule(ann_files, OUTPUT_DIR, BRAT_RE_DIR, FILE_DIR, worker = 8):
    # Create a partial function with the static argument fixed
    summarize_ann_with_fixed_dir = partial(summarize_thyroid_ann, FILE_DIR=FILE_DIR)
    
    # Use ThreadPool to parallelize the processing of annotation files
    print("Processing thyroid nodule annotations...")
    results_final = multiprocess(summarize_ann_with_fixed_dir, ann_files, worker)

    # Save the results to a CSV file before cleaning and filtering
    results_final.to_csv(os.path.join(OUTPUT_DIR, "thyroid_results_raw.csv"), index=False)

    # Split the results into two parts: nodule and lymph using the 'concept_cat' column
    nodule_results = results_final[results_final['concept_cat'] == 'thyroid_nodule']
    print("Total nodule: ", nodule_results.shape)
    # print("SAVING RESULTS TO CSV BEFORE CLEANING AND FILTERING")

    return nodule_results
    

def process_lymph_nodule(ann_files, OUTPUT_DIR, BRAT_RE_DIR, FILE_DIR, worker = 8):
    # Create a partial function with the static argument fixed
    summarize_ann_with_fixed_dir = partial(summarize_lymph_ann, FILE_DIR=FILE_DIR)
    results_final = multiprocess(summarize_ann_with_fixed_dir, ann_files, worker)

    # Save the results to a CSV file before cleaning and filtering
    results_final.to_csv(os.path.join(OUTPUT_DIR, "lymph_results_raw.csv"), index=False)

    # Split the results into two parts: nodule and lymph using the 'concept_cat' column
    lymph_results = results_final[results_final['concept_cat'] == 'lymph']
    print("Total lymph: ", lymph_results.shape)
    # print("SAVING RESULTS TO CSV BEFORE CLEANING AND FILTERING")

    return lymph_results




def convert_to_csv(OUTPUT_DIR, BRAT_RE_DIR, FILE_DIR, FILTERSIZE, worker = 8):
    ann_files = list(Path(BRAT_RE_DIR).glob("*.ann"))
    print("Total .ann file:", len(ann_files))
    # print(ann_files)
    
    # Process thyroid nodules
    thyroid_df = process_thyroid_nodule(ann_files, OUTPUT_DIR, BRAT_RE_DIR, FILE_DIR, worker)
    print("======== Nodule Results ========")
    print("Total thyroid nodules:", thyroid_df.shape)

    # Process lymph nodes
    lymph_df = process_lymph_nodule(ann_files, OUTPUT_DIR, BRAT_RE_DIR, FILE_DIR, worker)
    print("======== Lymph Results ========")
    print("Total lymph nodes:", lymph_df.shape)

    print(f"Filtering nodules with size greater than or equal to {FILTERSIZE} cm")
    thyroid_df, lymph_df = filtering_size(FILTERSIZE, thyroid_df, lymph_df)
    print("Filtering completed.")

    # Rearrange the columns in the DataFrame
    thyroid_df_copy = rearrange_thyroid_df(thyroid_df)
    lymph_df_copy = rearrange_lymph_df(lymph_df)
    

    # Save the filtered nodule results to a CSV file
    thyroid_df_copy.to_csv(os.path.join(OUTPUT_DIR, "thyroid_results_filtered.csv"), index=False)
    # print(thyroid_df_copy.columns)
    # print(thyroid_df_copy.sample(1).to_dict("records"))


    # Save the filtered lymph results to a CSV file
    lymph_df_copy.to_csv(os.path.join(OUTPUT_DIR, "lymph_results_filtered.csv"), index=False)
    # print(lymph_df_copy.columns)
    # print(lymph_df_copy.sample(1).to_dict("records"))
    print("================ ")
    print("Conversion to CSV completed and saved in", OUTPUT_DIR)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, default="run_config.yaml", help="configuration file")
    parser.add_argument("--experiment", type=str, required=True, default="thyroid_nodule_us_pred_2025", help="experiement to run")
    parser.add_argument("--gpu_nodes", nargs="+", default=0, help="gpu_device_id")
    # Add size argument to specify the filtering size of the nodule in cm. If size is 0, it will not filter.
    # If size is mentioned, it will filter the nodule whose dimension (any) is greater than or equal to the size.
    parser.add_argument("--filter_size", default=0, type=int, help="size of the nodule to filter in cm. If size is 0, it will not filter. If size is mentioned, it will filter the nodule whose dimension (any) is greater than or equal to the size.")
    args = parser.parse_args()


    if args.filter_size < 0:
        raise ValueError("Filter size cannot be negative")
    
    # Load configuration
    with open(Path(args.config), 'r') as f:
        experiment_info = yaml.safe_load(f)[args.experiment]
    experiment_info['gpu_nodes'] = args.gpu_nodes

    OUTPUT_DIR = experiment_info["root_dir"]
    BRAT_RE_DIR = OUTPUT_DIR + "/brat_re"
    FILE_DIR = experiment_info["raw_data_dir"]

    print("=============== Step 3: Run Post Processing ===============\n")
    
    # print(BRAT_RE_DIR, FILE_DIR)
    convert_to_csv(OUTPUT_DIR, BRAT_RE_DIR, FILE_DIR, args.filter_size)
    print("=============== Step 3 Completed ===============")
