from src_utils.functions import summarize_thyroid_ann
import unicodedata, os,sys, pickle
import argparse, torch
import cProfile, yaml, copy
from encode_text import preprocessing
from pathlib import Path
from NLPreprocessing.annotation2BIO import read_annotation_brat

from multiprocessing.dummy import Pool as ThreadPool 





def convert_to_csv(OUTPUT_DIR, BRAT_RE_DIR, FILE_DIR, worker = 8):
    ann_files = list(Path(BRAT_RE_DIR).glob("*.ann"))
    print("Total .ann file:", len(ann_files))
    
    print(ann_files)
    
    
    result = []
    with ThreadPool(worker) as p:
        result.extend(p.map(summarize_thyroid_ann, ann_files))
    results_final = []
    for answer in result:
        if isinstance(answer, list): 
            results_final.extend(answer)  
        else:
            results_final.extend(list(answer))
    results_final = pd.DataFrame(results_final)
    print(results_final.tail(5))
    print("Total nodule: ", result_final.shape)
    results_final.to_csv(OUTPUT_DIR + "/test.csv", index = False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="configuration file")
    parser.add_argument("--experiment", type=str, required=True, help="experiement to run")
    parser.add_argument("--gpu_nodes", nargs="+", default=None, help="gpu_device_id")
    args = parser.parse_args()
    
    # Load configuration
    with open(Path(args.config), 'r') as f:
        experiment_info = yaml.safe_load(f)[args.experiment]
    experiment_info['gpu_nodes'] = args.gpu_nodes
    OUTPUT_DIR = experiment_info["root_dir"]
    BRAT_RE_DIR = OUTPUT_DIR + "/brat_re"
    FILE_DIR = experiment_info["raw_data_dir"]
    
    print(BRAT_RE_DIR, FILE_DIR)
    convert_to_csv(OUTPUT_DIR, BRAT_RE_DIR, FILE_DIR)
    
    