import pickle
from NLPreprocessing.annotation2BIO import read_annotation_brat

CONTEXT_WINDOW = 0

# Increase the context window size if needed
# CONTEXT_WINDOW = 100

def rearrange_thyroid_df(thyroid_df):
    thyroid_df_copy = thyroid_df.copy()
    new_order = ['note_id', 'id', 'concept_cat','concept_value', 'size_numeric', 'size_qualitative', 'Size','size_numeric_dim_01', 'size_numeric_dim_02', 'size_numeric_dim_03',
       'unit', 'shape', 'laterality', 'composition', 'echogenic_foci',  'vascularity', 'margins', 'TIRADS_Score','context_start', 'context_end', 'context', 'location', 'echogenicity',
       'TIRADS_risk_category_processed', 'TIRADS_is_con', 'Risk_description', 'lymph', 'TIRADS_risk_category'
       ]
    # Ensure the columns are in the specified order
    thyroid_df_copy = thyroid_df_copy[new_order]

    return thyroid_df_copy

def rearrange_lymph_df(thyroid_df):
    thyroid_df_copy = thyroid_df.copy()
    new_order = ['note_id', 'concept_cat','concept_value', 'Risk_description', 'context_start', 'context_end', 'context','echogenic_foci', 'margins', 'Peripheral vascularization',
       'composition', 'echogenicity', 'hilium', 'laterality', 'Intranodal Necrosis', 'size_numeric',]
       
    # Ensure the columns are in the specified order
    thyroid_df_copy = thyroid_df_copy[new_order]

    return thyroid_df_copy   

def read_note(file_path):
    with open(file_path, "r") as file:
        contents = file.read()
    return contents

def thyroid_row():
    return {k: "" for k in ['context_start','context_end', 'note_id','id', 'concept_cat', 'concept_value', 'size_numeric', 'shape', 'location_', 'laterality', 'composition', 'echogenic_foci', "size_qualitative",
                            'vascularity', 'margins', 'echogenicity_', 'TIRADS_Score', 'Risk_description', 'lymph','TIRADS_risk_category']}

def lymph_row():
    return {k: "" for k in ["echogenic_foci", "margins", "Peripheral vascularization", "composition", "echogenicity", "hilium", "laterality", "Intranodal Necrosis", "size_numeric"]}

def get_rel_prop_name(ner_concept_name):
    if ner_concept_name[-1] == "_": # for location_, echogenicity_
        return ner_concept_name[:-1]
    else:
        return ner_concept_name
    
def load_mapping_file():
    with open("src_utils/mapping/entp2rel.pkl", "rb") as file:
        mapping_file = pickle.load(file)
    return mapping_file
        
def check_valid_thyroid_relationship(rel_concept_name):
    try:
        mapping_file
    except NameError as e:
        mapping_file = load_mapping_file()
        all_thy_combo = [x for x in mapping_file.values() if x.startswith("thyroid")]

    if "thyroid_nodule-{0}".format(rel_concept_name) in all_thy_combo:
        return True
    else: 
        return False

def summarize_thyroid_ann(ann_note, FILE_DIR = "input_text_files/"):
    note_id = ann_note.stem.split("_")[0]
    note_text = read_note(FILE_DIR + str(note_id) + ".txt") 
    _, ners, rels = read_annotation_brat(ann_note)

    nodules_dicts = {}
    nodule_set = set()
    nodule_flag, lymph_flag = False, False
    
    for idx, ner in enumerate(ners, start = 1):
        if ner[1] == "thyroid_nodule":
            nodule_flag = True
            
            nodules_dicts["T" + str(idx)] = thyroid_row()
            nodules_dicts["T" + str(idx)]["note_id"] = note_id
            nodules_dicts["T" + str(idx)]["concept_cat"] = ner[1]
            nodules_dicts["T" + str(idx)]["concept_value"] = ner[0]
            nodules_dicts["T" + str(idx)]["id"] = "T" + str(idx)
            nodules_dicts["T" + str(idx)]["context_start"] = ner[2][0]
            nodules_dicts["T" + str(idx)]["context_end"] = ner[2][1]
            nodules_dicts["T" + str(idx)]["context"] = ""      
            nodule_set.add("T" + str(idx))
            
            
        if ner[1] == "lymph":
            lymph_flag = True
            lymph_entity = "T" + str(idx)
            lymph_dict = lymph_row()
            
        
    if not nodule_flag:
        temp = {"-T100" : thyroid_row()}
        temp["-T100"]["note_id"] = str(ann_note).split("/")[-1].split(".")[0]
        return list(temp.values())
    
    for idx, rel in enumerate(rels, start = 1):
        _,e1,e2 = rel

        if e1 in nodule_set:
            pos_in_ners = int(e2[1:]) - 1
            prop_in_ners = ners[pos_in_ners][1]
            if nodules_dicts[e1].get(prop_in_ners,0):
                nodules_dicts[e1][prop_in_ners] =  nodules_dicts[e1][prop_in_ners] + ", " + ners[pos_in_ners][0]
            else:
                nodules_dicts[e1][prop_in_ners] =  ners[pos_in_ners][0]
            
            start = nodules_dicts[e1]["context_start"] = min(nodules_dicts[e1]["context_start"], ners[pos_in_ners][2][0])
            end = nodules_dicts[e1]["context_end"] = max(nodules_dicts[e1]["context_end"], ners[pos_in_ners][2][1])
            
            if not note_text:
                note_text = read_note(note_id)
            
            print("note_id:", note_id, "id", "T" + str(idx))
            print("context_start:", start, "context_end:", end)
            print("context:", note_text[max(0, start - CONTEXT_WINDOW): min(len(note_text), end + CONTEXT_WINDOW)])
            print("\n")


            # Extract the context from the note text
            nodules_dicts[e1]["context"] = note_text[max(0, start - CONTEXT_WINDOW): min(end + CONTEXT_WINDOW, len(note_text))]
            
    ner_pos = 0
    while ner_pos < len(ners):
        if ners[ner_pos][1] == "thyroid_nodule":
            j = ner_pos+1
            while j < len(ners) and not ners[j][1] == "thyroid_nodule":
                property_name = get_rel_prop_name(ners[j][1])
                # Check if thyroid_nodule-<relation property @ner[j] is a valid relationship 
                if check_valid_thyroid_relationship(property_name):
                    
                    #Check if the relationship already exists
                    if not nodules_dicts["T" + str(ner_pos + 1)].get(property_name,0):
                        nodules_dicts["T" + str(ner_pos + 1)][property_name] = ners[j][0]
                j +=1
            ner_pos = j - 1
        ner_pos += 1            
    return list(nodules_dicts.values())


def summarize_lymph_ann(ann_note, FILE_DIR = "input_text_files/"):
    _, ners, rels = read_annotation_brat(ann_note)
    lymph_dict = {}
    
    for idx, ner in enumerate(ners, start = 1):
        if ner[1] == "lymph":
            note_id = str(ann_note).split("/")[-1].split(".")[0]
            lymph_dict["T" + str(idx)] = lymph_row()
            lymph_dict["T" + str(idx)]["note_id"] = note_id
            lymph_dict["T" + str(idx)]["concept_cat"] = ner[1]
            lymph_dict["T" + str(idx)]["concept_value"] = ner[0]
            lymph_dict["T" + str(idx)]["context_start"] = ner[2][0]
            lymph_dict["T" + str(idx)]["context_end"] = ner[2][1]
            lymph_dict["T" + str(idx)]["context"] = ""
    
    for idx, rel in enumerate(rels):
        _,e1,e2 = rel
        if e1 in lymph_dict:
            pos_in_ners = int(e2[1:]) - 1
            prop_in_ners = ners[pos_in_ners][1]
            if lymph_dict[e1].get(prop_in_ners,0):
                lymph_dict[e1][prop_in_ners] = lymph_dict[e1][prop_in_ners] + ", " + ners[pos_in_ners][0]
            else:
                lymph_dict[e1][prop_in_ners] = ners[pos_in_ners][0]
            lymph_dict[e1]["context_start"] = min(lymph_dict[e1]["context_start"], ners[pos_in_ners][2][0])
            lymph_dict[e1]["context_end"] = max(lymph_dict[e1]["context_end"], ners[pos_in_ners][2][1])
            note_text = read_note(FILE_DIR + str(note_id) + ".txt")
            lymph_dict[e1]["context"] = note_text[lymph_dict[e1]["context_start"] : lymph_dict[e1]["context_end"]]
    return list(lymph_dict.values())