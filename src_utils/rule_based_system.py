# Author: Aman Pathak


# Libraries:

import numpy as np
import pandas as pd



## clean_thyroid_RE_results():
#     1. Add year information 
#     2. Add size column in thyroid from size_numeric and size_qualitative
#     3a. Fix echogenicity_ and echogenicity columns in thyroid
#     3b. Fix location_ and location columns in thyroid
#     4. Filter the aggregation results
#     5. Process TIRADS columns
#     6. Add TIRADS_is_con 
#     7. Handling size




def add_year(input_df):
    metadata_df = pd.read_csv("/home/aman.pathak/projects/2022_07_20_Thyroid_Nodule_AI/src/outputs/Thyroid_Nodule/note_ids_with_encounter_date.csv")
    metadata_df["EncounterDate"] = pd.to_datetime(metadata_df["EncounterDate"])
    metadata_df["NoteYear"] = metadata_df["EncounterDate"].dt.year
    
    input_df_copy = input_df.copy(deep= True)
    input_df_copy = input_df_copy.merge(metadata_df[["EncounterDate", "NoteYear","OrderKey"]], left_on="note_id", right_on = "OrderKey").iloc[:,:-1]
    
    return input_df_copy

def add_size(input_df):
    input_df_copy = input_df.copy(deep= True)
    input_df_copy["Size"] = np.where(input_df_copy[["size_numeric", "size_qualitative"]].notna().any(axis = 1), True, np.nan)
    
    return input_df_copy
    
def deal_echogenicity(input_df):
    input_df_copy = input_df.copy(deep= True)
    
    def merge_echogenicity(row):
        if pd.isnull(row['echogenicity_']):
            return row['echogenicity']
        elif pd.isnull(row['echogenicity']):
            return row['echogenicity_']
        elif row['echogenicity'] == row['echogenicity_']:
            return row['echogenicity']
        else:
            return row['echogenicity'] + ',' + row['echogenicity_']
    
    # Apply the function to create a new column 'merged_name'
    input_df_copy['echogenicity'] = input_df_copy.apply(merge_echogenicity, axis=1)
    input_df_copy = input_df_copy.drop("echogenicity_", axis = 1)

    return input_df_copy

def deal_location(input_df):
    input_df_copy = input_df.copy(deep= True)
    
    def merge_location(row):
        if pd.isnull(row['location_']):
            return row['location']
        elif pd.isnull(row['location']):
            return row['location_']
        elif row['location'] == row['location_']:
            return row['echogenicity']
        else:
            return row['location'] + ',' + row['location_']
    
    # Apply the function to create a new column 'merged_name'
    input_df_copy['location'] = input_df_copy.apply(merge_location, axis=1)
    input_df_copy = input_df_copy.drop("location_", axis = 1)

    return input_df_copy


# REVISE
def remove_thyroid_nans(input_df):
    input_df_copy = input_df.copy(deep= True)
    thyroid_columns = ["size_numeric", "shape", 'composition', 'echogenic_foci', 'vascularity', 'margins', 'size_qualitative']
    mask_1 = ~input_df_copy["concept_value"].isna()
    mask_2 = ~input_df_copy[thyroid_columns].isna().all(axis = 1)
    
    clean_df = input_df_copy[(mask_1) & (mask_2)]
    
    return clean_df

def process_TIRADS(input_df):
    input_df_copy = input_df.copy(deep= True)
    import re

    def preprocess_category(category):
        if pd.isna(category) or category == "":
            return np.nan

        category = category.strip()  # Remove leading/trailing whitespace

        # Handle variations with parentheses, "TI-RADS", and extra text
        match = re.match(r"^(?:TI-RADS\s?)?(TR?\d+)(?:\s*\(.*\))?$", category, re.IGNORECASE)
        if match:
            category = match.group(1)

        # Handle "4 or more" case explicitly
        category = re.sub(r"'4 or more'", "TR 4", category, flags=re.IGNORECASE)

        # Remove any remaining "TI-RADS" remnants with spaces
        category = re.sub(r"\s+TI-RADS", "", category)

        # If still not a valid TR category, assume standalone digit is TR
        if not category.startswith("TR"):
            category = f"TR {category}"

        # Convert to uppercase for consistency
        category = category.upper()

        # Ensure consistent spacing (e.g., "TR4" becomes "TR 4")
        category = re.sub(r"TR(\d+)", r"TR \1", category)
        category = re.sub(r"\s+TI-RADS", "", category)

        return category
    
    input_df_copy["TIRADS_risk_category_processed"] = input_df_copy["TIRADS_risk_category"].apply(preprocess_category)
    input_df_copy.loc[input_df_copy[input_df_copy["TIRADS_risk_category_processed"] == "TR 4)"].index, "TIRADS_risk_category_processed"] = "TR 4"
    input_df_copy.loc[input_df_copy[input_df_copy["TIRADS_risk_category_processed"] == "TR 3)"].index, "TIRADS_risk_category_processed"] = "TR 3"
    input_df_copy.loc[input_df_copy[input_df_copy["TIRADS_risk_category_processed"] == "TR 7 ("].index, "TIRADS_risk_category_processed"] = "TR 7"
    input_df_copy.loc[input_df_copy[input_df_copy["TIRADS_risk_category_processed"] == "TR 3 (MILDLY SUSPICIOUS)"].index, "TIRADS_risk_category_processed"] = "TR 3"
    input_df_copy.loc[input_df_copy[input_df_copy["TIRADS_risk_category_processed"] == "TR (TI-RADS 4)"].index, "TIRADS_risk_category_processed"] = "TR 4"
    input_df_copy.loc[input_df_copy[input_df_copy["TIRADS_risk_category_processed"] == "TR 2 (SCORE 6-7)"].index, "TIRADS_risk_category_processed"] = "TR 2"
    
    
    return input_df_copy
    
def add_TIRADS_is_con(input_df):
    input_df_copy = input_df.copy(deep= True)
    def is_consistent(row):
        category = row["TIRADS_risk_category_processed"]
        score = row["TIRADS_Score"]

        if pd.isna(category) or pd.isna(score):
            return np.nan

        try:
            score = int(score)
        except Exception as e:
            print(category, score)
            if score == "seven": score = 7
            elif score == "six": score = 6
            elif score == "Three": score = 3
            elif score == "Four": score = 4
            elif score == "None" or score == "": score = 0
            else: 
                score = int(score[0])

            


        if category == "TR 1" and score == 0:
            return True
        elif category == "TR 2" and score == 2:
            return True
        elif category == "TR 3" and score == 3:
            return True
        elif category == "TR 4" and score in (4, 5, 6):
            return True
        elif category == "TR 5" and score >= 7:
            return True
        else:
            return False
    input_df_copy_tirads = input_df_copy[input_df_copy[["TIRADS_risk_category_processed", "TIRADS_Score"]].notna().any(axis = 1)][["TIRADS_risk_category_processed", "TIRADS_Score", "note_id", "id"]].copy(deep = True)    
    input_df_copy["TIRADS_is_con"] = input_df_copy_tirads.apply(is_consistent, axis = 1)
    
    return input_df_copy



def clean_thyroid_RE(thyroid_df):
    thyroid_df_copy = thyroid_df.copy(deep = True)
    # thyroid_df_copy = add_year(thyroid_df_copy)
    thyroid_df_copy = add_size(thyroid_df_copy)
    thyroid_df_copy = deal_echogenicity(thyroid_df_copy)
    thyroid_df_copy = deal_location(thyroid_df_copy)
    thyroid_df_copy = remove_thyroid_nans(thyroid_df_copy)
    thyroid_df_copy = process_TIRADS(thyroid_df_copy)
    thyroid_df_copy = add_TIRADS_is_con(thyroid_df_copy)
    print("TODO: MERGE LOCATION")
    return thyroid_df_copy


# Clean Lymph :
# 1. Remove nans
def remove_lymph_nans(input_df):
    
    input_df_copy = input_df.copy(deep= True)
    
    lymph_columns = ['echogenicity_', 'Echogenic_hilium', 'size_numeric', 'shape', 'location_', 'size_qualitative', 'laterality', 'Risk_description']
    mask_1 = ~input_df_copy["concept_value"].isna()
    mask_2 = ~input_df_copy[lymph_columns].isna().all(axis = 1)
    
    clean_df = input_df_copy[(mask_1) & (mask_2)]

    return clean_df


def clean_lymph_RE(lymph_df):
    lymph_df_copy = lymph_df.copy(deep = True)
    # lymph_df_copy =  add_year(lymph_df_copy)
    lymph_df_copy = remove_lymph_nans(lymph_df_copy)
    return lymph_df_copy



# Handling size



def measurement_dict():
    dataframe_dict = {
        "note_id": "",
        "index":-1,
        "size_numeric_dim_01":np.nan,
        "size_numeric_dim_02":np.nan,
        "size_numeric_dim_03":np.nan,
        "unit":"",
        "id":""
    }
    return dataframe_dict

def convert_unit(unit):
    unit = unit.lower()
    unit_table = {
        "cm":"cm",
        "mm":"mm",
        "centimeter": "cm",
        "cms":"cm",
        "cc":"cm",
        "centimeters":"cm",
        "millimeters": "mm",
        "millimeter":"mm",
        "m": "m",
        "meter" : "m",
        "meters":"m"
    }
    return unit_table[unit]


def identify_unit_dimensions(input_df):
    input_df_copy = input_df.copy(deep = True)
    
    to_review = []
    value_results = []
    unit_results = []
    note_id_results = []
    index = []
    no_units = []
    thy_idx = []
    poss_units = ["cm", "mm", "millimeters", "centimeters", "centimeter", "cms", "meter", "m", "millimeter", "cc", "meters"]
    
    for rows in input_df.to_dict("records"):
        x = rows["size_numeric"].replace("x"," ").strip().split()

        if len(x) > 0:
            # target subcentimeter and less than, 5 x 4 x 4 mm -> 5 x 4 x 4 mm
            if x[0].startswith('sub') or x[0].startswith('less') :
                x = x[1:]

            # Check units?
            if len(x) == 0 or not (x[-1].lower() in poss_units or x[-1][-2:] in poss_units):
                # No units present
                no_units.append(rows)
        #             continue
            else:
                unit = x[-1]
                # Process units if necessary
                if x[-1][-2:] in poss_units:

                    unit = x[-1][-2:]

                value_dim = []
                to_review_flag = False
                for value in x[:-1]:
                    try:
                        value = value.replace(",","")
                        if not value in ["x", "cm", "mm", 'subcentimeter', 'Subcentimeter', 'subcentimeters']:
                            value = value.replace("cm", "")
                            value = value.replace("mm", "")
                            value_dim.append(float(value))
                    except Exception as e:
        #                 print(e)
                        to_review.append(rows)
        #                 print(x[-1], value, rows, len(to_review))
                        break
                value_results.append(value_dim)
                unit_results.append(unit)
                note_id_results.append(rows["note_id"])
                index.append(rows["index"])
                thy_idx.append(rows["id"])
    return value_results, unit_results, note_id_results, index, thy_idx, to_review, no_units



def get_exploded_df_part_1(value_results, unit_results, note_id_results, index, thy_idx):
    dataframe_list = []
    for values, unit, note_id, idx, thy_id  in zip(value_results, unit_results, note_id_results, index, thy_idx):
        dimensions = len(values)
        dataframe_dict = measurement_dict()
        dataframe_dict["note_id"] = note_id
        dataframe_dict["index"] = idx    
        dataframe_dict["id"] = thy_id
        if not dimensions == 0:
            dataframe_dict["size_numeric_dim_01"] = values[0]
            if dimensions > 1:
                dataframe_dict["size_numeric_dim_02"] = values[1]
            if dimensions > 2:
                dataframe_dict["size_numeric_dim_03"] = values[2]
            dataframe_dict["unit"] = convert_unit(unit)
        dataframe_list.append(dataframe_dict)
    exploded_df = pd.DataFrame(dataframe_list)
    return exploded_df
    

def clean(x):
    dataframe_dict = measurement_dict()
    size_numeric = x["size_numeric"]
#     dataframe_dict["size_numeric"] = size_numeric
    size_numeric = size_numeric.replace("-", " ")
    size_numeric = size_numeric.replace("x", " ")
    size_numeric = size_numeric.replace("subcentimeter,","")
    
    
    dataframe_dict["note_id"] = x["note_id"]
    dataframe_dict["index"] = x["index"]
    dataframe_dict["id"] = x["id"]
    
    
    
    if "to" in size_numeric or "and" in size_numeric:
        size_numeric = size_numeric.replace("and", '?')
        size_numeric = size_numeric.replace("to", '?')
        size_numeric_split = size_numeric.split(" ? ")
        try:
            left = size_numeric_split[0]
            left = left.replace("mm","")
            left = float(left.replace("cm",""))
        except:
#             print(size_numeric_split)
            return dataframe_dict
        if len(size_numeric_split) > 1:
            try:
                right, unit = size_numeric_split[1].split()
                dataframe_dict["unit"] = unit
                right = right.replace("mm","")
                right = float(right.replace("cm",""))
                
                left = (left+right)/2
            except:
                pass

        dataframe_dict["size_numeric_dim_01"] = round(left,2)
    
    return dataframe_dict
 

def get_part_2_no_units(no_units, to_review):

    # If no units and to_review are empty, return empty DataFrames
    if not no_units and not to_review:
        return pd.DataFrame(), pd.DataFrame()
    else:
        print("No units: ", len(no_units), "To review: ", len(to_review))

    no_units_df = pd.DataFrame(no_units)
    no_units_df["to_drop"] = True
    
    to_review_df = pd.DataFrame(to_review)
    to_review_df["unit"] = ""

    to_review_df.loc[to_review_df["size_numeric"].str.contains("mm"), "unit"] = "mm"
    to_review_df.loc[to_review_df["size_numeric"].str.contains("cm"), "unit"] = "cm"
    
    to_review_solved = []
    for row in to_review_df.to_dict("records"):
        to_review_solved.append(clean(row))
    
    to_review_solved_df = pd.DataFrame(to_review_solved)
    return to_review_solved_df, no_units_df



def handle_size_columns(thyroid_df):
    thyroid_df_copy = thyroid_df.copy(deep = True)
    temp_df = thyroid_df_copy[["size_numeric", "note_id", "id"]].reset_index(drop=True).reset_index().copy(deep = True)
    print(temp_df.shape)
    temp_df = temp_df[temp_df["size_numeric"].notna()]
    print(temp_df.shape)
    # Remove subcentimeters:
    subcentimeters_words = ['subcentimeter', 'Subcentimeter', 'subcentimeters', 'subcm', "less than a centimeter", "less than 1 cm",
                        'less than one centimeter', "less than, 1 cm", 'less than, one centimeter', "less"]
    temp_df = temp_df[~temp_df["size_numeric"].isin(subcentimeters_words)]
    
#     print(temp_df.shape)
    value_results, unit_results, note_id_results, index, thy_idx, to_review, no_units = identify_unit_dimensions(temp_df)
    
    exploded_df = get_exploded_df_part_1(value_results, unit_results, note_id_results, index, thy_idx)
    print("Exploded df", exploded_df.shape)

    # If exploded_df is not empty, proceed with merging
    if not exploded_df.empty:
        df_size = temp_df.merge(exploded_df, on=["note_id", "index", "id"], how = "left").copy(deep = True)
        print(df_size.shape)
        print(df_size.columns)
        
        to_review_solved_df, no_units_df = get_part_2_no_units(no_units, to_review)

        # If to_review_solved_df is not empty, proceed with merging
        if not to_review_solved_df.empty:
            df_size = df_size.merge(to_review_solved_df, on=["note_id", "index", "size_numeric_dim_01","size_numeric_dim_02", "size_numeric_dim_03", "unit","id"], how = "left")
            print(df_size.shape)

        # If no_units_df is not empty, proceed with merging
        if not no_units_df.empty:
            df_size = df_size.merge(no_units_df, on = ['index', 'size_numeric', 'note_id', 'id'], how = "left")
            print(df_size.shape)
    

        if "to_drop" not in df_size.columns:
            df_size["to_drop"] = False

        thyroid_df_copy_final = thyroid_df_copy.merge(df_size[["note_id","id","size_numeric_dim_01","size_numeric_dim_02","size_numeric_dim_03","unit","to_drop"]], on = ["note_id","id"], how = "left")

        thyroid_df_copy_final = thyroid_df_copy_final[~thyroid_df_copy_final["size_numeric"].isin(subcentimeters_words)]
        print(thyroid_df_copy_final.shape)
        thyroid_df_copy_final = thyroid_df_copy_final.drop(thyroid_df_copy_final[thyroid_df_copy_final["to_drop"] == True].index)
        thyroid_df_copy_final.drop("to_drop", axis = 1, inplace = True)
        return thyroid_df_copy_final

    else:
        print("No valid size_numeric found in the DataFrame.")
        return thyroid_df_copy


    
def drop_below_measurement(measurement_in_cm, thyroid_df, lymph_df = None):
    # Reject nodule where size of all the dimensions is less than the measurement_in_cm or accept if any dimension is greater than the measurement_in_cm
    measurement_in_mm = measurement_in_cm * 10
    thyroid_df_copy = thyroid_df.copy(deep=True)
    lymph_df_copy = lymph_df.copy(deep=True)
    thyroid_df_copy["to_drop"] = True
    
    mm_df = thyroid_df_copy[thyroid_df_copy["unit"] == "mm"]
    print("Notes with mm unit: ",mm_df.shape)
    cutoff = measurement_in_mm
    print("cutoff in mm", cutoff)
    mm_df_filtered = mm_df[(mm_df["size_numeric_dim_01"] > cutoff) | (mm_df["size_numeric_dim_02"] > cutoff) | (mm_df["size_numeric_dim_03"] > cutoff)]
    print("Notes with size > {0} mm: ".format(measurement_in_mm) ,mm_df_filtered.shape)
    thyroid_df_copy.loc[mm_df_filtered.index, "to_drop"] = False
    
    # unit = "cm"
    cm_df = thyroid_df_copy[thyroid_df_copy["unit"] == "cm"]
    print("Notes with cm unit: ",cm_df.shape)
    cutoff = measurement_in_cm
    print("cutoff in cm", cutoff)
    cm_df_filtered = cm_df[(cm_df["size_numeric_dim_01"] > cutoff) | (cm_df["size_numeric_dim_02"] > cutoff) | (cm_df["size_numeric_dim_03"] > cutoff)]
    print("Notes with size > 1 cm: ",cm_df_filtered.shape)
    thyroid_df_copy.loc[cm_df_filtered.index, "to_drop"] = False
    
    
    print("Total nodules to be dropped ",thyroid_df_copy["to_drop"].sum())
    thyroid_df_copy = thyroid_df_copy.drop(thyroid_df_copy[thyroid_df_copy["to_drop"] == True].index)

    print("Final nodules: ", thyroid_df_copy.shape)
    print("Unique notes: ", thyroid_df_copy["note_id"].nunique())
    

    print("Lymph notes before: ", lymph_df_copy.shape)
    note_ids = thyroid_df_copy["note_id"].unique()
    lymph_df_copy = lymph_df_copy[lymph_df_copy["note_id"].isin(note_ids)]
    print("Lymph notes after: ", lymph_df_copy.shape)

    
    thyroid_df_copy.drop("to_drop", axis = 1, inplace = True)

    return thyroid_df_copy, lymph_df_copy

    

def filtering_size(measurement_in_cm, thyroid_df, lymph_df, cleaned_already = False):
    thyroid_df_copy = thyroid_df.copy(deep = True)
    lymph_df_copy = lymph_df.copy(deep = True)
    
    if not cleaned_already:
        thyroid_df_copy = clean_thyroid_RE(thyroid_df_copy)
        print("UNCOMMENT THIS IF YOU WANT TO CLEAN LYMPH")
        # lymph_df_copy = clean_lymph_RE(lymph_df_copy)
    
    thyroid_df_size = handle_size_columns(thyroid_df_copy)
    thyroid_df_copy, lymph_df_copy = drop_below_measurement(measurement_in_cm, thyroid_df_size, lymph_df_copy)
    
    return thyroid_df_copy, lymph_df_copy
    



def filtering_year(thyroid_df, lymph_df, year, cleaned_already = False):
    thyroid_df_copy = thyroid_df.copy(deep = True)
    lymph_df_copy = lymph_df.copy(deep = True)
    
    if not cleaned_already:
        thyroid_df_copy = clean_thyroid_RE(thyroid_df_copy)
        lymph_df_copy = clean_lymph_RE(lymph_df_copy)
    
    thyroid_df_copy = thyroid_df_copy[thyroid_df_copy["NoteYear"] >= year]
    lymph_df_copy = lymph_df_copy[lymph_df_copy["NoteYear"] >= year]

    return thyroid_df_copy, lymph_df_copy


if __name__ == "__main__":
    lymph_df = pd.read_csv("../outputs/Thyroid_Nodule/Aggregation_lymph_result_1_8.csv")
    thyroid_df = pd.read_csv("../outputs/Thyroid_Nodule/Aggregation_result_1_8.csv")
    
    print("clean_thyroid_RE\n", clean_thyroid_RE(thyroid_df))
    print("clean_lymph_RE\n", clean_lymph_RE(lymph_df))
    
    
    print("filtering_size\n", filtering_size(0, thyroid_df, lymph_df))
    # print("filtering_year\n", filtering_year(thyroid_df, lymph_df,2018))