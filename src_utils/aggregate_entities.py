# Author: Aman Pathak


# Libraries:

import numpy as np
import pandas as pd
from rule_based_system import filtering_size, filtering_year, clean_thyroid_RE, clean_lymph_RE

# Size
def convert_size_to_cm(row):
    if pd.isnull(row['unit']):
        return row
    elif row["unit"] == "mm":
        row["size_numeric_dim_01"], row["size_numeric_dim_02"], row["size_numeric_dim_03"] = 10 * row["size_numeric_dim_01"], 10 * row["size_numeric_dim_02"],10 * row["size_numeric_dim_03"]
        row["unit"] = "cm"    
    return row

# Shape

def aggregate_shape(shape):
    
    if pd.isna(shape):
        return "Unknown", 0
    
    shape_lower = shape.lower()  # Normalize to lower case for easier matching
    if 'wider' in shape_lower and 'tall' in shape_lower and not 'taller' in shape_lower:
        return "Wider than tall", 0
    elif 'taller' in shape_lower and 'wide' in shape_lower:
        return "Taller than wide", 3
    else:
        return "OTHER", 0


# Location:

def merge_location(row):
    if pd.isnull(row['location_']):
        return row['location']
    elif pd.isnull(row['location']):
        return row['location_']
    elif row['location'] == row['location_']:
        return row['location']
    else:
        return row['location'] + ',' + row['location']

def process_location(row):

    location_dict = {
    "upper": ["upper", "superior"],
    "mid": ["mid", "middle", "interpolar"],
    "lower": ["lower", "inferior"]
    }
    location = row["location"]

    if pd.isnull(location):
        return "Unknown"

    location = location.lower()
    for category, keywords in location_dict.items():
        if any(keyword in location for keyword in keywords):
            return category
    return "OTHER"

# Laterality
def process_laterality(laterality):
    laterality_dict = {
       "isthmus" : ["isthmus", "the isthmus", 'the thyroid isthmus', 'isthmic'],
       "left_lobe" : ["left", "left lobe", "left thyroid lobe", "the left", "the left thyroid gland"],
       "right_lobe" : ["right", "right lobe", "right thyroid lobe", "the right", "the right thyroid gland"],
       'bilateral' : ["both lobes", "both", "bilateral", "bilaterally", "lobes"]
    }

    if pd.isnull(laterality):
        return 'Unknown'
    else:
        laterality = laterality.lower()
        if any([string in laterality for string in laterality_dict["isthmus"]]):
            
            return "Isthmus"
        elif any([string in laterality for string in laterality_dict["bilateral"]]):
            return "Bilateral"
        elif any([string in laterality for string in laterality_dict["left_lobe"]]):
            return "Left Lobe"
        elif any([string in laterality for string in laterality_dict["right_lobe"]]):
            return "Right Lobe"
        else:
            return "OTHER"


# Composition
def process_composition(composition):
    if pd.isna(composition) or composition == "NaN":
        return "Unknown", 0

    composition_lower = composition.lower()
    
    if 'cystic or completely cystic' in composition_lower or 'cystic' in composition_lower:
        if 'almost completely solid' in composition_lower or 'solid' in composition_lower:
            return "mixed cystic and solid", 1
        elif 'almost completely' not in composition_lower and 'solid' not in composition_lower:
            return "cystic or completely cystic", 0
    elif 'solid or almost completely solid' in composition_lower or 'solid' in composition_lower:
        return "solid or almost completely solid", 2
    elif 'spongiform' in composition_lower:
        return "spongiform", 0
    elif 'mixed' in composition_lower or 'complex' in composition_lower or 'heterogeneous' in composition_lower or 'heterogenous' in composition_lower:
        return "mixed cystic and solid", 1
    else:
        return "OTHER", 0



# Margin
margin_dict = {
    "extra_thyroidal_extension": ["extra-thyroidal extension", "extra-thyroidal", "extra-", "3"],
    "ill-defined": ["ill-defined", "ill defined", "poorly defined", "indistinct", "vague", "poor"],
    "lobulated_or_irregular": ["lobulated", "irregular", "sharp"],
    "smooth" : ["absent","no"],
    "smooth": ["smooth", "circumscribed", "well-", "well", "well-defined", "well defined", "well demarcated", "well marginated", "defined"]
}

def process_margin(margin):
    if pd.isna(margin):  # Check for NaN values explicitly
        return ["Unknown", 0]
    
    margin_lower = margin.lower()  # Normalize to lower case for easier matching
    
    for category, keywords in margin_dict.items():
        if any(keyword in margin_lower for keyword in keywords):
            if category == "extra_thyroidal_extension":
                return [category, 3]
            elif category == "lobulated_or_irregular":
                return [category, 2]
            elif category in ["ill-defined","smooth", "absent"]:
                return [category, 0]
    return ["OTHER", 0]


# Echogenicity

echogenicity_dict = {
    "Very hypoechoic": ["very hypoechoic", "3"],
    "Hyperechoic or isoechoic": ["1","low", "hyperechoic", "isoechoic", "hyperechoic or isoechoic", "isoechoic to hyperechoic", "hyperechoic to isoechoic", "slightly hyperechoic", "predominantly isoechoic", "mildly hyperechoic", "isointense"],
    "Anechoic": ["anechoic", "0"],
    "Hypoechoic": ["2","hypoechoic", "very hypoechoic", "mildly hypoechoic", "slightly hypoechoic", "minimally hypoechoic", "hypodense", "predominantly hypoechoic"],
    }


def process_echogenicity(echogenicity):
    if pd.isna(echogenicity):  # Check for NaN values explicitly
        return ["Unknown", 0]
    
    echogenicity_lower = echogenicity.lower()  # Normalize to lower case for easier matching
    
    for category, keywords in echogenicity_dict.items():
        if any(keyword in echogenicity_lower for keyword in keywords):
            if category == "Anechoic":
                return [category, 0]
            elif category == "Hyperechoic or isoechoic":
                return [category, 1]
            elif category == "Hypoechoic":
                return [category, 2]
            elif category == "Very hypoechoic":
                return [category, 3]
            
    return ["OTHER", 0]



# Echogenic Foci
echogenic_foci_dict = {
    "Punctate echogenic foci": ["punctate echogenic foci", "punctate echogenic foci (3)", "multiple punctate echogenic foci", "single punctate calcification", "punctate calcifications"],
    "None or absent or without or Large comet tail artifacts": ["none", "no", "without", "large comet-tail artifacts", "none or large comet-tail artifacts (0)", "none (0)","microcalcifications"],
    "Macrocalcifications": ["macrocalcification", "macrocalcifications", "macrocalcifications (1)", "macro calcification", "coarse calcifications"],
    "Peripheral rim calcifications": ["peripheral", "rim", "peripheral (rim) calcifications", "peripheral calcifications", "peripheral rim calcifications", "thin peripheral calcifications"],
    }

def process_echogenic_foci(echogenic_foci):
    if pd.isna(echogenic_foci):
        return ["Unknown", 0]
    
    echogenic_foci_lower = echogenic_foci.lower()
    
    for category, keywords in echogenic_foci_dict.items():
        if any(keyword in echogenic_foci_lower for keyword in keywords):
            if category == "None or absent or without or Large comet tail artifacts":
                return [category, 0]
            elif category == "Macrocalcifications":
                return [category, 1]
            elif category == "Peripheral rim calcifications":
                return [category, 2]
            elif category == "Punctate echogenic foci":
                return [category, 3]
    return ["OTHER", 0]

# Vascularity
vascularity_dict = {
    "no_vascularity": [
        "no", "No", "no flow", "no increased flow", "no internal flow", 
        "no hypervascularity", "no significant flow", "without internal vascularity", 
        "No increased vascularity", "no associated color Doppler flow", 
        "no internal Doppler flow", "no abnormal vascular flow", "definite, No", "NO"
    ],
    "mild_vascularity": [
        "Mildly hypervascular", "Mild increased vascularity", "Mild vascularity", 
        "mild increased vascularity", "Mild peripheral vascularity", "Minimal", 
        "Mildly", "Mildly hyperemic", "Mild hyperemia", "Slight", 
        "Slightly", "mild increase, vascularity", "mild hypervascularity", "Minimally increased"
    ],
    "increased_hypervascular": [
        "Hyperemic rim", "Hyperemic", "hypervascular", "increased vascularity", 
        "increased vascular flow", "hypervascularity", "Increased vascularity", 
        "increased color flow vasculature", "hypervascular components", 
        "prominent internal vascular flow"
    ],
    "peripheral_vascularity": [
        "Peripheral vascularity", "peripheral hypervascularity", "Mostly peripheral vascularity", 
        "increased peripheral vascularity", "peripheral vascular flow", "peripheral flow", 
        "Peripherally", "Peripheral hypervascularity", "peripherally increased vascularity", 
        "Hypervascular peripheral rim"
    ],
    "internal_vascularity": [
        "internal flow", "internal vascular flow", "internal Doppler flow", 
        "with internal flow", "internal vascularity", "internal and peripheral vascular flow", 
        "prominent internal vascular flow", "Some internal vascularity"
    ],
    "mixed_ambiguous": [
        "Normal vascularity", "vascularity", "flow", "Vascular flow is noted", 
        "with internal vasculature", "with peripheral color Doppler flow", 
        "Increased vascularity, the periphery", "Surrounding vascularity", "heterogenous vascularity"
    ]
}

def process_vascularity(vascularity):
    if pd.isna(vascularity):
        return "Unknown"
    vascularity = vascularity.lower()
    
    # Check for each category in the vascularity_dict
    for category, keywords in vascularity_dict.items():
        if any(keyword.lower() in vascularity for keyword in keywords):
            return category
    return "OTHERS"


# TIRADS_Score

def process_TIRADS_Score(TIRADS_Score):
    if pd.isna(TIRADS_Score):
        return np.nan
    
    if isinstance(TIRADS_Score, str):
        if ',' in TIRADS_Score:  # Check if there are multiple TIRADS scores
            scores = TIRADS_Score.split(',')
            scores = [int(s.strip()) for s in scores]
            return int(max(scores))
        else:
            return int(TIRADS_Score.strip())
    else:
        return int(TIRADS_Score)



def process_entities(thyroid_df, lymph_df, cleaned_already = False):
        
    thyroid_df_copy = thyroid_df.copy(deep = True)
    lymph_df_copy = lymph_df.copy(deep = True)

    if not cleaned_already:
        thyroid_df_copy, lymph_df_copy = filtering_size(0,thyroid_df_copy,lymph_df_copy)

    # Process Size
    thyroid_df_copy = thyroid_df_copy.apply(convert_size_to_cm, axis = 1)

    # Process Location
    thyroid_df_copy['location'] = thyroid_df_copy.apply(merge_location, axis=1)
    thyroid_df_copy['aggregated_location'] = thyroid_df_copy.apply(process_location, axis = 1)
    thyroid_df_copy.drop("location_", axis = 1, inplace = True)

    # Process Laterality
    thyroid_df_copy['aggregated_laterality'] = thyroid_df_copy['laterality'].apply(process_laterality)

    # Process Shape
    thyroid_df_copy[['aggregated_shape', 'TIRADS_shape_pts']] = thyroid_df_copy["shape"].apply(lambda x: pd.Series(aggregate_shape(x)))

    # Process Composition
    thyroid_df_copy[['aggregated_composition', 'TIRADS_comp_pts']] = thyroid_df_copy['composition'].apply(lambda x: pd.Series(process_composition(x)))

    # Process Margin
    thyroid_df_copy[['aggregated_margins', 'TIRADS_margin_pts']] = thyroid_df_copy["margins"].apply(lambda x: pd.Series(process_margin(x)))
    
    # Process Echogenicity
    thyroid_df_copy[['aggregated_echogenicity', 'TIRADS_echogenicity_pts']] = thyroid_df_copy["echogenicity"].apply(lambda x: pd.Series(process_echogenicity(x)))

    # Process Echogenic Foci
    thyroid_df_copy[['aggregated_echogenic_foci', 'TIRADS_echogenic_foci_pts']] = thyroid_df_copy["echogenic_foci"].apply(lambda x: pd.Series(process_echogenic_foci(x)))

    # Process Vascularity
    thyroid_df_copy['aggregated_vascularity'] = thyroid_df_copy['vascularity'].apply(process_vascularity)

    # Process TIRADS_SCORE
    thyroid_df_copy['TIRADS_Score'] = thyroid_df_copy["TIRADS_Score"].apply(process_TIRADS_Score)

    # Generate TIRADS_Calculated
    thyroid_df_copy["TIRADS_Calculated"] = thyroid_df_copy[["TIRADS_shape_pts", "TIRADS_comp_pts", "TIRADS_margin_pts", "TIRADS_echogenicity_pts", "TIRADS_echogenic_foci_pts"]].sum(axis = 1)

    return thyroid_df_copy, lymph_df_copy



if __name__ == "__main__":
    lymph_df = pd.read_csv("../outputs/Thyroid_Nodule/Aggregation_lymph_result_1_8.csv")
    thyroid_df = pd.read_csv("../outputs/Thyroid_Nodule/Aggregation_result_1_8.csv")
    
    thyroid_df, lymph_df = filtering_size(1, thyroid_df, lymph_df)
    thyroid_df, lymph_df = process_entities(thyroid_df, lymph_df)