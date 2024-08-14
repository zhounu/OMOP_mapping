# In this program we evaluate how many data in EHR is already presented in OMOP CDM
# Note that concepts presented on Athena do not necessarily have standard concept(s)
# This is an older version of our code and it might be a little bit different from the code we use in the paper

import pandas as pd
# Load OMOP datasets
pos_sample = pd.read_csv('/home/xz598/scripts/use_cdm_train_st/model_for_3_domains/pos_samples2.csv')
synonyms = pd.read_csv('/home/xz598/scripts/use_cdm_train_st/model_for_3_domains/synonyms.csv')
pos_sample = pos_sample[['concept_name_1', 'concept_name_2']]
synonyms = synonyms[['concept_name_1', 'concept_name_2']]
df = pd.concat([pos_sample, synonyms])
print(df.shape)
df2 = pd.DataFrame()
df2['col2'] = pd.concat([df['concept_name_1'].str.lower(), df['concept_name_2'].str.lower()], ignore_index=True) # remember to lowercase the sentences
print(df2.shape)
df2.drop_duplicates(inplace=True)
print(df2.shape)



# Load Yale's EHR data
df_ehr_full1 = pd.read_csv('/home/xz598/data_folder/cmp-jdat-data-folder/2356781_CarDS_Aim_1_Hosp_Enc_Dx.txt',
                 sep='\t', on_bad_lines = 'skip', low_memory = False)
df_ehr_full2 = pd.read_csv('/home/xz598/data_folder/cmp-jdat-data-folder/2356781_CarDS_Aim_1_Outpatient_Enc_Dx.txt',
                 sep='\t', on_bad_lines = 'skip', low_memory = False)
df = pd.concat([df_ehr_full1, df_ehr_full2])
# df = pd.read_csv('/home/xz598/data_folder/cmp-jdat-data-folder/2356781_CarDS_Aim_1_Meds.txt',
#                  sep='\t', on_bad_lines = 'skip', low_memory = False)
df1 = pd.DataFrame()
df1['col1'] = df['DX_NAME'].str.lower().unique() # remember to lowercase the sentences

# Data preprocessing
def data_preprocessing(text):
    return text.replace('(hc code)','').strip()
df1['col1'] = df1['col1'].map(data_preprocessing)

# Calculate the proportion of instances that are already on Athena
match_df = pd.DataFrame(columns=df1.columns)
no_match_df = pd.DataFrame(columns=df1.columns)
matches = df1['col1'].isin(df2['col2'])
match_df = pd.concat([match_df, df1[matches]])
no_match_df = pd.concat([no_match_df, df1[~matches]])

# Calculate the total number of unique concepts on Athena
total_values = len(df1['col1'])

# Count the number of matches
num_matches = matches.sum()

# Calculate the percentage of matches
percentage_matches = (num_matches / total_values) * 100

# Print the percentage
print(total_values)
print(num_matches)
print(f"Percentage of matches: {percentage_matches}%")


# Collect a sample of matches and non-matches to validate our approach
match_sample = match_df.sample(n=50, random_state = 42)
no_match_sample = no_match_df.sample(n=50, random_state = 42)
match_sample.to_csv('match_sample.csv', index=False)
no_match_sample.to_csv('no_match_sample.csv', index = False)
