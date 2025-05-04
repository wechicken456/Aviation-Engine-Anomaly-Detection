#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd


# In[10]:


dfs = pd.DataFrame()
for i in range(1, 2):
    df = pd.read_excel(f"./data/flight_data_batch{i}.xlsx")
    df = df.iloc[::5]
    df.insert(len(df.columns.tolist()), "is_fail_point", 0)
    df.insert(len(df.columns.tolist()), "is_fail_left", 0)
    df.insert(len(df.columns.tolist()), "is_fail_right", 0)
    fail_df = pd.read_csv(f"./data/Failures/Failure_Events_in_Batch_{i}.csv")
    
    # Get set of flight_ids from fail_df
    fail_flight_ids = set(fail_df["flight_id"])
    
    # Process each flight_id in fail_df
    """
    Have to be careful in indexing: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#why-does-assignment-fail-when-using-chained-indexing
    """
    for flight_id in fail_flight_ids:
        # Get indices of rows for this flight_id
        flight_indices = df[df["flight_id"] == flight_id].index
        
        # Sort these rows by time 
        flight_subset = df.loc[flight_indices].sort_values(by="time", kind="stable")
        
        # Update df with sorted rows
        df.loc[flight_indices] = flight_subset
        
        # Compute differences in rpm_right for consecutive rows
        rpm_right_diff = flight_subset["rpm_right"].diff()
        rpm_left_diff = flight_subset["rpm_left"].diff()
        
        # Identify rows where rpms are 1000 less than previous row
        # diff < -1000 means current - previous < -1000, i.e., current < previous - 1000
        condition_left = (rpm_left_diff <  -1000)
        condition_right = (rpm_right_diff < -1000)
        
        # Get indices of rows meeting the condition 
        failure_index_left = flight_subset.index[condition_left]
        matching_indices_right = flight_subset.index[condition_right]

        if not failure_index_left.empty:
            all_fail_indices = flight_subset.index[flight_subset.index >= failure_index_left]
            df.loc[all_fail_indices, "is_fail_left"] = True
            df.loc[failure_index_left, "
        print(failure_index_left, matching_indices_right)
        # Set is_fail to True for matching rows in original df
        df.loc[failure_index_left, "is_fail_left"] = 1
        df.loc[matching_indices_right, "is_fail_right"] = 1
    
    dfs = pd.concat([dfs, df], ignore_index=True)


# In[4]:


for i in range(1, 4):
    df = pd.read_excel(f"./data/flight_data_batch6_part{i}.xlsx")
    df = df.iloc[::5]
    df.insert(len(df.columns.tolist()), "is_fail_left", 0)
    df.insert(len(df.columns.tolist()), "is_fail_right", 0)
    fail_df = pd.read_csv(f"./data/Failures/Failure_Events_in_Batch_6_Part_{i}.csv")
    
    # Get set of flight_ids from fail_df
    fail_flight_ids = set(fail_df["flight_id"])
    
    # Process each flight_id in fail_df
    """
    Have to be careful in indexing: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#why-does-assignment-fail-when-using-chained-indexing
    """
    for flight_id in fail_flight_ids:
        # Get indices of rows for this flight_id
        flight_indices = df[df["flight_id"] == flight_id].index
        
        # Sort these rows by time 
        flight_subset = df.loc[flight_indices].sort_values(by="time", kind="stable")
        
        # Update df with sorted rows
        df.loc[flight_indices] = flight_subset
        
        # Compute differences in rpm_right for consecutive rows
        rpm_right_diff = flight_subset["rpm_right"].diff()
        rpm_left_diff = flight_subset["rpm_left"].diff()
        
        # Identify rows where rpms are 1000 less than previous row
        # diff < -1000 means current - previous < -1000, i.e., current < previous - 1000
        condition_left = (rpm_left_diff <  -1000)
        condition_right = (rpm_right_diff < -1000)
        
        # Get indices of rows meeting the condition 
        failure_index_left = flight_subset.index[condition_left]
        matching_indices_right = flight_subset.index[condition_right]
        # Set is_fail to True for matching rows in original df
        df.loc[failure_index_left, "is_fail_left"] = 1
        df.loc[matching_indices_right, "is_fail_right"] = 1
    
    dfs = pd.concat([dfs, df], ignore_index=True)


# In[ ]:


dfs.to_csv("./data/combined_correct_datasets.csv", index=False) # don't add another column for indices


# In[ ]:


print(sum(dfs["is_fail"] == 1))


# In[ ]:


print(dfs)


# In[ ]:




