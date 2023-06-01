print("alt main internship project")

import os
import pandas as pd

file_path_BIR = "20230515 - BIR.xlsx"
file_path_CLS = "20230515 - CLS.xlsx"
csv_file_path_BIR = "20230515 - BIR.csv"
csv_file_path_CLS = "20230515 - CLS.csv"

if not os.path.exists(csv_file_path_BIR):
    excel_BIR = pd.read_excel(file_path_BIR, sheet_name= "Data")
    excel_BIR.to_csv(csv_file_path_BIR, encoding="utf-8", index = False)

if not os.path.exists(csv_file_path_CLS):
    excel_CLS = pd.read_excel(file_path_CLS, sheet_name= "Data")
    excel_CLS.to_csv(csv_file_path_CLS, encoding= "utf-8", index = False)

df_BIR = pd.read_csv(csv_file_path_BIR)
df_CLS = pd.read_csv(csv_file_path_CLS, skiprows=1).iloc[:-4, :].rename(
    columns= {
        "Unnamed: 0" : "Stock Name",
        "Unnamed: 1" : "Stock Code",
        "Unnamed: 2" : "Issuer Name", 
        "Unnamed: 3" : "Facility Name",
        "Unnamed: 4" : "Outstanding Amount",
        "Unnamed: 11" : "Composite Liquidity Score",
        "Unnamed: 12" : "Date"
    }
)

#print(df_CLS)
#print(list(df_CLS.columns))

df_merged = pd.merge(df_BIR, df_CLS, how="inner", on = ["Stock Code", "Date"]).rename(columns={"Stock Name_x" : "Stock Name"})
df_merged = df_merged.drop(columns={"Stock Name_y"})
print(df_merged)
print(df_merged.shape)
print(list(df_merged.columns))