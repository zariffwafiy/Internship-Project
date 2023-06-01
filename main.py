import pandas as pd
from datetime import datetime 

def add_date_column(file_path, current_date):
    #read file 
    df = pd.read_csv(file_path)

    if "Date" not in df.columns:
        #parse existing data and have it in suitable format
        df["Date"] = current_date
        print(f"Date column added to {file_path}")

        df.to_csv(file_path, index=False)
    else: 
        print(f"Date column already exists in {file_path}")

#file1 = input("Provide file1 path : ")
#file2 = input("Provide file2 path : ")
file1 = "Book1.csv"
file2 = "Book2.csv"
#cur_date = input("Provide the current date (dd/mm/yyyy) : ")
cur_date = "18/05/2023"
date_format = "%d/%m/%Y"
cur_date = datetime.strptime(cur_date, date_format).date()

#clean book2
def clean_book2(file_path):
    df = pd.read_csv(file_path)

    if "Unnamed: 6" in df.columns:
        df.drop(index=df.index[2:])
        df = df.rename(columns= {"Recent Trade Statistics": "Last Trade Date", "Unnamed: 6": "No. of Trades", "Unnamed: 7": "Turnover Ratio", 
                                "Trade Liquidity Score": "Trade Recency", "Unnamed: 9" : "Trade Frequency", "Unnamed: 10": "Trade Turnover"})
        df.to_csv(file_path, index=False)
        print(f"{file_path} cleaned")
    else:
        print(f"{file_path} already cleaned")

add_date_column(file1, cur_date)
add_date_column(file2, cur_date)
clean_book2(file2)

def join_df(file1, file2): 
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    merged_df = pd.merge(df1, df2, how = "inner", on = ["Date"])
    return merged_df

merged_df = join_df(file1, file2)
print(merged_df.head())
