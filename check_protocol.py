# Standard library imports
from pathlib import Path

# Third party imports
import pandas as pd
import pyabf
from rich import print
from tabulate import tabulate

dirname = Path("C:\\Users\\KANG\\Desktop\\Today\\New folder")
# dirname = Path("E:/PRJ_Striatal_ACh_Dynamics/Data/2025_06_12/abfs")


# Get list of abf files in the directory
def get_abf_list(dirname):
    if not Path(dirname).exists():
        print("Directory does not exist, please try again!!")
        return []
    abf_list = sorted(Path(dirname).glob("*.abf"))
    return abf_list


# Read abf files and print out a table of filename and protocol name
def read_abfs(abf_list):
    if not abf_list:
        print("No abf files is in the directory, please try again!!")
        return [], [], [], []

    list_of_ID = []
    list_of_protocol = []
    for abf in abf_list:
        abf = pyabf.ABF(abf)
        list_of_ID.append(abf.abfID)
        list_of_protocol.append(abf.protocol)

    df_to_print = pd.DataFrame({"Filename": list_of_ID, "Protocol": list_of_protocol})
    return df_to_print


abf_list = get_abf_list(dirname)
df_to_print = read_abfs(abf_list)

print(tabulate(df_to_print, headers="keys", showindex=False, tablefmt="pretty"))
