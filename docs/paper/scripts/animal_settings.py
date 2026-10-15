# ruff: noqa: INP001
"""Scratch: imaging settings (rec_data.db) + animal / injection info (exp_info.db) of the 8 GACh3.0 spontaneous days."""

import sqlite3

import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 60)

rec = pd.read_excel("results/spontaneous/spontaneous_summary.xlsx", sheet_name="recordings")
rec = rec[rec["sensor"] == "GACh3.0"]
rec["date"] = rec["recording"].str[:10]
rec["filename"] = rec["recording"].str.replace("_BIEXP_ALS", "") + ".tif"
n_comp = rec.groupby("date")["n_compartments"].sum()

con = sqlite3.connect("data/rec_data.db")
rows = []
for date, files in rec.groupby("date")["filename"]:
    df = pd.read_sql(f"SELECT * FROM 'REC_{date}'", con)
    df = df[df["Filename"].isin(files)]
    row = {"date": date, "compartments": int(n_comp[date]), "n_rec": len(df)}
    for col in ("EXC", "LEVEL", "EXPO", "EMI", "FPS", "FRAMES", "SLICE", "CAM_TRIG_MODE", "NOTE"):
        if col in df:
            row[col] = ", ".join(sorted({str(v) for v in df[col] if v not in (None, "")})) or "-"
    rows.append(row)
con.close()
imaging = pd.DataFrame(rows).set_index("date")
print("=== imaging settings (only the analyzed recordings)\n", imaging.to_string())

con = sqlite3.connect("data/exp_info.db")
basic = pd.read_sql("SELECT * FROM BASIC_INFO", con)
inj = pd.read_sql("SELECT * FROM INJECTION_HISTORY", con)
con.close()
basic = basic[basic["DOR"].isin(imaging.index)].set_index("DOR")
print("\n=== animals\n", basic[["Animal_ID", "DOB", "Ages", "Genotype", "Sex", "CuttingOS", "HoldingOS",
                             "RecordingOS"]].join(n_comp.rename("compartments")).to_string())
inj = inj[inj["Animal_ID"].isin(basic["Animal_ID"])]
inj = inj.merge(basic.reset_index()[["Animal_ID", "DOR"]], on="Animal_ID").set_index("DOR").sort_index()
print("\n=== injections\n", inj[["Animal_ID", "DOI", "Inj_Mode", "Side", "Virus_Short", "Incubated", "Mix_Ratio",
                            "Volume_Per_Shot"]].join(n_comp.rename("compartments")).to_string())
