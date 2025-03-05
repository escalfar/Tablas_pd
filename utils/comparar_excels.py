import sys

import pandas as pd

df1_ = pd.read_excel(r'..\prueba_mediana_etc_v1.xlsx', list(range(41)), header=[0, 1], index_col=[0, 1])
# df2_ = pd.read_excel(r'..\prueba_mediana_etc_v2.xlsx', list(range(41)), header=[0, 1], index_col=[0, 1])
#  = pd.read_excel('.xlsx', None)

all_sheets = []
for name, sheet in df1_.items():
    # sheet['sheet'] = name
    # sheet = sheet.rename(columns=lambda x: x.split('\n')[-1])
    all_sheets.append(sheet)
    # print(sheet)
    # sys.exit()

# print(all_sheets)


"""
import pandas as pd
import numpy as np

xl_1 = pd.read_excel('file1.xlsx', sheet_name=None)
xl_2 = pd.read_excel('file2.xlsx', sheet_name=None)

with ExcelWriter('./Excel_diff.xlsx') as writer:
    for sheet,df1 in xl_1.items():
        # check if sheet is in the other Excel file
        if sheet in xl_2:
            df2 = xl_2[sheet]
            comparison_values = df1.values == df2.values
            
            print(comparison_values)
            
            rows, cols = np.where(comparison_values == False)
            for item in zip(rows,cols):
                df1.iloc[item[0], item[1]] = '{} --> {}'.format(df1.iloc[item[0], item[1]], df2.iloc[item[0], item[1]])

            df1.to_excel(writer, sheet_name=sheet, index=False, header=True)
"""