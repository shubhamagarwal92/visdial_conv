import pandas as pd


def pandas_excel_write(save_dir_path: str):
    data1 = """
    class    precision   recall 
    <18      0.0125         12           
    18-24    0.0250         16 
    25-34    0.00350         4
    """
    data2 = """
    sample    values
    <18      0
    18-24    0.25
    25-34    0.35
    """

    # create 2 df for sample
    df1 = pd.read_csv(pd.compat.StringIO(data1), sep='\s+')
    df1.name = "Dataframe1"
    df2 = pd.read_csv(pd.compat.StringIO(data2), sep='\s+')
    df2.name = "Dataframe2"
    print(df1)
    print(df2)


    write_file_path = f"{save_dir_path}/test_same_sheet.xlsx"
    writer = pd.ExcelWriter(write_file_path, engine='xlsxwriter')
    workbook = writer.book
    worksheet = workbook.add_worksheet('Result')
    writer.sheets['Result'] = worksheet
    worksheet.write_string(0, 0, df1.name)

    df1.to_excel(writer, sheet_name='Result', startrow=1, startcol=0)
    worksheet.write_string(df1.shape[0] + 4, 0, df2.name)
    df2.to_excel(writer, sheet_name='Result', startrow=df1.shape[0] + 5, startcol=0)


    ## Different sheets

    write_file_path = f"{save_dir_path}/test_diff_sheet.xlsx"
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(write_file_path, engine='xlsxwriter')

    s = pd.Series([1, 2, 3])
    df_describe = s.describe()

    # Write each dataframe to a different worksheet. you could write different string like above if you want
    df1.to_excel(writer, sheet_name='Sheet1')
    df2.to_excel(writer, sheet_name='Sheet2')
    df_describe.to_excel(writer, sheet_name='Sheet3')
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


if __name__ == "__main__":
    save_dir_path = "../data/data_analysis/"
    pandas_excel_write(save_dir_path)
