import pandas as pd
import csv

def xls_to_csv():

    data_xls = pd.read_excel('../data/bialaczka.xls', 'bialaczka', index_col=None)
    data_xls.to_csv('bialaczka.csv', encoding='utf-8')

if __name__ == '__main__':
    xls_to_csv()