import pandas as pd
from pandas import DataFrame


def read_xls_ziwm(file_name: str) -> DataFrame:
    return pd.read_excel(file_name)

