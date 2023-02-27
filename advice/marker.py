import pandas as pd

def posovet(message):
    return int('посовет' in message)

def add_mark(df: pd.DataFrame, target: str, condition):
    target_list = []
    for message in df['message'].values:
        condition(message)
    df[target] = target_list
    return df