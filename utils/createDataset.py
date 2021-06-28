import pandas as pd
import numpy as np


def openHistoryYFcsv(file, regFilter=None):
    df = pd.read_csv(file, delimiter=",", header=[0, 1])
    df.drop([0], axis=0, inplace=True)
    df[('Unnamed: 0_level_0', 'Unnamed: 0_level_1')] = pd.to_datetime(df[('Unnamed: 0_level_0', 'Unnamed: 0_level_1')],
                                                                      format='%Y-%m-%d')
    df.set_index(('Unnamed: 0_level_0', 'Unnamed: 0_level_1'), inplace=True)

    if regFilter is not None:
        df = df.filter(regex=regFilter)
        df.columns = df.columns.droplevel() # We do not want multi-level index in this case
    else:
        df.index.name = None

    return df


def createDataApp(data, datePicker, threshOut):

    r_cols = ["r_SPX", "r_Stock"]
    r_cols_daily = ["r_SPX_daily", "r_Stock_daily"]
    data[r_cols_daily] = data[["SPX", "Stock"]].ffill().pct_change()
    data["30d_std_Stock"] = data["r_Stock_daily"].rolling(30).std()
    data["30d_std_SPX"] = data["r_SPX_daily"].rolling(30).std()
    data["std_ratio"] = data["30d_std_Stock"] / data["30d_std_SPX"]
    data = data.drop(["r_SPX_daily", "r_Stock_daily"], axis=1)

    data = data.resample('W').ffill()

    data[r_cols] = data[["SPX", "Stock"]].ffill().pct_change()
    data["cum_ret_Stock"] = 100 * (1 + data["r_Stock"]).cumprod()
    data["cum_ret_SPX"] = 100 * (1 + data["r_SPX"]).cumprod()
    data["based_Ratio"] = data["cum_ret_Stock"] / data["cum_ret_SPX"]
    data["r_diff"] = data["r_Stock"] - data["r_SPX"]
    data["r_diff_shift"] = data["r_diff"].shift(1)
    data["week"] = data.index.isocalendar().week
    data["month"] = data.index.month

    conditions_date = [(data.index < datePicker),
                       (data.index >= datePicker),
                       ]

    values_date = ["train", "validation"]
    data["split"] = np.select(conditions_date, values_date)

    conditions_tgt = [(data["r_diff_shift"] < threshOut),
                      (np.isnan(data["r_diff_shift"])),
                      (data["r_diff_shift"] >= threshOut)]

    values_tgt = [0, 0, 1]
    data["target"] = np.select(conditions_tgt, values_tgt)

    return data


