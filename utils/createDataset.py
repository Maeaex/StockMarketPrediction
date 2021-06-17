import pandas as pd


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


def createStockForecastData(stock_pick):
    df_merge = df_stocks.merge(df_spx, how="inner", left_index=True, right_index=True)
    df_div = pd.DataFrame(pd.read_csv(file_div, delimiter=",", parse_dates=True, index_col="Date")[stock_pick],
                          columns=[stock_pick])
    df_div.columns = ["Div"]
    df_set = pd.DataFrame(df_merge, columns=["SPX", stock_pick])
    df_set = df_set.merge(df_div, how="left", left_index=True, right_index=True)
    df_set["div_yield"] = df_set["Div"] / df_set[stock_pick]
    df_set = df_set.drop(columns=["Div"])
    r_cols = ["r_SPX", "r_"+stock_pick]
    df_set[r_cols] = df_merge[["SPX", stock_pick]].ffill().pct_change()
    df_set["r_diff"] = df_set[stock_pick] - df_set["SPX"]
    df_set["r_diff_shift"] = df_set["r_diff"].shift(-1)
    df_set["30d_std_stock"] = df_set[stock_pick].rolling(30).std()
    df_set["30d_std_SPX"] = df_set["SPX"].rolling(30).std()
    df_set["cum_ret_stock"] = 100 * (1 + df_set[stock_pick]).cumprod()
    df_set["cum_ret_SPX"] = 100 * (1 + df_set["SPX"]).cumprod()
    df_set["week"] = df_set.index.isocalendar().week
    df_set["month"] = df_set.index.month
    df_set = df_set.merge(df_div, how="left", left_index=True, right_index=True)

    df_wk = df_set.resample('W').ffill()

    return df_wk


file_stocks = "data/spxSingleStockData.csv"
filter_price = "Adj Close"
df_stocks = openHistoryYFcsv(file_stocks, filter_price)
file_spx = "data/spxIndexData.csv"
df_spx = pd.DataFrame(pd.read_csv(file_spx, delimiter=",", parse_dates=True, index_col="Date")["Close"],
                      columns=["Close"])
df_spx.columns = ["SPX"]
file_div = "data/spxSingleStockData_div.csv"
stock_pick = "AAPL"

