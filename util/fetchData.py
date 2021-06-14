#https://github.com/ranaroussi/yfinance/blob/main/docs/quickstart.md
import yfinance as yf
import pandas as pd

def fetchSingleTickerData(ticker):
    symbol = yf.Ticker(ticker)
    print("Retrieving data for:", symbol)
    tickerData = symbol.history(period="5y")

    return tickerData

tickerList = pd.read_csv("../data/spxTickerList.csv", delimiter=",")

tickerTuple = tuple(list(tickerList["Symbol"]))

data = yf.download(tickerTuple, period="5y")

data.to_csv("../data/spxSingleStockData.csv")


df = pd.read_csv("data/spxSingleStockData.csv", delimiter=",", header=[0, 1])
df.drop([0], axis=0, inplace=True)  
df[('Unnamed: 0_level_0', 'Unnamed: 0_level_1')] = pd.to_datetime(df[('Unnamed: 0_level_0', 'Unnamed: 0_level_1')], format='%Y-%m-%d')
df.set_index(('Unnamed: 0_level_0', 'Unnamed: 0_level_1'), inplace=True)
df.index.name = None
df_red = df.filter(regex='Adj Close')






