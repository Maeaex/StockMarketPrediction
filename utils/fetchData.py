#https://github.com/ranaroussi/yfinance/blob/main/docs/quickstart.md
import yfinance as yf
import pandas as pd

def fetchTickerData(tickerList, mode):
    data_list = list()
    for ticker in tickerList:
        try:

            symbol = yf.Ticker(ticker)
            print("Retrieving data for:", symbol)
            if mode == 'hist':
                tickerData = symbol.history(period="5y", group_by="Ticker")
                tickerData['ticker'] = ticker
            elif mode == 'info':
                info = symbol.info
                tickerData = pd.DataFrame.from_dict(info, orient='index').T
                tickerData['ticker'] = ticker
                tickerData.reset_index()
            elif mode == 'div':
                info = symbol.dividends
                tickerData = info.to_frame(ticker)
            else:
                print("No mode or wrong arg was provided")
        except:
            print("Error thrown at:", symbol)
            continue

        data_list.append(tickerData)

    df = pd.concat(data_list)

    return df

tickerList = pd.read_csv("../data/spxTickerList.csv", delimiter=",")
tickerTuple = tuple(list(tickerList["Symbol"]))


args = ['info', 'div'] # hist was downloaded using bulk data

for arg in args:
    data = fetchTickerData(tickerTuple, arg)
    data.to_csv("../data/spxSingleStockData_" + arg + ".csv")

# Index Data seperate as no adj. Close column + Bulk data download with spx not possible
symbol = yf.Ticker("^GSPC")
spxData = symbol.history(period="5y", group_by="Ticker")
spxData.to_csv("../data/spxIndexData.csv")







