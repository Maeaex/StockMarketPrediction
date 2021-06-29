# Stock Market Prediction 

###Installation

App was developed on Python Version 3.7.10.

To use the app an active internet connection is required (Yahoo Finance API is used).

To install the App run following command in your env

```
pip install -r requirements.txt
```

Run script `main.py` (optionally app.py - the location of the app)



FYI: On load callback errors will be displayed in the console as the app's state-management has to be improved.

####Folder structure
    StockMarketPrediction
        assets |
            |---style.css   from https://github.com/STATWORX/blog/tree/master/DashApp
        data   |
            |---spxTickerList.csv
        model  |
            |---models.py           Data prep for ML, Model Pipeline, create data set for model visuals
        utils  |
            |---__init__.py      
            |---createDataset.py    Wrangle data and calculate features
            |---fetchData.py        Load data from YF API 
        .gitignore
        app.py
        main.py
        README.md
        requirements.txt
 

### App Setup

App uses [Yahoo Finance API](#https://github.com/ranaroussi/yfinance/blob/main/docs/quickstart.md) to access financial data of S&P500 constituents

1. Select a stock from the dropdown list or search by typing a stock's name.
2. Choose a period (3yrs, 5yrs, Max) for the stock history.
3. Choose a date as a breaking point to split the data in train/test & validation sets (default: timedelta -3 weeks from today)
4. Choose a reasonable outperformance threshold between Stock and S&P500 (Stock return - S&P500 return) - based on this selection the data set will be labeled (outperformance = yes/no) 
5. Click "Get Data" Button

--> Data for Charts  section are loaded - Descriptive Charts will load
6. Enter a full integer in the input field for the number of splits - [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
7. Click "Run Models" - Depending on the sample size the models need a couple of seconds to compute


### App enhancements

* Layouting (Fonts, Grid-Color if no data is available for charts yet)
* App State-Management - prevent App from updating if no user interaction occurred
* DatePicker Update (default value is selected after "Get Data" button was clicked)



# Motivation
Aim is to demonstrate DataScience/Econometrics skills combined with know-how in process automatization & app development. 
The app setup allows to integrate new Machine Learning models and data pre-processing steps easily as Sklearn's Pipeline combined with GridSearchCV were implemented.

In the following I will explain my thoughts about research design, the data set and chosen features. 
I will conclude with features that should be implemented in this app.

### Thoughts on statistical design 
Goal of the use case is to predict if a particular stock outperforms an associated benchmark (SP500) on a weekly basis by a self-defined threshold.  
Thus the statistical design involves time-series  as well as classification characteristics.                          

#### Classic Time-Series Prediction
                                
*Technical:*
 Basic built-in implementations of ARIMA models in python (statmodels) compared to R packages
 No built-in model pipeline with model selection
 
 *Statistical:*
 Different statistical designs possible 
    * estimate & predict returns from stock and benchmark independently and calculate outperformance ex-post
    * calculate outperformance ex-ante and use this return series to fit and predict AR-models 
  Do assumptions about stationarity, residuals etc. still hold in second approach? - Happy to [discuss](https://quant.stackexchange.com/questions/65715/prediciting-outperformance-choice-of-statistical-design) :-)
    
#### Time-Series as a Classification Problem

*Technical:*
A lot of different ml and statistical models available. Pipeline and GridSearch allow for fast experimenting 
Flexible with Deep Learning. Integration of PyTorch' DL models into Pipeline & GridSearchCV also possible with Skorch library

*Statistical:*
Financial Data often inhabit features of high non-linearity that may be not well captured by classical statitical models (e.g. linear regression)
that also may be time-varying (e.g. volatility clusters). Models like Decision-trees, Support Vector Machines can caputure these non-linearites.
Usage of sklearn's TimeSeriesSplit helps to model time-series and classification problem at the same time while help to prevent model overfitting.

*Model Pipeline:*
* As evaluation metrics we use Balanced_Accuracy_Score as the data set can be unbalanced
* No Scaler was used
* Used models with different parameters: Support Vector Machine, Decision Tree Classifier, Naive Bayes Classifier


## Data & Feature Set

Based on the input parameters prices for a stock and SP500 are obtained using the Yahoo Finance API. 
From these prices daily returns are calculated. Based on the daily returns a rolling standard deviation is calculated to better capture volatility clusters.
We also build a ratio of the rolling standard deviations to may caputure industry specifc trends that do not directly correlate with the broader market.
The daily return are then dropped and the data set is resampled on a weekly basis and returns are calculated again to obtain weekly return.
Afterwards cumulative (cumprod as no log return) returns are calculated and rebased on level=100. 
As another feature we divide the cumulative returns to obtain an out-/underperformance ratio to capture relative momentum. 
We then calculate our *target* variable (the difference between weekly performances) which is labeled later according to the threshold. 
This measurement is shifted by one week as we want to predict outperformance based on data of the last weeks.

We then obtain categorical data from dates (Calender Week, Month of the year) to hopefully capture seasonal patterns like earnings, dividend announcements etc.
This data is transformed using dummy variables instead of ordinal notation.

List of features:
1. Rolling 30d standard deviation (not annualized) Stock
2. Rolling 30d std (not annualized) S&P
3. Rolling 30d std ratio (std_Stock / std_S&P)
4. Weekly return Stock 
5. Weekly return S&P500
6. Cumulative return Stock - For some stocks outperformance in the long run looks suspicious - check if close price is indeed adj for CA 
7. Cumulative return S&P
8. Ratio cumulative return Stock / S&P
9. Calendar week (transformed)
10. Month of the year (transformed)
11. *Target* - weekly return difference shifted by one week - which is then labeled based on threshold condition as 0 (underperformance) or 1 (outperformance)

## Data/Feature interpretation

1. Returns are likely heavy tailed and skewed
2. Return differential even more heavy tailed
3. Weekly out-/underperformance tends to be clustered (stock momentum, industry trends etc.)

## Features for development in this app
1. Descriptive Analysis - Better understand dynamics in the data (e.g. Scatter plot, correlation matrix etc.)
2. Better model interpretation - overfitting, feature selection/importance -- plot results of GridSearchCV 




