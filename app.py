import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta, datetime

# Custom Functions
import utils.createDataset as ds
import utils.fetchData as fetch
import model.models as ml

import pandas as pd
import numpy as np

# Initialize the app
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True

df_tickers = pd.read_csv("data/spxTickerList.csv")

def get_tickerList(tickerData):
    tickerList = []
    for index, row in tickerData.iterrows():
        tickerList.append({"label": row["Name"], "value": row["Symbol"]})

    return tickerList

def get_paneDates(data):
    train_min = data.index[data['split'] == "train"].min()
    train_max = data.index[data['split'] == "train"].max()


    val_max = data.index[data['split'] == "validation"].max()

    return train_min, train_max, val_max

def labelOnCondition(df, col):
    conditions = [(df[col] <= 0),
                  (df[col] > 0),
                  (np.isnan(df[col]))]
    values = ["Negative", "Positive", "Negative"]

    labeled = pd.DataFrame(np.select(conditions, values),columns=["labeled"])

    return labeled




app.layout = html.Div(

    children=[
        dcc.Store(id='stockData'),
        html.Div(className='row',
                 children=[
                    html.Div(className='four columns div-user-controls',
                             children=[
                                 html.H2('Stock Outperformance Predicition'),
                                 html.P('Pick Stocks from the SP500 constituents list'),
                                 html.Div(
                                     className='div-for-dropdown',
                                     children=[
                                         dcc.Dropdown(id='stockSelector', options=get_tickerList(df_tickers),
                                                      multi=False, value="MSFT",
                                                      style={'backgroundColor': '#1E1E1E'},
                                                      className='stockselector'
                                                      ),
                                     ],
                                     style={'color': '#1E1E1E'}),
                                 html.P('Select period for data'),
                                 html.Div(
                                     className='div-for-dropdown',
                                     children=[
                                         dcc.Dropdown(id='periodSelector', options=[
                                             {"label": "3 years", "value": "3y"},
                                             {"label": "5 years", "value": "5y"},
                                             {"label": "Max", "value": "max"},
                                         ],
                                                      multi=False, value="3y",
                                                      style={'backgroundColor': '#1E1E1E'},
                                                      className='stockselector'
                                                      ),
                                     ],
                                     style={'color': '#1E1E1E'}),
                                 html.P('Select date for validation split (green pane is used for validation)'),
                                 html.Div(
                                     className='div-for-dropdown',
                                     children=[
                                         dcc.DatePickerSingle(
                                             id='datePicker',
                                             display_format='DD-MM-YYYY',
                                             first_day_of_week=1,
                                             initial_visible_month=str(date.today() - timedelta(days=28)),
                                             max_date_allowed=date.today() - timedelta(days=7),
                                             style={'background-color': '#1E1E1E'}
                                         )
                                     ]
                                     ,
                                     style={'color': '#1E1E1E'}
                                 ),
                                 html.P('Enter outperformance threshold for labeling'),
                                 html.Div(
                                     className='div-for-dropdown',
                                     children=[
                                            dcc.Input(
                                                id="thresholdInput",
                                                type="number",
                                                placeholder="Float number for %",
                                                value=0.015,
                                                style={'background-color': '#1E1E1E', 'color': '#fff'}
                                                    )
                                     ]
                                 ),
                                 html.Div(
                                     className='div-for-button',
                                     children=[
                                         html.Button('Get Data', id='triggerEvent')
                                     ]
                                 ),
                                 html.P('Enter number of splits for TimeSeriesSplit'),
                                 html.Div(
                                     className='div-for-dropdown',
                                     children=[
                                            dcc.Input(
                                                id="splitML",
                                                type="number",
                                                placeholder="Integer for splits",
                                                value=5,
                                                style={'background-color': '#1E1E1E', 'color': '#fff'}
                                                    )
                                     ]
                                 ),
                                 html.Div(
                                     className='div-for-button',
                                     children=[
                                         html.Button('Run Models', id='triggerML')
                                     ]
                                 ),
                                 html.Div(
                                     className='div-for-button',
                                     children=[
                                         dcc.Markdown('''
            
                                            ## App Setup
            
                                            App uses [Yahoo Finance API](#https://github.com/ranaroussi/yfinance/blob/main/docs/quickstart.md) to access financial data of S&P500 constituents
                                            
                                            1. Select a stock from the dropdown list or search by typing a stock's name.
                                            2. Choose a period (3yrs, 5yrs, Max) for the stock history.
                                            3. Choose a date as a breaking point to split the data in train/test & validation sets (default: timedelta -3 weeks from today)
                                            4. Choose a reasonable outperformance threshold between Stock and S&P500 (Stock return - S&P500 return) - based on this selection the data set will be labeled (outperformance = yes/no) 
                                            5. Click "Get Data" Button
                                            
                                            --> Data for Charts  section are loaded - Descriptive Charts will load
                                            6. Enter a full integer in the input field for the number of splits - [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
                                            7. Click "Run Models" - Depending on the sample size the models need a couple of seconds to compute
                                            
                                            
                                            ## App enhancements
                                            
                                            * Layouting (Fonts, Grid-Color if no data is available for charts yet)
                                            * App State-Management - prevent App from updating if no user interaction occured                                       
                                            ''')
                                     ]
                                 ),
                                ]
                             ),
                    html.Div(className='eight columns div-for-charts bg-grey',
                             children=[
                                dcc.Graph(id='timeseries'),
                                dcc.Graph(id='outperformance'),
                                html.H2("Charts below and ML fitting do not include validation set (green pane)", style={'text-align': 'center'}),
                                dcc.Graph(id='distPlot'),
                                dcc.Graph(id='trainPlot'),
                                dcc.Graph(id='valPlot'),
                                dcc.Markdown('''
                                
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
                                
                                
                                '''),
                             ])
                              ])
        ]

)

@app.callback(Output('stockData', 'data'),
              [Input('triggerEvent', 'n_clicks')],
              [State('stockSelector', 'value')],
              [State('periodSelector', 'value')],
              [State('datePicker', 'date')],
              [State('thresholdInput', 'value')],
              )


def getData(n_clicks, stockValue, periodValue, datePicker, threshOut):
    print("Get Data Button:", n_clicks)
    if datePicker is None:
        datePicker = str(date.today() - timedelta(days=28))

    if threshOut is None:
        threshOut = 0.03

    if n_clicks is not None:

     data = fetch.fetchTickerDataApp(stockValue, periodValue)
     dataPrep = ds.createDataApp(data, datePicker, threshOut)
     dataPrepJson = dataPrep.to_json(date_format='iso', orient='split')
    else:
        dataPrepJson = None

    n_clicks = None
    return dataPrepJson

@app.callback(
    [Output('datePicker', 'date')], # This updates the field start_date in the DatePicker
    [Input('stockData', 'data')],
)
def updateDataPicker(jsonified_cleaned_data):
    dff = pd.read_json(jsonified_cleaned_data, orient='split')
    start = dff.index.max() - timedelta(weeks=4)
    start = [datetime.strftime(start, '%Y-%m-%d')]

    return start

@app.callback(Output('timeseries', 'figure'),
              [Input('stockData', 'data')])
def update_timeseries(jsonified_cleaned_data):
    dff = pd.read_json(jsonified_cleaned_data, orient='split')
    train_min, train_max, val_max = get_paneDates(dff)


    df_sub = dff[["cum_ret_SPX", "cum_ret_Stock"]]

    figure = go.Figure()
    for stock in df_sub.columns:
        figure.add_trace(go.Scatter(x=df_sub.index,
                                 y=df_sub[stock].values,
                                 mode='lines',
                                 opacity=0.7,
                                 name=stock,
                                 textposition='bottom center'),
                         )

    figure.add_vrect(
        x0=train_min, x1=train_max,
        fillcolor="LightSalmon", opacity=0.5,
        layer="below", line_width=0,
    ),

    figure.add_vrect(
        x0=train_max, x1=val_max,
        fillcolor="lightgreen", opacity=0.5,
        layer="below", line_width=0,
    ),
    figure.update_layout(
        colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
        template='plotly_dark',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        margin={'b': 15},
        hovermode='x',
        autosize=True,
        title={'text': 'Performance SPX vs. Stock (rebased)', 'font': {'color': 'white'}, 'x': 0.5},
        xaxis={'range': [df_sub.index.min(), df_sub.index.max()]},
        legend=dict(
            orientation="h")
    )


    return figure

@app.callback(Output('outperformance', 'figure'),
              [Input('stockData', 'data')])
def update_change(jsonified_cleaned_data):
    dff = pd.read_json(jsonified_cleaned_data, orient='split')
    train_min, train_max, val_max = get_paneDates(dff)
    df_sub = pd.DataFrame(dff['r_diff'], columns=['r_diff'])

    conditions = [(df_sub["r_diff"] <= 0),
                  (df_sub['r_diff'] > 0),
                  (np.isnan(df_sub['r_diff']))]
    values = ["red", "green", "black"]
    df_sub["color"] = np.select(conditions, values)

    figure = go.Figure(go.Bar(x=df_sub.index,
                             y=df_sub["r_diff"].values,
                             #mode='lines',
                             marker={'color': df_sub['color']},
                             opacity=0.7,
                             name="Return Performance (r(Stock) - r(SPX))",
                             textposition='auto'))

    figure.add_vrect(
        x0=train_min, x1=train_max,
        fillcolor="LightSalmon", opacity=0.5,
        layer="below", line_width=0,
    ),


    figure.add_vrect(
        x0=train_max, x1=val_max,
        fillcolor="lightgreen", opacity=0.5,
        layer="below", line_width=0,
    ),

    figure.update_layout(
        colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
        template='plotly_dark',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        margin={'t': 50},
        height=250,
        hovermode='x',
        autosize=True,
        title={'text': 'Weekly Stock Under-/Outperfromance', 'font': {'color': 'white'}, 'x': 0.5},
        legend=dict(orientation="h")
                         )

    return figure



@app.callback(Output('distPlot', 'figure'),
              [Input('stockData', 'data')])
def update_change(jsonified_cleaned_data):
    dff = pd.read_json(jsonified_cleaned_data, orient='split')

    dff = dff.loc[(dff['split'] != "validation")]

    df_r_diff_shift = dff["r_diff_shift"]
    df_r_stock = dff["r_Stock"]
    df_r_spx = dff["r_SPX"]

    figure = make_subplots(rows=1, cols=3)
    figure.add_trace(go.Histogram(x=df_r_stock, name="Dist. returns Stock"), row=1, col=1)
    figure.add_trace(go.Histogram(x=df_r_spx, name="Dist. returns SPX"), row=1, col=2)
    figure.add_trace(go.Histogram(x=df_r_diff_shift, name="Dist. diff returns"), row=1, col=3)

    figure.update_layout(
        colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
        template='plotly_dark',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        margin={'t': 50},
        height=250,
        hovermode='x',
        autosize=True,
        title={'text': 'Return Distribution', 'font': {'color': 'white'}, 'x': 0.5},
        legend=dict(orientation="h")
    )

    return figure

@app.callback([Output('trainPlot', 'figure'),
              Output('valPlot', 'figure')],
              [Input('stockData', 'data')],
              [Input('triggerML', 'n_clicks')],
              [State('splitML', 'value')],
              )
def update_change(jsonified_cleaned_data, n_clicks, splits):
    print("ML Button was clicked", n_clicks)
    if n_clicks is not None:

        dff = pd.read_json(jsonified_cleaned_data, orient='split')
        print("raw data loaded - Records:", str(len(dff)))
        data = ml.encodeLabels(dff)
        print("data transformed")
        tscv, X, data_split, y, X_val, y_val = ml.timeSeriesSplit(data, splits)
        print("data splitted")

        mlGrid = ml.initModels(data_split)
        print("model initizalized")

        mlGrid = mlGrid.fit(X,y)
        print("Model fitted")

        ys, ys_val, bModel, conf, conf_val, y_pred_prob, y_pred_val_prob = ml.creatMlData(mlGrid, X, X_val, y, y_val)
        print("visual data generated")

        figure1 = make_subplots(rows=2, cols=1)


        figure1.add_trace(go.Bar(x=ys.index,
                                  y=ys["check"].values,
                                  # mode='lines',
                                  marker={'color': ys['color']},
                                  opacity=0.7,
                                  name="Correct/Incorrect predictions on TimeSeries data set",
                                  textposition='auto'),
                         row=1, col=1)

        figure1.add_trace(go.Scatter(x=ys.index,
                                 y=ys['y_pred_prob_pos'],
                                 mode='lines',
                                 opacity=0.7,
                                 name="Class 0 Probability (underperformance)",
                                 textposition='bottom center',
                                 ),
                          row=2, col=1)

        figure2 = make_subplots(rows=2, cols=1)

        figure2.add_trace(go.Bar(x=ys_val.index,
                                 y=ys_val["check"].values,
                                 # mode='lines',
                                 marker={'color': ys_val['color']},
                                 opacity=0.7,
                                 name="Correct/Incorrect predictions",
                                 textposition='auto'),
                          row=1, col=1)

        figure2.add_trace(go.Scatter(x=ys_val.index,
                                     y=ys_val['y_pred_prob_pos'],
                                     mode='lines',
                                     opacity=0.7,
                                     name="Class 0 Probability (underperformance)",
                                     textposition='bottom center',
                                     ),
                          row=2, col=1)

        figure1.update_layout(
            colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
            template='plotly_dark',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            margin={'t': 50},
            height=250,
            hovermode='x',
            autosize=True,
            title={'text': 'Correct and Incorrect classifications (TimeSeriesSplit)', 'font': {'color': 'white'},
                   'x': 0.5},
            legend=dict(orientation="h"),
        )

        figure2.update_layout(
            colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
            template='plotly_dark',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            margin={'t': 50},
            height=250,
            hovermode='x',
            autosize=True,
            title={'text': 'Correct and Incorrect classifications (validation)', 'font': {'color': 'white'}, 'x': 0.5},
            legend=dict(orientation="h")
        )
        return figure1, figure2

    else:
        pass

    n_clicks = None


if __name__ == '__main__':
    app.run_server(debug=True)