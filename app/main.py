import datetime
# import matplotlib
import pandas as pd
from pandas import DataFrame
import numpy as np
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
from fastapi import FastAPI
import uvicorn
from prophet import Prophet
import pandas_datareader.data as web  # pandas_datareader-0.10.0
from dateutil.relativedelta import relativedelta
from pydantic import BaseModel

app = FastAPI(
    title='testing',
    description='separate product',
    version='1.0.0',
    docs_url='/docs',
)


@app.post(path="/product",
          tags=['market product'])

async def product(name: str = '^DJI'):
    end = datetime.datetime.now()

    try:
        start = end - relativedelta(years=50)
        data = web.DataReader(name, 'yahoo', start, end)['Adj Close']
    except IOError:
        start = end - relativedelta(years=20)
    else:
        start = end - relativedelta(years=10)


    data2 = data.to_frame()
    data2 = data2.reset_index()
    data2.columns = ['ds', 'y']

    modelT = Prophet()
    modelT.fit(data2)
    PreTime = modelT.make_future_dataframe(freq='Y', periods=5, include_history=False)
    PreDF = modelT.predict(PreTime)

    PreInflationTimeS1 = PreDF['yhat']

    # PreTime = modelT.make_future_dataframe(freq='D', periods=365)
    # PreDF = modelT.predict(PreTime)
    #
    # fig1 = modelT.plot(PreDF)  # Plot the fit to past data and future forcast.
    # plt.show()
    return PreInflationTimeS1


class Portfolio(BaseModel):
    ETFcodes: list = '0P0001EQUG', '0P0000XAH1'
    proportion: list = 50, 50
    totalAmount: int = 1000


@app.post(path="/portfolio",
          tags=['market product'])

async def portfolio(portfolio: Portfolio):
    end = datetime.datetime.now()
    changeRateList = []
    changeRateListUpper = []
    changeRateListLower = []

    pricePreList =[]
    pricePreListUpper = []
    pricePreListLower = []

    amountPreList = []
    amountPreUpperList = []
    amountPreLowerList = []
    preTimeList = []

    for i in range(len(portfolio.ETFcodes)):

        try:
            start = end - relativedelta(years=50)
            data = web.DataReader(portfolio.ETFcodes[i], 'yahoo', start, end)['Adj Close']

        except IOError:
            start = end - relativedelta(years=20)
        else:
            start = end - relativedelta(years=5)
            data = web.DataReader(portfolio.ETFcodes[i], 'yahoo', start, end)['Adj Close']

        data2 = data.to_frame()
        data2 = data2.reset_index()
        data2.columns = ['ds', 'y']
        rate = []
        for j in range(len(data2.y) - 1):
            temp = (data2.y[j + 1] - data2.y[j]) / data2.y[j]
            rate.append(temp)
        data2 = data2.drop(0)
        fitData = DataFrame({'ds': data2.ds, 'y': rate})

        modelT = Prophet()
        modelT.fit(data2)

        # PreTime = modelT.make_future_dataframe(freq='D', periods=730)
        # PreDF = modelT.predict(PreTime)
        # fig1 = modelT.plot(PreDF)  # Plot the fit to past data and future forcast.
        # plt.title(portfolio.ETFcodes[i] + ' price forecasting')
        # plt.show()

        PreTime = modelT.make_future_dataframe(freq='D', periods=365, include_history=False)
        PreDF = modelT.predict(PreTime)

        pricePre = list(PreDF['yhat'])
        pricePreUpper = list(PreDF['yhat_upper'])
        pricePrelower = list(PreDF['yhat_lower'])
        pricePreList.append(pricePre)
        pricePreListUpper.append(pricePreUpper)
        pricePreListLower.append(pricePrelower)


        modelT = Prophet()
        modelT.fit(fitData)

        # PreTime = modelT.make_future_dataframe(freq='D', periods=365)
        # PreDF = modelT.predict(PreTime)
        #
        # fig2 = modelT.plot(PreDF)  # Plot the fit to past data and future forcast.
        # plt.title(portfolio.ETFcodes[i] + ' change rate forecasting')
        # plt.show()

        PreTime = modelT.make_future_dataframe(freq='D', periods=365, include_history=False)
        PreDF = modelT.predict(PreTime)

        changeRatePre = list(PreDF['yhat'])
        changeRatePreUpper = list(PreDF['yhat_upper'])
        changeRatePrelower = list(PreDF['yhat_lower'])
        changeRateList.append(changeRatePre)
        changeRateListUpper.append(changeRatePreUpper)
        changeRateListLower.append(changeRatePrelower)
        temp = portfolio.proportion[i] / 100 * portfolio.totalAmount
        tempH = portfolio.proportion[i] / 100 * portfolio.totalAmount
        tempL = portfolio.proportion[i] / 100 * portfolio.totalAmount

        changeRateList.append(changeRatePre)

        amountPre = []
        amountPreUpper = []
        amountPreLower = []
        for j in range(len(changeRatePre)):
            amountPre.append(temp)
            amountPreUpper.append(tempH)
            amountPreLower.append(tempL)
            temp = temp * (1 + changeRatePre[j])
            tempH = tempH * (1 + changeRatePreUpper[j])
            tempL = tempL * (1 + changeRatePrelower[j])
        amountPreList.append(amountPre)
        amountPreUpperList.append(amountPreUpper)
        amountPreLowerList.append(amountPreLower)

        strTimeList = []
        for single_date in PreTime.values:
            # times = time.strptime(single_date[0], '%Y-%m-%d %H:%M:%S') # for json file
            dateTime = pd.to_datetime(single_date)
            strTime = dateTime.strftime('%Y-%m-%d')[0]
            strTimeList.append(strTime)

        preTimeList.append(strTimeList)

        if i == 0:
            totalAmountPre = np.array(amountPreList[i])
        else:
            totalAmountPre = totalAmountPre + np.array(amountPreList[i])

    # color = ['#708069', '#FF6347', '#808A87', '#FAEBD7', '#ED9121']
    #
    # plt.figure(figsize=(10, 8))
    # for n in range(len(portfolio.ETFcodes)):
    #     plt.plot(preTimeList[n], amountPreList[n], label=portfolio.ETFcodes[n],
    #              color=color[n])  # pretime is different from the last one
    #     plt.plot(preTimeList[n], amountPreUpperList[n], linestyle='dashed', label=portfolio.ETFcodes[n] + ' higher line',
    #              color=color[n])  # add lower higher line
    #     plt.plot(preTimeList[n], amountPreLowerList[n], linestyle='dashed', label=portfolio.ETFcodes[n] + ' lower line',
    #              color=color[n])
    # if n > 0:
    #     plt.plot(preTimeList[n], totalAmountPre, label='total amount')
    # plt.legend()
    # plt.title('ETF invest amount prediction')
    # plt.show()

    returnCombineList = []
    tempList = []
    for ETF in range(len(portfolio.ETFcodes)):
        for i in range(len(preTimeList[ETF])):
            temp = {'time': preTimeList[ETF][i], 'changeRatePre': changeRateList[ETF][i],
                    'changeRateUpper': changeRateListUpper[ETF][i], 'changeRateLower': changeRateListLower[ETF][i],
                    'pricePre': pricePreList[ETF][i], 'pricePreUpper': pricePreListUpper[ETF][i],
                    'pricePreLower': pricePreListLower[ETF][i],
                    'amountPre': amountPreList[ETF][i], 'amountPreUpper': amountPreUpperList[ETF][i],
                    'amountPreLower': amountPreLowerList[ETF][i]}
            tempList.append(temp)
        returnCombineList.append({portfolio.ETFcodes[ETF]: tempList})


    return returnCombineList

if __name__ == '__main__':
     uvicorn.run('main:app', port=9090)

