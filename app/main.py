import datetime
import json
import time
from sklearn.model_selection import train_test_split
from patsy.highlevel import dmatrix
import statsmodels.api as sm
import re


import matplotlib
import pandas as pd
from pandas import DataFrame
import numpy as np

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from fastapi import FastAPI
import uvicorn
from prophet import Prophet
from pydantic import BaseModel
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()



app = FastAPI(
    title='testing',
    description='analytics combination',
    version='1.0.0',
    docs_url='/docs',
)

def decimal2(listin):
    listout = []
    for i in listin:
        listout.append(float("%.2f" % i))
    return listout

@app.post(path="/product",
          tags=['market product'])
async def product(name: str = '^DJI'):
    try:
        data = yf.download(name)['Adj Close']
    except:
        data = pdr.get_data_yahoo(name)['Adj Close']

    data2 = data.to_frame()
    data2 = data2.reset_index()
    data2.columns = ['ds', 'y']
    data2['ds'] = data2['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))
    data2['ds'] = pd.to_datetime(data2['ds'])
    data2 = data2.drop(0)

    modelT = Prophet()
    modelT.fit(data2)
    PreTime = modelT.make_future_dataframe(freq='D', periods=365, include_history=False)
    PreDF = modelT.predict(PreTime)

    PreInflationTimeS1 = PreDF['yhat']

    # PreTime = modelT.make_future_dataframe(freq='D', periods=365)
    # PreDF = modelT.predict(PreTime)
    #
    fig1 = modelT.plot(PreDF)  # Plot the fit to past data and future forcast.
    plt.show()
    return PreInflationTimeS1


class Portfolio(BaseModel):
    ETFcodes: list = 'voo', 'qqq'
    proportion: list = 50, 50
    totalAmount: int = 1000


@app.post(path="/portfolio",
          tags=['market product'])
async def portfolio(portfolio: Portfolio):
    end = datetime.datetime.now()
    changeRateList = []
    changeRateListUpper = []
    changeRateListLower = []

    pricePreList = []
    pricePreListUpper = []
    pricePreListLower = []

    amountPreList = []
    amountPreUpperList = []
    amountPreLowerList = []
    preTimeList = []

    for i in range(len(portfolio.ETFcodes)):

        try:
            data = yf.download(portfolio.ETFcodes[i])['Adj Close']
        except:
            data = pdr.get_data_yahoo(portfolio.ETFcodes[i])['Adj Close']

        data2 = data.to_frame()
        data2 = data2.reset_index()
        data2.columns = ['ds', 'y']
        data2['ds'] = data2['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))
        data2['ds'] = pd.to_datetime(data2['ds'])

        rate = []
        for j in range(len(data2.y) - 1):
            temp = (data2.y[j + 1] - data2.y[j]) / data2.y[j]
            rate.append(temp)
        data2 = data2.drop(0)
        fitData = DataFrame({'ds': data2.ds, 'y': rate})

        modelT = Prophet()
        modelT.fit(data2)

        PreTime = modelT.make_future_dataframe(freq='D', periods=365)
        PreDF = modelT.predict(PreTime)
        fig1 = modelT.plot(PreDF)  # Plot the fit to past data and future forcast.
        plt.title(portfolio.ETFcodes[i] + ' price forecasting')
        plt.show()

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

        PreTime = modelT.make_future_dataframe(freq='D', periods=365)
        PreDF = modelT.predict(PreTime)

        fig2 = modelT.plot(PreDF)  # Plot the fit to past data and future forcast.
        plt.title(portfolio.ETFcodes[i] + ' change rate forecasting')
        plt.show()

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

    color = ['#708069', '#FF6347', '#808A87', '#FAEBD7', '#ED9121']

    plt.figure(figsize=(10, 8))
    for n in range(len(portfolio.ETFcodes)):
        plt.plot(preTimeList[n], amountPreList[n], label=portfolio.ETFcodes[n],
                 color=color[n])  # pretime is different from the last one
        plt.plot(preTimeList[n], amountPreUpperList[n], linestyle='dashed',
                 label=portfolio.ETFcodes[n] + ' higher line',
                 color=color[n])  # add lower higher line
        plt.plot(preTimeList[n], amountPreLowerList[n], linestyle='dashed', label=portfolio.ETFcodes[n] + ' lower line',
                 color=color[n])
    if n > 0:
        plt.plot(preTimeList[n], totalAmountPre, label='total amount')
    plt.legend()
    plt.title('ETF invest amount prediction')
    plt.show()

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


class Networth(BaseModel):
    ETF: list = [[{'code': 'voo', 'currency': 'USD', 'value': 1000, 'startDate': datetime.date(2022, 2, 1)},
                 {'code': 'qqq', 'currency': 'USD', 'value': 1000, 'startDate': datetime.date(2022, 2, 1)}],
                 {'reportCurrency': 'USD'}]
    shares: list = [[{'code': 'amzn', 'currency': 'USD', 'buyPrice': 130, 'heldNumber': 100}], {'reportCurrency': 'USD'}]
    mFunds: list = [[{'code': 'vtbix', 'currency': 'USD', 'value': 1000, 'startDate': datetime.date(2022, 2, 1)}], {'reportCurrency': 'USD'}]
    cash: list = [[
        {'value': 1000, 'currency': 'HKD', 'yearLength': 11, 'cashInterest': 3}], {'reportCurrency': 'USD'}]  # cash face value and actual value
    termDeposit: list = [[
        {'currency': 'USD', 'value': 1000, 'termInterest': 3, 'termStart': datetime.date(2022, 2, 1),
         'termEnd': datetime.date(2032, 2, 1)}], {'reportCurrency': 'USD'}]
    #asset: list = [{'currency': 'USD', 'value': 1000}]  # fix asset, just fix number in this version


def noneminus(list):
    outList = []
    for i in list:
        if i >= 0:
            m = i
        else:
            m = 0
        outList.append(m)
    return outList

# functions for combination

def possibility(LU, pre, std_z):
    std = np.std(pre)
    prediction_interval = std * std_z

    if LU == 'Lower':
        result = []
        for i in pre:
            result.append(i - prediction_interval)
    else:
        result = []
        for i in pre:
            result.append(i + prediction_interval)
    return result

def yh_forecast(product, type):

    return_price_list_ori = []
    return_price_list_trans = []

    preTimeList = []
    for i in range(len(product[0])):
        try:
            data = yf.download(product[0][i]['code'])['Adj Close']
        except:
            data = pdr.get_data_yahoo(product[0][i]['code'])['Adj Close']

        data2 = data.to_frame()
        data2 = data2.reset_index()
        data2.columns = ['ds', 'y']
        data2['ds'] = data2['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))
        data2['ds'] = pd.to_datetime(data2['ds'])
        data2 = data2.drop(0)

        modelT = Prophet()
        modelT.fit(data2)
        PreTime = modelT.make_future_dataframe(freq='D', periods=365, include_history=False)
        PreDF = modelT.predict(PreTime)
        pricePre = list(PreDF['yhat'])
        pricePreUpper = list(PreDF['yhat_upper'])  # did not add in git
        pricePreLower = list(PreDF['yhat_lower'])

        # 0.86 for 60%
        pricePre60Upper = possibility('Upper', pricePre, 0.85)
        pricePre60Lower = possibility('Lower', pricePre, 0.85)

        return_price = []
        return_price_60upper = []
        return_price_60lower = []

        # add start time

        if type == 'ETF' or type == 'mFunds':
            currencyRate = yf.download(product[0][i]['currency'] + '=X', period='1d')['Adj Close'][0]
            value = product[0][i]['value'] / currencyRate
            try:
                start = product[0][i]['startDate']
                start = datetime.datetime.strptime(start, "%Y-%m-%d").date()
                end = start + datetime.timedelta(days=7 - start.weekday() if start.weekday() > 3 else 1)
                buyPrice = yf.download(product[0][i]['code'], start=start, end=end)['Adj Close'].values[0]
            except:
                start = product[0][i]['startDate'] + datetime.timedelta(days=7)
                start = datetime.datetime.strptime(start, "%Y-%m-%d").date()
                end = start + datetime.timedelta(days=7 - start.weekday() if start.weekday() > 3 else 1)
                buyPrice = yf.download(product[0][i]['code'], start=start, end=end)['Adj Close'].values[0]

            for index in range(len(pricePre)):
                temp1 = pricePre[index] * value / buyPrice  # using held number of product
                temp2 = pricePre60Upper[index] * value / buyPrice
                temp3 = pricePre60Lower[index] * value / buyPrice
                return_price.append(temp1)
                return_price_60upper.append(temp2)
                return_price_60lower.append(temp3)
        elif type == 'shares':
            for index in range(len(pricePre)):
                temp1 = pricePre[index] * product[0][i]['heldNumber']   # using held number of product
                temp2 = pricePre60Upper[index] * product[0][i]['heldNumber']
                temp3 = pricePre60Lower[index] * product[0][i]['heldNumber']
                return_price.append(temp1)
                return_price_60upper.append(temp2)
                return_price_60lower.append(temp3)
        else:
            print('type name wrong')
        # return_price_list.append(noneminus(return_price))
        # return_price60upper_list.append(noneminus(return_price_60upper))
        # return_price60lower_list.append(noneminus(return_price_60lower))

        strTimeList = []
        for single_date in PreTime.values:
            dateTime = pd.to_datetime(single_date)
            strTime = dateTime.strftime('%Y-%m-%d')[0]
            strTimeList.append(strTime)

        preTimeList.append(strTimeList)

        return_price = noneminus(return_price)
        return_price_60lower = noneminus(return_price_60lower)
        return_price_60upper = noneminus(return_price_60upper)
        if i == 0:
            total_return = np.array(return_price)
            total_return_upper = np.array(return_price_60upper)
            total_return_lower = np.array(return_price_60lower)
        else:
            total_return = total_return + np.array(return_price)
            total_return_upper = total_return_upper + np.array(return_price_60upper)
            total_return_lower = total_return_lower + np.array(return_price_60upper)

    # return separet product
    # Ori currency
        currencyRate = yf.download(product[0][i]['currency'] + '=X', period='1d')['Adj Close'][0]
        singleList = []
        for n in range(len(return_price)):
            temp = {'time': preTimeList[0][n],
                    'PricePre': return_price[n] * currencyRate,
                    'PricePreUpper': return_price_60upper[n] * currencyRate,
                    'PricePreLower': return_price_60lower[n] * currencyRate
                    }
            singleList.append(temp)
        singleList = [{'USD/inputCurrencyRate': currencyRate}, singleList]
        return_price_list_ori.append(singleList)


        # add different codes amount pre together
        # transfer to report currency
        reportCurrencyRate = yf.download(product[1]['reportCurrency'] + '=X', period='1d')['Adj Close'][0]
        singleList = []
        for n in range(len(return_price)):
            temp = {'time': preTimeList[0][n],
                    'PricePre': return_price[n] * reportCurrencyRate,
                    'PricePreUpper': return_price_60upper[n] * reportCurrencyRate,
                    'PricePreLower': return_price_60lower[n] * reportCurrencyRate
                    }
            singleList.append(temp)

        if product[0][i]['currency'] == product[1]['reportCurrency']:
            oriReportRate = 1
        else:
            oriReportRate = yf.download(product[0][i]['currency'] + product[1]['reportCurrency']
                                             + '=X', interval='1m', period='1d')['Adj Close'][-1]

        singleList = [{'inputCurrency/reportCurrencyRate': oriReportRate}, singleList]
        return_price_list_trans.append(singleList)

    returnCombineList = []
    for n in range(len(total_return)):
        temp = {'time': preTimeList[0][n],
                'totalPricePre': total_return[n] * reportCurrencyRate,
                'totalPricePreUpper': total_return_upper[n] * reportCurrencyRate,
                'totalPricePreLower': total_return_lower[n] * reportCurrencyRate
                }
        returnCombineList.append(temp)

    returnCombineList = {'perAccountOriCurrency': return_price_list_ori, 'perAccountTransCurrency': return_price_list_trans, 'total': returnCombineList}
# # pics
#     color = ['#708069', '#FF6347', '#808A87', '#FAEBD7', '#ED9121']
#
#     plt.figure(figsize=(10, 8))
#     for n in range(len(product)):
#         plt.plot(preTimeList[n], return_price_list[n], label=product[n]['code'], color=color[n]) # pretime is different from the last one
#         plt.plot(preTimeList[n], return_price60upper_list[n], linestyle='dashed', label=product[n]['code'] + 'higher line', color=color[n]) # add lower higher line
#         plt.plot(preTimeList[n], return_price60lower_list[n], linestyle='dashed', label=product[n]['code'] + 'lower line', color=color[n])
#     if n > 0:
#         plt.plot(preTimeList[n],total_return,label='total amount')
#     plt.legend()
#     plt.title(product[n]['code'] + ' return forecasting')
#     plt.show()

    return returnCombineList


def Ftimestamp2(time1):
    times = time.strptime(str(time1), "%Y%m%d")  # for json file
    if time1 > 19700101:
        time2 = time.mktime(times)
    else:  # calculating from 1940, 94108800 is 19700101 T8 from 19400101
        time2 = (times.tm_year - 1940) * 31536000 + (times.tm_mon - 1) * 2628000 + (
                times.tm_mday - 1) * 86400 - 946108800

    return int(time2)

@app.post(path="/netWorth",
          tags=['net worth combine'])

async def netWorth(networth: Networth):

    # asset based on yahoo finance
    try:
        forecastETF = {'forecastETF': yh_forecast(networth.ETF, 'ETF')}
    except:
        forecastETF = 'import wrong'
    try:
        forecastShares = {'forecastShares': yh_forecast(networth.shares, 'shares')}
    except:
        forecastShares = 'import wrong'
    try:
        forecastMutralFunds = {'forecastMuralFunds': yh_forecast(networth.mFunds, 'mFunds')}
    except:
        forecastMutralFunds = 'import wrong'

    # term deposit without inflation
    try:
        oriCurrencyList = []
        transCurrencyList = []
        for index in range(len(networth.termDeposit[0])):
            if networth.termDeposit[0][index]['currency'] == networth.termDeposit[1]['reportCurrency']:
                oriReportRate = 1
            else:
                oriReportRate = yf.download(networth.termDeposit[0][index]['currency'] + networth.termDeposit[1]['reportCurrency']
                                            + '=X', interval='1m', period='1d')['Adj Close'][-1]
            value = networth.termDeposit[0][index]['value']
            end = networth.termDeposit[0][index]['termEnd']
            end = datetime.datetime.strptime(end, "%Y-%m-%d").date()
            start = networth.termDeposit[0][index]['termStart']
            start = datetime.datetime.strptime(start, "%Y-%m-%d").date()
            T = (end - start).days / 365
            oriCurrency = value * (1 + networth.termDeposit[0][index]['termInterest'] / 100) ** T
            oriCurrency = {'maturityTime': end, 'termDepositOriCurrency': oriCurrency,
                           'inputCurrency': networth.termDeposit[0][index]['currency']}
            transCurrency = value * (1 + networth.termDeposit[0][index]['termInterest'] / 100) ** T * oriReportRate
            transCurrency = {'maturityTime': end, 'termDepositTransCurrency': transCurrency,
                             'reportCurrency': networth.termDeposit[1]['reportCurrency'],
                             'input/reportCurrencyRate': oriReportRate}
            oriCurrencyList.append(oriCurrency)
            transCurrencyList.append(transCurrency)
        termReturnList = {'termDeposit': {'perAccountOriCurrency': oriCurrencyList,
                                          'perAccountTransCurrency': transCurrencyList}}

    except:
        termReturnList = 'import wrong'

    # cash
    try:
        thisYear = int(time.strftime('%Y', time.localtime()))  # 2022

        # cash prediction
        cashReturnList = []
        cashReturnListT = []

        for cashN in range(len(networth.cash[0])):
            interest = networth.cash[0][cashN]['cashInterest'] / 100
            yearLength = networth.cash[0][cashN]['yearLength']
            try:
                with open('data/' + networth.cash[0][cashN]['currency'][0:2] + 'data.json', 'r') as file:   # /app/data/ in api
                    contents = json.load(file)
                inflationData = DataFrame(contents['inflation'])
            except:
                with open('data/HKdata.json', 'r') as file:    # in git 'data/'
                    contents = json.load(file)
                inflationData = DataFrame(contents['inflation'])

            if networth.cash[0][cashN]['currency'][0:2] == 'AU' or networth.cash[0][cashN]['currency'][0:2] == 'NZ':  # inflation forecast with different model
                if networth.cash[0][cashN]['currency'][0:2] == 'AU':
                    fixKnots = (-570304800, 162835200, 991324800)  # used, just couldn't show
                    '''
                        19511201: -570304800
                        19750301: 162835200
                        20010601: 991324800
                    '''
                    preStartDate = -641260800  # 19490901

                if networth.cash[0][cashN]['currency'][0:2] == 'NZ':
                    fixKnots = (-1161604800, -294364800, 707328000)  # used, just couldn't show
                    '''
                        19330301: -1161604800
                        19600901: -294364800
                        19920601: 707328000
                    '''
                    preStartDate = -1342936800  # 19270601


                x = []
                for single_date in inflationData.ds:
                    single_date = time.strptime(single_date, '%Y-%m-%d %H:%M:%S')  # for json file
                    single_date = int(time.strftime('%Y%m%d', single_date))
                    timestamp = Ftimestamp2(single_date)
                    x.append(timestamp)

                # split train test data
                train_x, valid_x, train_y, valid_y = train_test_split(x, list(inflationData.y), test_size=0.33, random_state=1)
                # spline modelT training
                df_cut, bins = pd.cut(train_x, 4, retbins=True, right=True)
                df_cut.value_counts()
                # fit GLM with 3 knots
                transformed_x = dmatrix("bs(train, knots=fixKnots, degree=3, include_intercept=False)",
                                        {"train": train_x}, return_type='dataframe')
                model = sm.GLM(train_y, transformed_x).fit()
                thisDate = int(time.strftime('%Y%m%d', time.localtime()))  # 20221110
                preDate = np.linspace(preStartDate, int(Ftimestamp2(thisDate + yearLength * 10000)),
                                      500)  # change to timestamp

                matrixX = dmatrix("bs(PreBondDate, knots=fixKnots, include_intercept=False)",
                                  {"PreBondDate": preDate}, return_type='dataframe')
                preInflationSpl = model.predict(matrixX)
                #  20221110 added
                preIFDate2 = []  # change timestamp into date numbers e.g.20221110
                for i in preDate:
                    i = str(datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=i))  # for negative timestamp
                    i = re.findall(r'(.+?)\ ', i)[0]
                    strucTime = time.strptime(i, '%Y-%m-%d')  # %H:%M:%S')
                    temp = int(time.strftime('%Y%m%d', strucTime))
                    preIFDate2.append(temp)

                df = DataFrame(list(preInflationSpl), index=preIFDate2)
                dateShort = []
                for i in range(thisYear, thisYear + yearLength):
                    dateShort.append(i * 10000)

                InflationShort = []  # calculating the prediction number
                for i in dateShort:
                    temp = []
                    for index in df.index:
                        if i < index < i + 10000:
                            temp.append(df.loc[index])
                    InflationShort.append(np.mean(temp))

                dateShort = []
                for i in range(thisYear, thisYear + yearLength):
                    dateShort.append(i)

                inflationLower = possibility('Lower', InflationShort, 1.645)
                inflationUpper = possibility('Upper', InflationShort, 1.645)
                inflation60Lower = possibility('Lower', InflationShort, 0.85)
                inflation60Upper = possibility('Upper', InflationShort, 0.85)

            else:

                modelT = Prophet()
                modelT.fit(inflationData)
                preTime = modelT.make_future_dataframe(freq='Y', periods=yearLength, include_history=False)
                preDF = modelT.predict(preTime)

                InflationShort = preDF['yhat']

                dateShort = []
                for i in range(thisYear, thisYear + yearLength):
                    dateShort.append(i)
                inflationLower = possibility('Lower', InflationShort, 1.645)
                inflationUpper = possibility('Upper', InflationShort, 1.645)
                inflation60Lower = possibility('Lower', InflationShort, 0.85)
                inflation60Upper = possibility('Upper', InflationShort, 0.85)

            # Cash amount prediction
            def cashAmount(Inflation):
                cashAmountPre = []
                tSaveShort = networth.cash[0][cashN]['value']
                for i in dateShort:
                    i = int(i - thisYear)
                    cashAmountPre.append(tSaveShort)
                    tSaveShort = tSaveShort * (1 + interest - Inflation[i])
                cashAmountPre = decimal2(cashAmountPre)
                return cashAmountPre

            cashAmountPre = cashAmount(InflationShort)
            cashAmountLower = cashAmount(inflationUpper)
            cashAmountUpper = cashAmount(inflationLower)
            cashAmount60Lower = cashAmount(inflation60Upper)
            cashAmount60Upper = cashAmount(inflation60Lower)

            ## calculation of report currency
            if networth.cash[0][cashN]['currency'] == networth.cash[1]['reportCurrency']:
                reportCurrencyRate = 1
            else:
                reportCurrencyRate = yf.download(networth.cash[0][cashN]['currency'] + networth.cash[1]['reportCurrency']
                                                 + '=X', interval='1m', period='1d')['Adj Close'][-1]
            cashAmountPreT = np.array(cashAmountPre) * reportCurrencyRate
            cashAmountLowerT = np.array(cashAmountLower) * reportCurrencyRate
            cashAmountUpperT = np.array(cashAmountUpper) * reportCurrencyRate
            cashAmount60LowerT = np.array(cashAmount60Lower) * reportCurrencyRate
            cashAmount60UpperT = np.array(cashAmount60Upper) * reportCurrencyRate

            cashSingleList = []
            for i in range(len(dateShort)):
                temp = {'year': dateShort[i], 'cashPrediction': cashAmountPre[i], 'lower': cashAmountLower[i],
                        'upper': cashAmountUpper[i],
                        'sixtyPercentLower': cashAmount60Lower[i], 'sixtyPercentUpper': cashAmount60Upper[i],
                        'inflation': InflationShort[i]}
                cashSingleList.append(temp)
            cashSingleList = [{'reportCurrencyRate': reportCurrencyRate}, cashSingleList]
            cashReturnList.append(cashSingleList)

            cashSingleListT = []
            for i in range(len(dateShort)):
                temp = {'year': dateShort[i], 'cashPrediction': cashAmountPreT[i], 'lower': cashAmountLowerT[i],
                        'upper': cashAmountUpperT[i],
                        'sixtyPercentLower': cashAmount60LowerT[i], 'sixtyPercentUpper': cashAmount60UpperT[i],
                        'inflation': InflationShort[i]}
                cashSingleListT.append(temp)
            cashSingleListT = [{'reportCurrencyRate': reportCurrencyRate}, cashSingleListT]
            cashReturnListT.append(cashSingleListT)



            ########### below testing
            if cashN == 0:
                cash_total_return = np.array(cashAmountPreT)
                cash_total_return_lower = np.array(cashAmountLowerT)
                cash_total_return_upper = np.array(cashAmountUpperT)
                cash_total_return_60lower = np.array(cashAmount60LowerT)
                cash_total_return_60upper = np.array(cashAmount60UpperT)

            else:
                if len(cash_total_return) >= len(cashAmountPre):
                    for i in range(len(cashAmountPre)):
                        cash_total_return[i] = cash_total_return[i] + cashAmountPreT[i]
                        cash_total_return_lower[i] = cash_total_return_lower[i] + cashAmountLowerT[i]
                        cash_total_return_upper[i] = cash_total_return_upper[i] + cashAmountUpperT[i]
                        cash_total_return_60lower[i] = cash_total_return_60lower[i] + cashAmount60LowerT[i]
                        cash_total_return_60upper[i] = cash_total_return_60upper[i] + cashAmount60UpperT[i]
                else:
                    for i in range(len(cash_total_return)):
                        cash_total_return[i] = cash_total_return[i] + cashAmountPreT[i]
                        cash_total_return_lower[i] = cash_total_return_lower[i] + cashAmountLowerT[i]
                        cash_total_return_upper[i] = cash_total_return_upper[i] + cashAmountUpperT[i]
                        cash_total_return_60lower[i] = cash_total_return_60lower[i] + cashAmount60LowerT[i]
                        cash_total_return_60upper[i] = cash_total_return_60upper[i] + cashAmount60UpperT[i]

            # find the longest year length
            dateShort = []
            for i in range(thisYear, thisYear + len(cash_total_return)+1):
                dateShort.append(i)
            # find the longest inflation list
            # if cashN == 0:
            #     InflationShort_long = InflationShort
            # else:
            #     if len(InflationShort)>len(InflationShort_long):
            #         InflationShort_long = InflationShort

            cashTotalOutput = []
            for i in range(len(cash_total_return)):
                totalTemp = {'year': dateShort[i], 'cashPrediction': cash_total_return[i],
                             'lower': cash_total_return_lower[i],
                             'upper': cash_total_return_upper[i],
                             'sixtyPercentLower': cash_total_return_60lower[i],
                             'sixtyPercentUpper': cash_total_return_60upper[i]}
                             #'inflation': InflationShort_long[i]}      combine different cash account, no need return inflation in total
                cashTotalOutput.append(totalTemp)

            cashReturnArray = {'forcastCash': {'perAccountOriCurrency': cashReturnList, 'perAccountTransCurrency': cashReturnListT, 'total': cashTotalOutput}}
    except:
        cashReturnArray = 'import wrong'

    return cashReturnArray, termReturnList, forecastETF, forecastMutralFunds #, forecastShares

if __name__ == '__main__':
    uvicorn.run('main:app', port=9090)
