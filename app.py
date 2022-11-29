import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fastapi import FastAPI
import uvicorn
from prophet import Prophet
import pandas_datareader.data as web  # pandas_datareader-0.10.0
from dateutil.relativedelta import relativedelta
from prophet.plot import plot_plotly, plot_components_plotly

app = FastAPI(
    title='testing',
    description='separate product',
    version='1.0.0',
    docs_url='/docs',
)


@app.get(path="/product",
          tags=['market product'])

async def product(name: str = '^DJI'):
    end = datetime.datetime.now()

    try:
        start = end - relativedelta(years=50)
        data = web.DataReader(name, 'yahoo', start, end)['Adj Close']
        data.plot()
        plt.show()
    except IOError:
        start = end - relativedelta(years=20)
    else:
        start = end - relativedelta(years=10)

        # data = web.DataReader("JBALX", 'yahoo', start, end)['Adj Close']
        # data.plot()
        # plt.show()

    data2 = data.to_frame()
    data2 = data2.reset_index()
    data2.columns = ['ds', 'y']

    modelT = Prophet()
    modelT.fit(data2)
    PreTime = modelT.make_future_dataframe(freq='Y', periods=11, include_history=False)
    PreDF = modelT.predict(PreTime)

    PreInflationTimeS1 = PreDF['yhat']

    PreTime = modelT.make_future_dataframe(freq='D', periods=365)
    PreDF = modelT.predict(PreTime)

    # fig1 = modelT.plot(PreDF)  # Plot the fit to past data and future forcast.
    # fig1.set_xlabel('date')
    # fig1.set_ylabel('inflation rate')
    # fig2 = model.plot_components(forecast) # Plot breakdown of components.

    # fig2 = plot_plotly(modelT, PreDF)
    # plt.show()
    return PreInflationTimeS1


if __name__ == '__main__':
     uvicorn.run('app:app', port=8080)

