#-----------------------------------------------------------------------------------------FS Predictor-------------------------------------------------------------------------------------------------------------#########
#Coded By:Mr. Jithin Jose
#11-07-2019



#--------------------------------------------------------------------------- Header Section -------------------------------------------------------------------------------------------------------------#########


from sklearn.feature_extraction.text import CountVectorizer # for feature exrtraction
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime 
from statsmodels.tsa.stattools import adfuller#Dicky fuller test
from statsmodels.tsa.arima_model import ARIMA
from matplotlib.pylab import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error  
from statsmodels.tsa.stattools import acf, pacf  
import numpy as np
import csv
import pandas as pd #for data processing and analysis
import matplotlib.pyplot as plt  #for data visualisation
rcParams['figure.figsize']=10,6






#------------------------------------------------------------------------ Function for testing stationarity--------------------------------------------------------------------------------------------#########
def test_stationarity(timeseries,v,c):
	#rollling mean rollig std
	rolmean=timeseries.rolling(window=12).mean()
	rolstd=timeseries.rolling(window=12).std()
	orig=plt.plot(timeseries,color='blue',label='original')
	mean=plt.plot(rolmean,color='red',label='Rolling Mean')
	std=plt.plot(rolstd,color='black',label='Rolling STD')
	plt.legend(loc='best')
	plt.title("Rolling Mean & Standard Deviation for "+" "+v+" "+c)
	plt.show()

	#dicky fuller test
	print('Results of Dickey-Fuller Test :'+" "+v)
	dftest = adfuller(timeseries, autolag='AIC')
	dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
	for key,value in dftest[4].items():
		dfoutput['Critical Value (%s)'%key] = value
	print(dfoutput)





def core_process(dataset,category):
	tb=dataset['Sales']
	#------------------------------------------------------------------- Plotting Time Series -----------------------------------------------------------------------------------------------------------#########
	plt.title(category+" "+"Sales Plot")
	plt.xlabel("Date")
	plt.ylabel("Sales")
	plt.plot(tb)
	plt.show()
    #------------------------------------------------------------------  Checking Stationarity ----------------------------------------------------------------------------------------------------------#########


	print("Stationarity Check of"+" "+category+"  Sales")
	test_stationarity(tb,category," ")
	print("Non-Stationarity found as mean & std is not constant along with p-value is larger")


	#-----------------------------------------------------------------  MAKING tb STATIONARY -------------------------------------------------------------------------------------------------------------#########

	#-----------------------------------------------------------------  Estimating trend -----------------------------------------------------------------------------------------------------------------#########
	tblog = np.log(tb)
	mov_avgb=tb.rolling(window=12).mean()
	plt.plot(tblog)
	plt.title("Estimating Trend of"+category)
	plt.show()
	plt.plot(mov_avgb,color='red')
	plt.title("Moving average of" + category+" Sales")
	plt.show()


	#------------------------------------------------------------------  Smoothing- -----------------------------------------------------------------------------------------------------------------------#########

	tmvavdiff=tblog-mov_avgb


	tmvavdiff= tmvavdiff.dropna()
	#print(tmvavdiff.head)

	test_stationarity(tmvavdiff,category,"Smoothing")
	## Exponentially Weighted Moving  Average
	cvb=tb.rolling(window=12).std()
	sdf=tmvavdiff-1000*cvb
	sdf=sdf.dropna()
	#print(sdf)
	test_stationarity(sdf,category,"Custom calculations for getting stationarity")
	###now p-value is very lesss & critical value is approximately equals to test value
	###Eliminating Trend & Seasonality

	#Differencing
	tblogs=tblog-tblog.shift()
	plt.plot(tblogs)
	plt.title("Log Shifted Trend of "+" "+ category)
	plt.show()
	tblogs=tblogs.dropna()
	test_stationarity(tblogs,category,"After Differencing")
	#-------------------------------------------------------------------  Decomposition -----------------------------------------------------------------------------------------------------------------#########

	decomposition = seasonal_decompose(tblog,freq=10)

	trend = decomposition.trend
	seasonal = decomposition.seasonal
	residual = decomposition.resid
	plt.title("4 components plot of "+" "+category)
	plt.subplot(411)
	plt.plot(tblog, label='Original')
	plt.legend(loc='best')
	plt.subplot(412)
	plt.plot(trend, label='Trend')
	plt.legend(loc='best')
	plt.subplot(413)
	plt.plot(seasonal,label='Seasonality')
	plt.legend(loc='best')
	plt.subplot(414)
	plt.plot(residual, label='Residuals')
	plt.legend(loc='best')
	plt.tight_layout()
	plt.show()
	tblog_decompose = residual
	tblog_decompose.dropna(inplace=True)
	test_stationarity(tblog_decompose,category,"After Decompostiton")

	#------------------------------------------------------------------  Final Forecasting -------------------------------------------------------------------------------------------------------------#########

	
	lag_acf = acf(tblogs, nlags=20)
	lag_pacf = pacf(tblogs, nlags=20, method='ols')

	#Plot ACF:    
	plt.subplot(121)    
	plt.plot(lag_acf)
	plt.axhline(y=0,linestyle='--',color='gray')
	plt.axhline(y=-1.96/np.sqrt(len(tblogs)),linestyle='--',color='gray')
	plt.axhline(y=1.96/np.sqrt(len(tblogs)),linestyle='--',color='gray')
	plt.title('Autocorrelation Function')
	plt.show()
	#Plot PACF:	
	plt.subplot(122)
	plt.plot(lag_pacf)
	plt.axhline(y=0,linestyle='--',color='gray')
	plt.axhline(y=-1.96/np.sqrt(len(tblogs)),linestyle='--',color='gray')
	plt.axhline(y=1.96/np.sqrt(len(tblogs)),linestyle='--',color='gray')
	plt.title('Partial Autocorrelation Function')
	plt.tight_layout()
	plt.show()

	#---------------------------------------------------------------- ---- ARIMA Model  -------------------------------------------------------------------------------------------------------------#########

	model = ARIMA(tblog, order=(2, 1, 2))  
	results_ARIMA = model.fit(disp=-1)  
	plt.plot(tblogs)
	plt.plot(results_ARIMA.fittedvalues, color='red')
	plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-tblogs)**2))
	plt.show()

	#--------------------- Original Scale_--------------------------------
	predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
	print (predictions_ARIMA_diff.head())
	predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
	#print(predictions_ARIMA_diff_cumsum.head())

	predictions_ARIMA_log = pd.Series(tblog.ix[0], index=tblog.index)
	predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
	#print(predictions_ARIMA_log.head())


	predictions_ARIMA = np.exp(predictions_ARIMA_log)
	#print(tblog.shape)

	##no.of rows ==228
	#to predict $ years we need 4 * 12 =48 points + given 228 points so total of 276 points is needed
	x=results_ARIMA.plot_predict(1,276)
	plt.title("Next 4 year prediction of" +" "+ category)
	plt.show()



   


#----------------------------------------------# Input  Details & Coreprocess initialization & calling--------------------------------------------------------############


dateparse = lambda x: pd.datetime.strptime(x, '%d-%m-%Y')
dfb = pd.read_csv(r'resources\Bookcases sorted .csv', index_col='Date',date_parser=dateparse)
#dfc = pd.read_csv(r'resources\chairs sorted.csv', index_col='Date',date_parser=dateparse)
#dff = pd.read_csv(r'resources\furnishings sorted.csv', index_col='Date',date_parser=dateparse)
#dft = pd.read_csv(r'resources\Tables.csv', index_col='Date',date_parser=dateparse)

core_process(dfb,"Bookcases")
#core_process(dff,"Chairs")
#core_process(dfc,"Furnishings")
#core_process(dft,"Tables")




"""
NB:- Prediction of other categories (Chairs,Tables & Furnishings can be done by the same function but you need to find out stationarity, Arima model(p,q,d)values......)
	Rolling Mean & Std, Dickey Fuller Test values shows whether the data is stationary or not.
	For further proceedings data must be stationary i.e, we need to convert non stationary data to stationary.
	Taking log,Differencing,Moving average differencing,exponential weighted average differencing are some methods to change the non stationarity
	Find p,q values for ARIMA model from ACF PACF graphs or from MA or AR models
"""


