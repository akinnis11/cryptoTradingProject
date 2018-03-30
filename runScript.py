

from getData import getPriceData, trendData, relatedData
from build_model_data import build_model_data, reformat_data
from build_model import build_model, run_model

import pandas as pd
import numpy as np

# variables to set:
num_terms = 5
start_date = '2013-03-10'
end_date = '2018-03-01'
split_date = '2017-01-11' # also split by date into training & test data
num_epochs = 10
num_iter = 2
pred_range = 1
window_len = 10

# get bitcoin historical price data using coindesk's publically available API
url = "https://api.coindesk.com/v1/bpi/historical/close.json?start=" + start_date + "&end=" + end_date
df = getPriceData(url)

# get google trend data for term "bitcoin"
interest_over_time_df, pytrends = trendData(['bitcoin'],[]) 

# get google trend data for the search terms related to "bitcoin" (e.g. trend for "ethereum", "coinbase", etc)
top_terms, rising_terms = relatedData(currency='bitcoin',nterms=num_terms,pytrends=pytrends)

# concatenate all the google trend data that you want to include in the model
trend_data = pd.concat([interest_over_time_df,top_terms,rising_terms],axis=1)

# merge, 'clean-up', and prep data for putting in the model
model_data, training_set, test_set = build_model_data(df,trend_data,split_date)
LSTM_training_inputs, LSTM_test_inputs, LSTM_training_outputs, LSTM_test_outputs = reformat_data(training_set,test_set,pred_range)

# we'll try to predict the closing price for the next 5 days 
btc_model = build_model(LSTM_training_inputs, output_size=pred_range, neurons = 20)

btc_model = run_model(btc_model,LSTM_training_inputs,LSTM_training_outputs,num_epochs,num_iter,pred_range)

# Reformat the predictions for plotting
pred_prices=((btc_model.predict(LSTM_test_inputs)[:-pred_range]+1)*\
	test_set['price'].values[:-(window_len + pred_range)].reshape(int(np.ceil((len(LSTM_test_inputs)-pred_range)/float(1))),1))







