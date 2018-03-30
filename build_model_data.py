
import pandas as pd
import numpy as np

def build_model_data(data_left,data_right,split_date):


	# Merge datasets
	model_data = pd.merge(left=data_left,right=data_right, how='left', left_index=True, right_index=True)
	model_data.fillna(method="ffill",inplace=True)
	model_data = model_data.dropna()

	#Instead of date as the index, create a date column. Reset column labels to be correct with concatenated data.
	model_data.reset_index(level=0, inplace=True) 
	model_data.columns = ['date'] + ['price'] + list(data_right.columns)

	#Drop any duplicate columns, some terms show up in both 'top' and 'rising' (e.g. ethereum)
	model_data=model_data.T.drop_duplicates().T
	model_data['date'] = model_data['date'].astype(str)


	# SPLIT DATA INTO TRAINING & TEST DATA AND NORMALIZE FOR LSTM MODEL ###################

	# Randomly choose a date as the cutoff between training and test data set
	training_set, test_set = model_data[model_data['date'] <= split_date], model_data[model_data['date'] > split_date]

	#Now drop date columns, as we do not want to put this information in the model
	training_set = training_set.drop(['date'],axis=1)
	training_set = training_set.apply(pd.to_numeric)

	test_set = test_set.drop(['date'],axis=1)
	test_set = test_set.apply(pd.to_numeric)

	return model_data,training_set,test_set


def reformat_data(training_set,test_set,pred_range):
	# Normalise the inputs to the first value in the moving window
	window_len = 10
	training_inputs = []
	test_inputs = []
	norm_cols = training_set.columns

	for i in range(len(training_set)-window_len):
		temp_set = training_set[i:(i+window_len)].copy()
		for col in norm_cols:
			temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1        
		training_inputs.append(temp_set)
	#training_outputs = (training_set['price'][window_len:].values/training_set['price'][:-window_len].values)-1
	
	for i in range(len(test_set)-window_len):
		temp_set = test_set[i:(i+window_len)].copy()
		for col in norm_cols:
			temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
		test_inputs.append(temp_set)
	test_outputs = (test_set['price'][window_len:].values/test_set['price'][:-window_len].values)-1

	#Convert to numpy arrays since all data is numeric
	training_inputs = [np.array(training_input) for training_input in training_inputs]
	training_inputs = np.array(training_inputs)

	test_inputs = [np.array(test_input) for test_input in test_inputs]
	test_inputs = np.array(test_inputs)

	# initialize outputs for the model
	training_outputs = []
	inc = pred_range
	for i in range(window_len, len(training_set['price'])-inc):
		training_outputs.append((training_set['price'][i:i+inc].values/
			training_set['price'].values[i-window_len])-1)
	
	training_outputs = np.array(training_outputs)

	print(training_inputs.shape)
	print(test_inputs.shape)

	print(training_outputs.shape)
	print(test_outputs.shape)


	return training_inputs, test_inputs, training_outputs, test_outputs




