
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
import numpy as np

def build_model(inputs, output_size, neurons, activ_func="linear",
                dropout=0.25, loss="mae", optimizer="adam"):
    
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model

	
def run_model(temp_model,training_inputs,training_outputs,num_epochs,num_iter,pred_range):
	
	# random seed for reproducibility
	for rand_seed in range(775,775+num_iter+1): 
		print(rand_seed)

		# random seed for reproducibility
		np.random.seed(rand_seed)

		# train model on data & save to file
		temp_model.fit(training_inputs[:-pred_range], training_outputs, epochs=num_epochs, 
			batch_size=1, verbose=2, shuffle=True)
		temp_model.save('btc_model_5tpts_randseed_%d.h5'%rand_seed)

		return temp_model
		





