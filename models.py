from consts import *
from keras.layers import Dense, Flatten, Input, concatenate, Reshape, Lambda
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU

def AmplitudeModel():
    ain = Input(shape=(FFT_BINS, 2, 11))
    aout = Flatten()(ain)
    aout = Lambda(lambda x:(x - AMPLITUDE_MEAN)/AMPLITUDE_STD)(aout)
    aout = Dense(500, activation=LeakyReLU())(aout)
    aout = Dense(500, activation=LeakyReLU())(aout)
    return ain, aout

def PhaseModel():
    pin = Input(shape=(FFT_BINS, 2 * 2, 11))
    pout = Flatten()(pin)
    pout = Dense(500, activation=LeakyReLU())(pout)
    pout = Dense(500, activation=LeakyReLU())(pout)
    return pin, pout

def APModel(_name):
    ain, aout = AmplitudeModel()
    pin, pout = PhaseModel()

    output = concatenate([aout, pout])
    output = Dense(2*FFT_BINS, activation=LeakyReLU())(output)
    output = Reshape((FFT_BINS, 2))(output)
    
    model = Model(input=[ain, pin], output=output, name=_name)
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

