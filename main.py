import dsdtools
import gc
import keras

from preprocessing import *
from consts import *
from models  import *

dsd = dsdtools.DB(root_dir=DB_PATH)
dev_set = dsd.load_dsd_tracks(subsets='Dev')
test_set = dsd.load_dsd_tracks(subsets='Test')
targets = TARGETS
models = list()
for t in targets:
    models.append(APModel(t))
models_input = processInput(dev_set)

def trainModel(target, epochs=EPOCHS, batch_size=BATCH_SIZE, save_epochs=False):
    model = models[targets.index(target)]
    model_target = processTarget(dev_set, target)
    checkpointer = keras.callbacks.ModelCheckpoint(model.name + '_DNN_model', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    callbacks = None
    if save_epochs:
        callbacks = [checkpointer]

    history = model.fit(models_input, model_target,  epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        

def training(epochs=10, batch_size=BATCH_SIZE):
    for target in TARGETS:
        trainModel(target, epochs, batch_size)
    runEstimations(test_set)

def singleEstimation(track):
    estimations = {}
    i = 0
    for target in targets:
        estimations[target] = makePredictions(models[i], track, target)
        i += 1
    saveEstimates(estimations, track, estimates_dir=ESTIMATES_PATH)

def runEstimations(t):
    for track in t:
        singleEstimation(track)


training(epochs=EPOCHS)
