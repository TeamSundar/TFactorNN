# Import modules
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Display
import os
import tensorflow as tf


print('Tensorflow version:', tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import tensorflow
import collections

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from functions import encode, build_model

# Load the TensorBoard notebook extension
#%load_ext tensorboard
import tensorflow as tf
import datetime

# Clear any logs from previous runs
#!rm -rf logs/

tfs = ['CEBPB', 'MAX', 'MYC', 'JUNB', 'REST','CTCF', 'FOSL2','JUN']
#tfs = ['CTCF', 'FOSL2','JUN']
#tfs = ['JUN']
mode = 'base'
for TF in tfs:
    print('************ Running for %s ************\n'%(TF))

    #x2 = np.load('/mnt/media/tfbind3/training_v4/'+'x2_'+TF+'.npy')
    X1_train = np.load('/mnt/media/tfbind3/training_v6_'+mode+'/'+'x1_train_'+TF+'.npy')
    X1_test = np.load('/mnt/media/tfbind3/training_v6_'+mode+'/'+'x1_test_'+TF+'.npy')
    y_train = np.load('/mnt/media/tfbind3/training_v6_'+mode+'/'+'y_train_'+TF+'.npy')
    y_test = np.load('/mnt/media/tfbind3/training_v6_'+mode+'/'+'y_test_'+TF+'.npy')

    print(X1_train.shape, X1_test.shape)
    print(collections.Counter(y_train))

    model = build_model()

    # Metrics
    metrics_c = ['accuracy',
        tensorflow.keras.metrics.Precision(name="precision"),
        tensorflow.keras.metrics.Recall(name="recall"),
        tensorflow.keras.metrics.AUC(name="auc_pr",curve="PR"),
        tensorflow.keras.metrics.AUC(name="auc_roc",curve="ROC")
    ]

    # if TF=='CEBPB':
    #     EPOCHS=100
    # else:
    EPOCHS = 500
    BATCH_SIZE = 512
    LEARNING_RATE = 0.001
    print('========')
    print('EPOCHS:', EPOCHS)
    print('BATCH_SIZE:', BATCH_SIZE)
    print('LEARNING_RATE:', LEARNING_RATE)
    print('========')
    
    # Training the model 
    model.compile(loss = keras.losses.binary_crossentropy,
                optimizer = keras.optimizers.SGD(lr=LEARNING_RATE),
                metrics=metrics_c)

    log_dir = "logs/"+TF+"_"+mode+"/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # if not os.path.exists(log_dir):
    #     os.makedirs('log_dir')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)


    # Create a callback that saves the model's weights
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    checkpoint_path = 'checkpoints/model_checkpoint_'+mode+'_'+TF+'.h5'
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=False,
                                                    save_best_only=True,
                                                    verbose=1)


    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)

    # fitting the model 
    print('\nTraining')
    history = model.fit(X1_train, y_train,
            batch_size = BATCH_SIZE,
            epochs = EPOCHS,
            verbose = 1,
            validation_data =(X1_test, y_test),callbacks=[cp_callback, es, tensorboard_callback])

    # evaluating and printing results  Wired LAN connection not working

    #model.save('checkpoints/model_checkpoint_'+TF+'.h5')
    if not os.path.exists('train_results'):
        os.makedirs('train_results')
    np.save('train_results/history_'+mode+'_'+TF+'.npy',history.history)
    del model
