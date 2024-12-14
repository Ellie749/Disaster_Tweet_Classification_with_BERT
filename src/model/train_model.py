from tensorflow.keras.optimizers import Adam
import tensorflow as tf

OPTIMIZER = Adam(learning_rate = 6e-06)
EPOCHS = 2
BATCH_SIZE = 32

def run_train(model, X_train, y_train):
    

    # input_ids = tf.convert_to_tensor(X_train['input_ids'])
    # #print(input_ids)
    # attention_mask = tf.convert_to_tensor(X_train['attention_mask'])

    model.compile(optimizer = OPTIMIZER, loss= 'binary_crossentropy', metrics=['accuracy'])
    H = model.fit(x={'input_IDs': X_train['input_ids'], 'attention_mask': X_train['attention_mask']},
              y=y_train,
              #validation_split = 0.1,
              epochs = EPOCHS,
              batch_size = BATCH_SIZE)
    #callback save weights
    return H