import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import Model
from transformers import TFBertModel

MAX_LENGTH = 36

def build_network():
    bert = TFBertModel.from_pretrained('bert-base-uncased')
    
    input_ids = Input(name='input_IDs', shape=(MAX_LENGTH,), dtype=tf.int32)
    input_mask = Input(name='attention_mask', shape=(MAX_LENGTH,), dtype=tf.int32)

    cls = bert(input_ids, attention_mask=input_mask)[1]

    x = Dense(32, activation='relu')(cls)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation='softmax')(x)

    model = Model(inputs=[input_ids, input_mask], outputs=output)

    model.layers[2].trainable = False
    print(model.summary)

    return model
