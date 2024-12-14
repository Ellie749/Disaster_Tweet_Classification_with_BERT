import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from transformers import AutoTokenizer


def read_data(train_df_path: str, test_df_path: str):
    train_dataframe = pd.read_csv(train_df_path, usecols=['text', 'target'])
    test_dataframe = pd.read_csv(test_df_path, usecols=['text'])

    #print(train_dataframe)
    #print(test_dataframe)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    X_train = tokenizer(train_dataframe.text.tolist(), padding=True, max_length=36, truncation=True, return_tensors='tf')
    print(X_train)
    X_test = tokenizer(test_dataframe.text.tolist(), padding=True, max_length=36, truncation=True, return_tensors='tf')
    y_train = train_dataframe.target.values
    y_test = None
    
    print(np.array(X_train).shape)
    print(np.array(X_test).shape)
    print(np.array(y_train).shape)

    return X_train, X_test, y_train, y_test



