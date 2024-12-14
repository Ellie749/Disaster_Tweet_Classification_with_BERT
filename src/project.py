import warnings
warnings.filterwarnings('ignore')
from dataset.tokenizer import read_data
from network.network_architecture import build_network
from model.train_model import run_train
from utils import visualize_train_results

X_train, X_test, y_train, y_test = read_data('src/dataset/train.csv', 'src/dataset/test.csv')
build_network()
model = build_network()
H = run_train(model, X_train, y_train)
visualize_train_results(H)
# #inference
