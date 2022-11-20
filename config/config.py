from yacs.config import CfgNode as ConfigurationNode

_C = ConfigurationNode()
_C.TRAINING = ConfigurationNode()
_C.TRAINING.SEED = 0
_C.TRAINING.N_EPOCH = 100
_C.TRAINING.LEARNING_RATE = 0.1
_C.TRAINING.BATCH_SIZE = 10
_C.TRAINING.N_TRAIN = 200
_C.TRAINING.N_TEST = 100

_C.MODEL = ConfigurationNode()
_C.MODEL.N_IN = 1
_C.MODEL.N_LAYERS = 1
_C.MODEL.N_NODES = 5
_C.MODEL.N_OUT = 1
_C.MODEL.Q = 100
_C.MODEL.HIDDEN_Q = 10

_C.DATASET = ConfigurationNode()
_C.DATASET.DATA_DIR = './OU_q100/'


def get_cfg_defaults():
    return _C.clone()


run_id = 0
seed = 0
Nepoch = 100
learning_rate = 0.1
batch_size = 10

#NW architecture: Nhidden layers of Nnodes each, ip and op 1 node
# n_in - [n_layers] x n_nodes - n_out
n_in = 1
n_layers = 1
n_nodes = 5
n_out = 1

data_dir = './OU_q100/'
q = 100
hidden_q = 10
N_train = 200
N_test = 100
