CUDA_VISIBLE_DEVICES=1 python main.py simple_NN adam relu relu 
CUDA_VISIBLE_DEVICES=1 python main.py simple_NN adam softmax softmax
CUDA_VISIBLE_DEVICES=1 python main.py simple_NN adam tanh tanh
CUDA_VISIBLE_DEVICES=1 python main.py simple_NN adam sigmoid sigmoid
CUDA_VISIBLE_DEVICES=1 python main.py simple_NN adam hard_sigmoid hard_sigmoid
CUDA_VISIBLE_DEVICES=1 python main.py simple_NN adam linear linear
CUDA_VISIBLE_DEVICES=1 python main.py simple_NN adam selu selu
CUDA_VISIBLE_DEVICES=1 python main.py simple_NN adam softplus softplus
CUDA_VISIBLE_DEVICES=1 python main.py simple_NN adam softsign softsign
CUDA_VISIBLE_DEVICES=1 python main.py simple_NN adam elu elu



CUDA_VISIBLE_DEVICES=1 python main.py simple_CNN adam relu relu relu
CUDA_VISIBLE_DEVICES=1 python main.py simple_CNN adam softmax softmax softmax
CUDA_VISIBLE_DEVICES=1 python main.py simple_CNN adam tanh tanh tanh
CUDA_VISIBLE_DEVICES=1 python main.py simple_CNN adam sigmoid sigmoid sigmoid
CUDA_VISIBLE_DEVICES=1 python main.py simple_CNN adam hard_sigmoid hard_sigmoid hard_sigmoid
CUDA_VISIBLE_DEVICES=1 python main.py simple_CNN adam linear linear linear
CUDA_VISIBLE_DEVICES=1 python main.py simple_CNN adam selu selu selu
CUDA_VISIBLE_DEVICES=1 python main.py simple_CNN adam softplus softplus softplus
CUDA_VISIBLE_DEVICES=1 python main.py simple_CNN adam softsign softsign softsign
CUDA_VISIBLE_DEVICES=1 python main.py simple_CNN adam elu elu elu
