import numpy as np
import tensorflow as tf
import pandas as pd


def get_data():
    dataset = pd.read_csv('dataset/colunas_apagadas_exclui_linhas_valor_ruim.csv')
    df_y = pd.read_csv('dataset/labels.csv')

    X = dataset.values[:,:-1]
    y = df_y.values[:, :]

    X_train = X[:55932]
    y_train = y[:55932]

    X_validation = X[55932: 62923]
    y_validation = y[55932: 62923]

    X_test = X[62923:]
    y_test = y[62923:]

    return X_train, y_train, X_validation, y_validation, X_test, y_test


# help function to sampling data
def get_sample(num_samples, X_data, y_data):
    positions = np.arange(len(y_data))
    np.random.shuffle(positions)

    X_sample = []
    y_sample = []

    for posi in positions[:num_samples]:
        X_sample.append(X_data[posi])
        y_sample.append(y_data[posi])

    return X_sample, y_sample

INPUT_SIZE = 175
SIGMOID = tf.nn.sigmoid
ELU = tf.nn.elu
RELU = tf.nn.relu
TANH = tf.nn.tanh
SOFTPLUS = tf.nn.softplus

def mlp_1_layer(l1_act, layer_size):

    # input placeholder
    # mudei pra 1024 pra caber o hog encode
    x = tf.placeholder(tf.float32, [None, INPUT_SIZE])

    # output placeholder
    y_ = tf.placeholder(tf.float32, [None, 2])


    # weights of the neurons in first layer
    W1 = tf.Variable(tf.random_normal([INPUT_SIZE, layer_size], stddev=0.35))
    b1 = tf.Variable(tf.random_normal([layer_size], stddev=0.35))

    # weights of the neurons in second layer
    W2 = tf.Variable(tf.random_normal([layer_size, 2], stddev=0.35))
    b2 = tf.Variable(tf.random_normal([2], stddev=0.35))

    # hidden_layer value
    hidden_layer = l1_act(tf.matmul(x, W1) + b1)

    # output of the network
    y_estimated = tf.nn.softmax(tf.matmul(hidden_layer, W2) + b2)

    return x, y_, y_estimated

X_train, y_train, X_validation, y_validation, X_test, y_test = get_data()

models = [
    {
        'func': mlp_1_layer,
        'args': [ELU, 300],
        'title': 'mlp 1 layer com elu 300 nos'
    },
]

# for ret_net, title in models:
for model in models:
    net = model.get('func')
    title = model.get('title')
    x, y_, y_estimated = net(*model.get('args'))

    # function to measure the error
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_estimated), reduction_indices=[1]))


    # how to train the model
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


    # how to evaluate the model
    # correct_prediction = tf.equal(tf.argmax(y_estimated,1), tf.argmax(y_,1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




    ######################## training the model #######################################

    # applying a value for each variable (in this case W and b)
    init = tf.initialize_all_variables()



    # a session is dependent of the enviroment where tensorflow is running
    sess = tf.Session()
    sess.run(init)

    # a = tf.cast(tf.argmax(y_estimated, 1), tf.float32)
    # b = tf.cast(tf.argmax(y_, 1), tf.float32)
    auc = tf.contrib.metrics.streaming_auc(y_estimated, y_)

    num_batch_trainning = 500
    # x_grafico = []
    # y_grafico = []


    iteracoes = 1000

    # melhores_acuracia = []
    for i in range(iteracoes):
        # randomizing positions
        X_sample, y_sample = get_sample(num_batch_trainning, X_train, y_train)

        # where the magic happening

        sess.run(train_step, feed_dict={x: X_sample, y_:  y_sample})

        # print the accuracy result
        if i % 100 == 0:
            # acuracia_atual = (sess.run(accuracy, feed_dict={x: X_validation, y_: y_validation}))

            train_auc = (sess.run(auc, feed_dict={x: X_validation, y_: y_validation}))
            # x_grafico.append(i)
            # y_grafico.append(acuracia_atual)
            print(i, ": ", train_auc)
            # if acuracia_atual > 0.499:
            #     melhores_acuracia.append((i, acuracia_atual))

    print('\n\n\n')
    print("TEST RESULT: ", (sess.run(auc, feed_dict={x: X_test, y_: y_test})))
    # print("Melhores acuracias: {}".format(melhores_acuracia))