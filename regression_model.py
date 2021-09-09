import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from numpy.random import seed
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python import set_random_seed
from tqdm import tqdm

from read_data import get_doc_scores_bq

seed(1)
set_random_seed(2)


def get_top_k_doc_names(doc_scores_bn, nelements):
    dnames = []
    dscores = []
    for k, v in doc_scores_bn.items():
        dnames.append(k)
        dscores.append(v)
    dnames = np.array(dnames)
    dscores = np.array(dscores)

    return dnames[np.argsort(-dscores)][0:nelements]


def new_q_dist(qname, rdbq, retrieved_doc_scores_by_query, nbins=200, n_top=10):
    # nbins = 200 # is the best on the Robust04 collection
    # nbins = 128  # is the best on the GOV2 collection
    doc_scores_bn = retrieved_doc_scores_by_query[qname]
    pred = np.array([doc_scores_bn[dn] for dn in doc_scores_bn.keys()])
    top_ranked_d = get_top_k_doc_names(doc_scores_bn, n_top)
    if qname not in rdbq.keys():
        num_rel_retr_docs = 0
    else:
        num_rel_retr_docs = sum([1 for dn in top_ranked_d if dn in rdbq[qname]])
    doc_scores = pred[np.argsort(-pred)][0:min(len(pred), n_top)]
    doc_scores = doc_scores / max(doc_scores)
    # bins = list(doc_scores)
    # while len(bins) < n_top:
    #     bins.append(0.0)
    bins, edges = np.histogram(doc_scores, bins=nbins, density=False)
    # normalize the freqs so that they become prob distributions
    bins = bins / np.sum(bins)
    return np.array(bins), num_rel_retr_docs


def compute_training_distributions(retrieved_doc_scores_by_query, rdbq, qnames, collection):
    if collection == 'Robust04':
        nbins = 200
    else:
        nbins = 128
    x = []
    y = []
    for qname in tqdm(qnames):
        if qname not in rdbq.keys() or qname not in retrieved_doc_scores_by_query.keys():
            continue
        dist, n_rel_ret_d = new_q_dist(qname, rdbq, retrieved_doc_scores_by_query, nbins=nbins)
        y.append(n_rel_ret_d)
        x.append(dist)
    return np.array(x), np.array(y)


def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    tfd = tfp.distributions
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(  # pylint: disable=g-long-lambda
            tfd.Normal(loc=t[..., :n],
                       scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])


def prior_trainable(kernel_size, bias_size=0, dtype=None):
    tfd = tfp.distributions
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(
            lambda t: tfd.Independent(tfd.Normal(loc=t, scale=1),  # pylint: disable=g-long-lambda
                                      reinterpreted_batch_ndims=1)),
    ])


def train_prob_reg_model_alt(x, y):
    hidd_l_size = 128
    # hidd_l_size = 50
    y = np.array(y, dtype=np.float32)
    x = np.array(x, dtype=np.float32)
    y = np.expand_dims(y, 1)

    tfd = tfp.distributions
    batch_size = 16
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(x.shape[1]), dtype=tf.float32),
        # tf.expand_dims(x, axis=-2),
        # tf.keras.layers.LSTM(units=x.shape[1], activation=tf.nn.relu),
        tfp.layers.DenseVariational(hidd_l_size, posterior_mean_field, prior_trainable, activation=tf.nn.relu,
                                    kl_weight=1 / batch_size),
        tfp.layers.DenseVariational(1, posterior_mean_field, prior_trainable, activation=tf.nn.relu,
                                    kl_weight=1 / batch_size)])

    kl = tf.reduce_sum(model.losses) / x.shape[0]
    loss = lambda y_true, y_pred: tf.reduce_mean(
        tfd.Normal(loc=y_pred, scale=0.5).kl_divergence(tfd.Normal(loc=y_true, scale=0.5))) + kl
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=loss, metrics=['mse', 'mae'])
    model.fit(x, y, batch_size=batch_size, epochs=100, verbose=True, use_multiprocessing=True)
    return model


def train_prob_reg_model(x, y):
    return train_prob_reg_model_alt(x, y)


def pred_w_prob_reg_model(model, x_test):
    # Make predictions.
    sampled_prediction = [model.predict(np.array([x_test])) for i in range(50)]
    sampled_prediction = np.mean([np.squeeze(yv) for yv in sampled_prediction])
    return sampled_prediction


def pred_w_prob_reg_model_batch(model, x_test):
    # Make predictions.
    sampled_prediction = [model.predict(np.array(x_test)) for i in range(50)]
    sampled_prediction = np.mean([np.squeeze(yv) for yv in sampled_prediction], axis=1)
    print(sampled_prediction)
    return sampled_prediction


def pred_w_prob_reg_model_batch_w_conf(model, x_test):
    # Make predictions.
    sampled_prediction = [model.predict(np.array(x_test)) for i in range(50)]
    preds = np.mean([np.squeeze(yv) for yv in sampled_prediction], axis=1)
    confs = np.std([np.squeeze(yv) for yv in sampled_prediction], axis=1)
    print(sampled_prediction)
    return preds, confs


def reg_model(hidd_size, input_dim):
    model = Sequential()
    model.add(Dense(hidd_size, input_dim=input_dim, kernel_initializer='normal', activation='tanh'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['mse', 'mae'])
    return model


def train_multipl_prob_reg_models(rdbq, train_qnames, run_paths, collection):
    trained_regressors = []
    for run_path in tqdm(run_paths):
        x, y = compute_training_distributions(get_doc_scores_bq(run_path), rdbq, train_qnames, collection)
        print('fitting regression model')
        model = train_prob_reg_model(x, y)
        trained_regressors.append(model)
    return trained_regressors


# def train_multipl_reg_models(rdbq, train_qnames, run_paths):
#     trained_regressors = []
#     for run_path in tqdm(run_paths):
#         x, y = compute_training_distributions(get_doc_scores_bq(run_path), rdbq, train_qnames)
#         print('fitting regression model')
#         model = reg_model(64, 100)
#         model.fit(x, y, batch_size=32, epochs=2, use_multiprocessing=True)
#         trained_regressors.append(model)
#     return trained_regressors


def eval(pred, true):
    mae = mean_absolute_error(true, pred)
    print('mean absolute error: ' + str(mae))

    mse = mean_squared_error(true, pred)
    print('mean squared error: ' + str(mse))
    return mae, mse
