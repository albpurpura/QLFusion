import os
import uuid
from os import listdir

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

from read_data import get_doc_scores_bq

np.random.seed(0)
tf.random.set_random_seed(0)


class ProbRegressor:
    def __init__(self, seed, input_size, learning_rate):
        det_model_w_mse = False
        self.seed = seed
        self.global_step = tf.Variable(0, trainable=False)
        self.scaler = tf.Variable(1., trainable=True)
        self.input_data = tf.placeholder(dtype=tf.float32, shape=(None, input_size), name='data')
        self.labels = tf.placeholder(dtype=tf.float32, shape=None, name='labels')
        self.hidd_repr = self.input_data
        self.bn = tf.keras.layers.LayerNormalization()
        self.hidd_repr = self.bn(self.hidd_repr)
        self.hidd_repr = tf.layers.dense(self.hidd_repr, 32, activation=tf.nn.sigmoid)
        if det_model_w_mse:
            print('DET model with MSE')
            self.last_layer = tfp.layers.DenseFlipout(1, activation=tf.nn.sigmoid)  # 8 is the best for trec 5, 16 is not bad
            self.output = self.last_layer(self.hidd_repr)
            self.mse_loss = tf.reduce_mean(tf.square(self.output - self.labels))
            self.loss = self.mse_loss
            # n = input_size
            # self.loss = tf.reduce_mean(tf.log((1e-6 + self.labels) / (1e-6 + self.output)) * n * self.labels + tf.log(
            #     (1e-6 + 1 - self.labels) / (1e-6 + 1 - self.output)) * n * (1 - self.labels))
        else:
            print('PROB model with KL LOSS')
            self.last_layer = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
            self.output = self.last_layer(self.hidd_repr)

            # self.mse_loss = tf.reduce_mean(tf.square(self.output - self.labels))
            # self.loss = self.mse_loss
            n = input_size
            self.loss = tf.reduce_mean(tf.log((1e-6 + self.labels) / (1e-6 + self.output)) * n * self.labels + tf.log(
                (1e-6 + 1 - self.labels) / (1e-6 + 1 - self.output)) * n * (1 - self.labels))
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        self.init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                                tf.compat.v1.local_variables_initializer())
        self.train_op = tf.group([train_op, update_ops])
        self.saver = tf.train.Saver(max_to_keep=None)


class ProbLayer(tf.keras.layers.Layer):
    @staticmethod
    def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
        tfd = tfp.distributions
        n = kernel_size + bias_size
        c = np.log(np.expm1(1.))
        return tf.keras.Sequential([
            tfp.layers.VariableLayer(2 * n, dtype=dtype),
            tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                tfd.Normal(loc=t[..., :n], scale=1e-5 + tf.nn.softplus(c + t[..., n:])), reinterpreted_batch_ndims=1)),
        ])

    @staticmethod
    def prior_trainable(kernel_size, bias_size=0, dtype=None):
        tfd = tfp.distributions
        n = kernel_size + bias_size
        return tf.keras.Sequential([
            tfp.layers.VariableLayer(n, dtype=dtype),
            tfp.layers.DistributionLambda(
                lambda t: tfd.Independent(tfd.Normal(loc=t, scale=1), reinterpreted_batch_ndims=1)),
        ])

    def __init__(self, seed, output_size, batch_size, activ):
        super(ProbLayer, self).__init__()
        self.seed = seed
        self.layer = tfp.layers.DenseVariational(output_size, self.posterior_mean_field, self.prior_trainable,
                                                 activation=activ,
                                                 kl_weight=1 / batch_size)

    def call(self, input, **kwargs):
        return self.layer(input)


def train_multiple_models(run_paths, rdbq, train_qnames, collection, seed, learning_rate,
                          best_models_prev_folds, models_dir, fold, batch_size=8, n_epochs=300):
    # batch_size = 4  # 8 is best on robust, 4 on gov2
    # n_epochs = 300  # 500
    # models_dir = os.getcwd() + '/saved_models'
    print('epochs: {}, batch_size: {}'.format(n_epochs, batch_size))
    best_model_paths = []
    for run_path in tqdm(run_paths):
        x, y = compute_training_distributions(get_doc_scores_bq(run_path), rdbq, train_qnames, collection)
        # print('fitting regression model')
        best_model_path = train_model(x, y, seed, learning_rate, models_dir, fold, n_epochs=n_epochs,
                                      batch_size=batch_size)
        best_model_paths.append(best_model_path)
        models_files = [f for f in listdir(models_dir) if os.path.isfile(os.path.join(models_dir, f))]
        for mf in models_files:
            found = False
            union_list = best_models_prev_folds + best_model_paths
            for best_model_prefix in union_list:
                if mf.startswith(best_model_prefix.split('/')[-1]):
                    found = True
            if not found:
                os.remove(os.path.join(models_dir, mf))

    return best_model_paths


def get_batches(x, y, batch_size):
    x_batch = []
    y_batch = []
    for i in range(len(x)):
        x_batch.append(x[i])
        y_batch.append(y[i])
        if len(x_batch) == batch_size:
            yield np.array(x_batch), np.array(y_batch)
            x_batch = []
            y_batch = []
    if len(x_batch) > 0:
        yield np.array(x_batch), np.array(y_batch)


def train_model(x, y, seed, learning_rate, output_models_folder, fold, n_epochs=200, batch_size=16):
    valid_indices = np.random.choice([i for i in range(0, len(x))], int(len(x) * 0.1), replace=False)
    x_vali = np.array(x)[valid_indices]
    y_vali = np.array(y)[valid_indices]

    x = np.array([x[i] for i in range(len(x)) if i not in valid_indices])
    y = np.array([y[i] for i in range(len(y)) if i not in valid_indices])

    measures = []
    model_paths = []
    prev_MAE = np.inf
    max_patience = 20
    patience = 10
    tf.set_random_seed(0)
    tf.reset_default_graph()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config, graph=tf.get_default_graph()) as sess:
        tf.set_random_seed(seed)
        model = ProbRegressor(seed, x.shape[-1], learning_rate)
        sess.run(model.init_op)
        tf.set_random_seed(0)
        for epoch in range(n_epochs):
            for (data_batch, labels_batch) in get_batches(x, y, batch_size):
                _, step, loss, preds = sess.run([model.train_op, model.global_step, model.loss, model.output],
                                                feed_dict={model.input_data: data_batch, model.labels: labels_batch})
                # mae = float(np.mean(np.abs(preds - y)))
            preds_vali, loss_vali = sess.run([model.output, model.loss],
                                             feed_dict={model.input_data: x_vali, model.labels: y_vali})
            # mae_vali = float(np.mean(np.abs(preds_vali - y_vali)))
            # measures.append(mae_vali)
            measures.append(loss_vali)

            if loss_vali >= prev_MAE:
                patience -= 1
                prev_MAE = loss_vali
                if patience == 0:
                    print('Stopping early')
                    break
            else:
                patience = max_patience
                prev_MAE = loss_vali
            if epoch % 10 == 0:
                print('epoch: %d, step: %d, loss: %2.4f, loss valid: %2.8f' % (epoch, step, loss, loss_vali))
            model_name = str(uuid.uuid4())
            model_save_path = os.path.join(output_models_folder, model_name + '_fold=' + fold + '_' + str(step))
            model_paths.append(model_save_path)
            model.saver.save(sess, save_path=model_save_path)
    best_model_path = model_paths[np.argmin(measures)]
    print('best model path on fold {}: {}'.format(fold, best_model_path))
    # return model_save_path
    return best_model_path


def compute_training_distributions(retrieved_doc_scores_by_query, rdbq, qnames, collection):
    x = []
    y = []
    for qname in tqdm(qnames):
        if qname not in rdbq.keys() or qname not in retrieved_doc_scores_by_query.keys():
            continue
        dist, n_rel_ret_d = new_q_dist(qname, rdbq, retrieved_doc_scores_by_query)
        y.append(n_rel_ret_d)
        x.append(dist)
    return np.array(x), np.array(y)


def plot_scores_dist(dist, n_rel, relevance_labels):
    import matplotlib.pyplot as plt
    plt.plot(dist)
    # plt.axvline(n_rel, 0, 1, label='pyplot vertical line', color='r')
    markers_on = [i for i in range(len(relevance_labels)) if relevance_labels[i] > 0]
    plt.plot(dist, '-bD', markevery=markers_on)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('Position', fontsize=18)
    plt.ylabel('Normalized relevance score', fontsize=18)
    plt.show()
    # plt.ylabel('some numbers')
    plt.show()


def new_q_dist(qname, rdbq, retrieved_doc_scores_by_query, n_top=100):
    # print('ntop: {}'.format(n_top))
    doc_scores_bn = retrieved_doc_scores_by_query[qname]
    pred = np.array([doc_scores_bn[dn] for dn in doc_scores_bn.keys()])
    top_ranked_d = get_top_k_doc_names(doc_scores_bn, n_top)
    if qname not in rdbq.keys():
        num_rel_retr_docs = 0
    else:
        num_rel_retr_docs = sum([1 for dn in top_ranked_d if dn in rdbq[qname]])
        rel_labels = [1 if dn in rdbq[qname] else 0 for dn in top_ranked_d if dn in rdbq[qname]]
    doc_scores = pred[np.argsort(-pred)][0:min(len(pred), n_top)]
    # doc_scores = doc_scores / sum(doc_scores)
    doc_scores = list(doc_scores) + [0] * max(0, n_top - len(doc_scores))
    # plot_scores_dist(doc_scores, num_rel_retr_docs, rel_labels)
    return np.array(doc_scores), num_rel_retr_docs / len(doc_scores)


def get_top_k_doc_names(doc_scores_bn, nelements):
    dnames = []
    dscores = []
    for k, v in doc_scores_bn.items():
        dnames.append(k)
        dscores.append(v)
    dnames = np.array(dnames)
    dscores = np.array(dscores)

    return dnames[np.argsort(-dscores)][0:nelements]


def pred_w_prob_reg_model_batch(model_path, x_test, learning_rate):
    n_samples = 50
    tf.set_random_seed(0)
    tf.reset_default_graph()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config, graph=tf.get_default_graph()) as sess:
        tf.set_random_seed(0)
        model = ProbRegressor(0, np.array(x_test).shape[-1], learning_rate)
        sess.run(model.init_op)
        tf.set_random_seed(0)
        model.saver.restore(sess, model_path)
        # Make predictions.
        preds = np.mean(
            [sess.run(tf.squeeze(model.output), feed_dict={model.input_data: x_test}) for _ in range(n_samples)],
            axis=0)
        assert len(preds) == len(x_test)
        # sampled_prediction = [model.predict(np.array(x_test)) for i in range(n_samples)]
    return preds
