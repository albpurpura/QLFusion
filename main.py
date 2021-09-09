import argparse
import logging
import os
import pickle
import uuid

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import new_model
import read_data
from rankfusion import fuse_runs, sel_best, combsum, combmnz, meta
from regression_analysis import compute_MSE
from util import evaluate, eval_final_run

flags = tf.app.flags
FLAGS = flags.FLAGS

"""
learning_rate : 5e-4
batch_size : 1
epochs : 500
collection : TREC3

learning_rate : 5e-4
batch_size : 8
epochs : 500
collection : Robust04
"""


def add_arguments(parser):
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4)  # 2 is best for TREC5, 4 is the best for TREC3 and CLEF
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--collection", type=str, default='CLEF')  # TREC-COVID-R5
    parser.add_argument("--test_type", type=str, default='QLFusion')  # combsum, combmnz, QLFusion
    # parser.add_argument("--test_type", type=str, default='META')  # combsum, combmnz, QLFusion
    # parser.add_argument("--test_type", type=str, default='combsum')  # combsum, combmnz, QLFusion
    # parser.add_argument("--test_type", type=str, default='combsum')  # combsum, combmnz, QLFusion


def perform_rankfusion(run_paths, rdbq, train_test_qnames_by_fold, rel_j, oracle, run_name, collection, lr=1e-3,
                       bs=8, ne=300, reg_models_by_fold=None):
    test_qnames_by_fold = []
    sim_scores_by_qry = {}
    rel_docs_by_qry = {}
    exp_tag = str(uuid.uuid4())
    for i, (train_qnames, test_qnames) in tqdm(enumerate(train_test_qnames_by_fold)):
        test_qnames_by_fold.append(test_qnames)

    if reg_models_by_fold is None:
        reg_models_by_fold = []
        all_best_models = []

        models_dir = os.getcwd() + '/saved_models_' + exp_tag
        os.makedirs(models_dir)
        for i, (train_qnames, test_qnames) in tqdm(enumerate(train_test_qnames_by_fold)):
            if not oracle:
                print('FOLD: %d' % i)
                reg_models = new_model.train_multiple_models(run_paths, rdbq, train_qnames, collection,
                                                             models_dir=models_dir, fold=str(i),
                                                             best_models_prev_folds=all_best_models, seed=0,
                                                             learning_rate=lr, batch_size=bs, n_epochs=ne)
            else:
                reg_models = [None] * len(test_qnames_by_fold)
            reg_models_by_fold.append(reg_models)
            all_best_models.extend(reg_models)
            # test_qnames_by_fold.append(test_qnames)
    pred_scores_all = []
    true_scores_all = []
    for i in range(len(test_qnames_by_fold)):
        ssbq_part, rdbq_part, pred_scores, true_scores = fuse_runs(reg_models_by_fold[i], rdbq, test_qnames_by_fold[i],
                                                                   run_paths, learning_rate=lr,
                                                                   oracle=oracle)
        pred_scores_all.append(pred_scores)
        true_scores_all.append(true_scores)
        for k, v in ssbq_part.items():
            sim_scores_by_qry[k] = v
        for k, v in rdbq_part.items():
            rel_docs_by_qry[k] = v
    os.makedirs('results_' + exp_tag)
    measure, final_run_path = evaluate(rel_docs_by_qry, sim_scores_by_qry, rel_j, 'recall_10', 'all', run_name,
                                       'results_' + exp_tag)
    # if measure > best_measure:
    #     print('new best recall_10 = %2.4f with c = %2.3f' % (measure, c_value))
    #     best_measure = measure
    #     best_final_run_path = final_run_path

    print('Best models by fold:')
    for i, mods in enumerate(reg_models_by_fold):
        print('Fold {}, models={}'.format(i, mods))
    print('exp tag: ' + exp_tag)
    print('final run path: {}'.format(final_run_path))
    return final_run_path, pred_scores_all, true_scores_all
    # return best_final_run_path


def select_best(run_paths, rdbq, train_test_qnames_by_fold, rel_j, oracle, run_name, collection, lr=1e-3,
                bs=8, ne=300, reg_models_by_fold=None):
    test_qnames_by_fold = []
    sim_scores_by_qry = {}
    rel_docs_by_qry = {}
    exp_tag = str(uuid.uuid4())
    for i, (train_qnames, test_qnames) in tqdm(enumerate(train_test_qnames_by_fold)):
        test_qnames_by_fold.append(test_qnames)

    if reg_models_by_fold is None:
        reg_models_by_fold = []
        all_best_models = []

        models_dir = os.getcwd() + '/saved_models_' + exp_tag
        os.makedirs(models_dir)
        for i, (train_qnames, test_qnames) in tqdm(enumerate(train_test_qnames_by_fold)):
            if not oracle:
                print('FOLD: %d' % i)
                reg_models = new_model.train_multiple_models(run_paths, rdbq, train_qnames, collection,
                                                             models_dir=models_dir, fold=str(i),
                                                             best_models_prev_folds=all_best_models, seed=0,
                                                             learning_rate=lr, batch_size=bs, n_epochs=ne)
            else:
                reg_models = [None] * len(test_qnames_by_fold)
            reg_models_by_fold.append(reg_models)
            all_best_models.extend(reg_models)
            # test_qnames_by_fold.append(test_qnames)
    pred_scores_all = []
    true_scores_all = []
    all_n_rel_docs_all_runs_by_q = {}
    for i in range(len(test_qnames_by_fold)):
        ssbq_part, rdbq_part, pred_scores, true_scores, n_rel_docs_all_runs_by_q = sel_best(reg_models_by_fold[i], rdbq,
                                                                                            test_qnames_by_fold[i],
                                                                                            run_paths, learning_rate=lr,
                                                                                            oracle=oracle)
        for k in n_rel_docs_all_runs_by_q.keys():
            all_n_rel_docs_all_runs_by_q[k] = n_rel_docs_all_runs_by_q[k]
        pred_scores_all.append(pred_scores)
        true_scores_all.append(true_scores)
        for k, v in ssbq_part.items():
            sim_scores_by_qry[k] = v
        for k, v in rdbq_part.items():
            rel_docs_by_qry[k] = v
    os.makedirs('results_' + exp_tag)
    measure, final_run_path = evaluate(rel_docs_by_qry, sim_scores_by_qry, rel_j, 'recall_10', 'all', run_name,
                                       'results_' + exp_tag)
    # if measure > best_measure:
    #     print('new best recall_10 = %2.4f with c = %2.3f' % (measure, c_value))
    #     best_measure = measure
    #     best_final_run_path = final_run_path

    print('Best models by fold:')
    for i, mods in enumerate(reg_models_by_fold):
        print('Fold {}, models={}'.format(i, mods))
    print('exp tag: ' + exp_tag)
    print('final run path: {}'.format(final_run_path))

    return final_run_path, pred_scores_all, true_scores_all, all_n_rel_docs_all_runs_by_q


def perform_combsum(all_runs, train_test_qnames_by_fold, rel_j, run_name):
    sim_scores_by_qry = {}
    rel_docs_by_qry = {}
    for i, (train_qnames, test_qnames) in tqdm(enumerate(train_test_qnames_by_fold)):
        ssbq_part, rdbq_part = combsum(test_qnames, all_runs)
        for k, v in ssbq_part.items():
            sim_scores_by_qry[k] = v
        for k, v in rdbq_part.items():
            rel_docs_by_qry[k] = v
    measure, final_run_path = evaluate(rel_docs_by_qry, sim_scores_by_qry, rel_j, 'recall_10', 'all', run_name,
                                       'results')
    return final_run_path


def perform_meta(all_runs, train_test_qnames_by_fold, rel_j, run_name, rdbq, collection):
    sim_scores_by_qry = {}
    rel_docs_by_qry = {}
    for i, (train_qnames, test_qnames) in tqdm(enumerate(train_test_qnames_by_fold)):
        ssbq_part, rdbq_part = meta(test_qnames, all_runs, train_qnames, rdbq, collection)
        for k, v in ssbq_part.items():
            sim_scores_by_qry[k] = v
        for k, v in rdbq_part.items():
            rel_docs_by_qry[k] = v
    measure, final_run_path = evaluate(rel_docs_by_qry, sim_scores_by_qry, rel_j, 'recall_10', 'all', run_name,
                                       'results')
    return final_run_path


def perform_combmnz(all_runs, train_test_qnames_by_fold, rel_j, run_name):
    sim_scores_by_qry = {}
    rel_docs_by_qry = {}
    for i, (train_qnames, test_qnames) in tqdm(enumerate(train_test_qnames_by_fold)):
        ssbq_part, rdbq_part = combmnz(test_qnames, all_runs)
        for k, v in ssbq_part.items():
            sim_scores_by_qry[k] = v
        for k, v in rdbq_part.items():
            rel_docs_by_qry[k] = v
    measure, final_run_path = evaluate(rel_docs_by_qry, sim_scores_by_qry, rel_j, 'recall_10', 'all', run_name,
                                       'results')
    return final_run_path


def run():
    models_dir = './saved_models/'
    # reg_models_by_fold_by_coll = {'Robust04': Robust04_best_models, 'WP': WP_best_models, 'GOV2': GOV2_best_models}
    reg_models_by_fold_by_coll = {'CLEF': None, 'TREC3': None, 'TREC5': None, 'CDS14': None, 'CDS15': None,
                                  'CDS16': None, 'OHSUMED': None, 'TREC-COVID-R5': None, 'TREC-DL-D': None,
                                  'TREC-DL-P': None}
    # reg_models_by_fold_by_coll = all_best_models
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

    arg_parser = argparse.ArgumentParser()
    add_arguments(arg_parser)
    FLAGS, unparsed = arg_parser.parse_known_args()
    for arg in vars(FLAGS):
        print(arg, ":", getattr(FLAGS, arg))

    oracle = True

    run_name = 'run.det_model_with_KLdiv.' + FLAGS.collection + '.' + str(oracle) + '.' + FLAGS.test_type
    train_test_qnames_by_fold, all_runs, qrels_file, rdbq = read_data.get_collections_data(FLAGS.collection)
    print('AVG # rel docs per query: {}'.format(np.mean([len(v) for v in rdbq.values()])))
    # exit()

    final_run = None
    if FLAGS.test_type == 'QLFusion':
        final_run, pred_scores, true_scores = perform_rankfusion(all_runs, rdbq, train_test_qnames_by_fold, qrels_file,
                                                                 oracle, run_name,
                                                                 FLAGS.collection,
                                                                 FLAGS.learning_rate, FLAGS.batch_size,
                                                                 FLAGS.epochs,
                                                                 reg_models_by_fold=reg_models_by_fold_by_coll[
                                                                     FLAGS.collection])
        if not oracle:
            pickle.dump((pred_scores, true_scores),
                        open('preds_and_true_det_model_w_kl_loss/' + FLAGS.collection + '_preds_and_true.pkl', 'wb'))
            compute_MSE(pred_scores, true_scores)
    elif FLAGS.test_type == 'SelectBest':
        final_run, pred_scores, true_scores, model_run_preds_by_topic = select_best(all_runs, rdbq,
                                                                                    train_test_qnames_by_fold,
                                                                                    qrels_file,
                                                                                    oracle, run_name,
                                                                                    FLAGS.collection,
                                                                                    FLAGS.learning_rate,
                                                                                    FLAGS.batch_size,
                                                                                    FLAGS.epochs,
                                                                                    reg_models_by_fold=
                                                                                    reg_models_by_fold_by_coll[
                                                                                        FLAGS.collection])
        # if not oracle:
        # pickle.dump(model_run_preds_by_topic,
        #             open('model_run_preds_by_topic/' + FLAGS.collection + '_preds_by_run_dict.pkl', 'wb'))
    elif FLAGS.test_type == 'combsum':
        final_run = perform_combsum(all_runs, train_test_qnames_by_fold, qrels_file, run_name)
    elif FLAGS.test_type == 'META':
        final_run = perform_meta(all_runs, train_test_qnames_by_fold, qrels_file, run_name, rdbq, FLAGS.collection)
    elif FLAGS.test_type == 'combmnz':
        final_run = perform_combmnz(all_runs, train_test_qnames_by_fold, qrels_file, run_name)

    print('collection: %s, test type: %s, oracle: %s' % (FLAGS.collection, FLAGS.test_type, str(oracle)))
    eval_final_run(final_run, qrels_file)


if __name__ == '__main__':
    run()
