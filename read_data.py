import os
import pickle

import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

from util import run_trec_eval, eval_final_run


def get_rel_docs_by_query(qrels_file):
    rdbq = {}
    for line in open(qrels_file, 'r', encoding='utf-8'):
        data = line.split()
        qid = data[0]
        did = data[2]
        rel_j = int(data[3])
        if rel_j > 0:
            if qid not in rdbq.keys():
                rdbq[qid] = []
            rdbq[qid].append(did)
    return rdbq


def get_doc_scores_bq(run_path):
    ranked_docs_by_query = {}
    for line in open(run_path, 'r'):
        data = line.split()
        qid = data[0]
        did = data[2]
        dscore = data[4]
        if qid not in ranked_docs_by_query.keys():
            ranked_docs_by_query[qid] = {}
        ranked_docs_by_query[qid][did] = float(dscore)
    return ranked_docs_by_query


def get_collections_data(collection, n_folds=10):
    qrels_file = ''
    data_dir = ''
    if collection == 'TREC3':
        qrels_file = 'TREC-3/qrels.151-200.disk1-3.txt'
        data_dir = 'TREC-3/runs'
    elif collection == 'TREC5':
        qrels_file = 'TREC-5/qrels.251-300.disk2.disk4.txt'
        data_dir = 'TREC-5/runs'
    elif collection == 'CLEF':
        qrels_file = 'CLEF/qrel_abs_task2.txt'
        data_dir = 'CLEF/CLEF-TAR_2018_subtask2'
    elif collection == 'CDS14':
        qrels_file = 'CDS14/qrels_treceval_2014.txt'
        data_dir = 'CDS14/runs'
    elif collection == 'CDS15':
        qrels_file = 'CDS15/qrels_treceval_2015.txt'
        data_dir = 'CDS15/runs'
    elif collection == 'CDS16':
        qrels_file = 'CDS16/qrels_treceval_2016.txt'
        data_dir = 'CDS16/runs'
    elif collection == 'OHSUMED':
        qrels_file = './OHSUMED/qrels_orig_ohsumed.txt'
        data_dir = './OHSUMED/runs'
    elif collection == 'TREC-COVID-R5':
        qrels_file = '/Users/alberto/Downloads/TREC_COVID_round5_runs/trec_covid_r5_qrels.txt'
        data_dir = '/Users/alberto/Downloads/TREC_COVID_round5_runs/runs'
    elif collection == 'TREC-DL-D':
        qrels_file = '/Users/alberto/Downloads/TREC_DL/qrels_doc_retr.txt'
        data_dir = '/Users/alberto/Downloads/TREC_DL/runs_doc'
    elif collection == 'TREC-DL-P':
        qrels_file = '/Users/alberto/Downloads/TREC_DL/qrels_passage_retr.txt'
        data_dir = '/Users/alberto/Downloads/TREC_DL/runs_passage'
    all_runs = get_runs(data_dir, qrels_file)

    print('selecting the following runs: {}'.format(all_runs))
    for f in tqdm(all_runs):
        print('run: {}'.format(f))
        eval_final_run(f, qrels_file)

    rdbq = get_rel_docs_by_query(qrels_file)
    qnames = list(rdbq.keys())
    qnames = np.array(qnames)

    kf = KFold(n_splits=n_folds, random_state=0, shuffle=True)
    train_test_splits_path = 'train_test_qnames_splits_' + collection
    if not os.path.isfile(train_test_splits_path):
        train_test_stuff = []
        for train_index, test_index in kf.split(qnames):
            train_test_stuff.append((train_index, test_index))

        train_test_qnames_by_fold = []
        for train_index, test_index in train_test_stuff:
            train_qnames = qnames[train_index]
            test_qnames = qnames[test_index]
            train_qnames = [q for q in train_qnames if q in rdbq.keys()]
            test_qnames = [q for q in test_qnames if q in rdbq.keys()]
            train_test_qnames_by_fold.append((train_qnames, test_qnames))
        pickle.dump(train_test_qnames_by_fold, open(train_test_splits_path, 'wb'))
    else:
        train_test_qnames_by_fold = pickle.load(open(train_test_splits_path, 'rb'))

    return train_test_qnames_by_fold, all_runs, qrels_file, rdbq


def get_runs(data_dir, qrels_file):
    seed = 0
    trec_eval_path = '../trec_eval-master'
    data_folder = os.path.join(os.getcwd(), data_dir)
    onlyfiles = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if
                 os.path.isfile(os.path.join(data_folder, f)) and not f.startswith('.')]
    np.random.seed(seed)
    # qrels_file = 'anserini_runs/Anserini_Robust04/qrels.txt'
    maps = []
    for f in tqdm(onlyfiles):
        value = run_trec_eval(trec_eval_path=trec_eval_path, run_to_eval=f, qrels_file=qrels_file,
                              measure='map')
        maps.append(value)
    all_runs = np.array(onlyfiles)[np.argsort(-np.array(maps))[:6]]  # 10 for trec3, 6 for trec 5
    # all_runs = np.array(onlyfiles)[np.argsort(-np.array(maps))[:3]]  # 10 for trec3, 6 for trec 5
    print('considering top {} runs'.format(len(all_runs)))
    return all_runs
