import json
import os
import pickle
import subprocess

from tqdm import tqdm


# kstemmer = krovetz.PyKrovetzStemmer()


def evaluate(rel_docs_by_qry, sim_scores_by_qry, qrels, measure, fn, run_name, res_output_folder):
    trec_eval_path = '../trec_eval-master'
    m_v, run_path = create_evaluate_ranking(fn, rel_docs_by_qry, sim_scores_by_qry, qrels, run_name,
                                            measure=measure, output_folder=res_output_folder, te_path=trec_eval_path)

    return m_v, run_path


def load_indri_stopwords():
    fpath = '../ExperimentalCollections/indri_stoplist_eng.txt'
    sws = []
    for line in open(fpath, 'r'):
        sws.append(line.strip())
    return sws


def save_json(model, output_path):
    with open(output_path, 'w') as outfile:
        json.dump(model, outfile)


def load_json(path):
    with open(path, 'r') as json_file:
        return json.load(json_file)


def save_model(model, output_path):
    with open(output_path, 'wb') as handle:
        pickle.dump(model, handle)
        handle.close()


def load_model(path):
    #  model = pickle.loads(open(path, 'rb').read())
    model = pickle.load(open(path, 'rb'), encoding='bytes')
    return model


def invert_wi(wi):
    iwi = {}
    for k, v in wi.items():
        iwi[v] = k
    return iwi


def create_ranking(output_file, rel_docs_by_qry, sim_scores_by_qry, run_name, separator=' '):
    """
    Computes a ranking in TREC format.
    :param output_file: output path for the run to create.
    :param rel_docs_by_qry: ordered list of ranked documents for each query. Dictionary (query_id, [doc_id1, ...])
    :param sim_scores_by_qry: similarity scores computed for each doc for each query.
    Dictionary (query_id, [sim_score_doc_1, ...])
    :param run_name: name of the system which computed the run.
    :param separator: separator between the columns of the trec runs. Space is default.
    :return: Nothing. It creates the run at the specified path.
    """
    out = open(output_file, 'w')
    for q, rd in rel_docs_by_qry.items():
        for i in range(len(rd)):
            dname = rd[i]
            sim_score = sim_scores_by_qry[q][i]
            line = str(q) + separator + 'Q0' + separator + str(dname) + separator + str(i) + separator + str(
                sim_score) + separator + run_name + '\n'
            out.write(line)
    out.close()


def run_trec_eval(trec_eval_path, qrels_file, run_to_eval, measure='map'):
    """
    Runs trec eval on the specified run computing the measure indicated (default is map).
    :param trec_eval_path: path to main trec eval folder (github version).
    :param qrels_file: qrels file to judge the input run.
    :param run_to_eval: input run to evaluate.
    :param measure: measure to compute.
    :return: the value of the computed measure. It returns -1 if there was an error.
    """
    # print('using the qrels file: %s' % qrels_file)
    command = os.path.join(trec_eval_path, 'trec_eval') + ' -m all_trec' + ' ' \
              + os.path.join(os.getcwd(), qrels_file) + ' ' \
              + os.path.join(os.getcwd(), run_to_eval)
    # print(command)
    (measure_line, err) = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True).communicate()
    measure_line = measure_line.decode("utf-8")
    measures = {}
    if len(measure_line) > 0:
        lines = measure_line.split('\n')
        for l in lines:
            data = l.split('\t')
            if len(data) >= 3:
                measures[data[0].strip()] = data[2]
        measure_value = measures[measure]
    else:
        print('Error evaluating the run: {}'.format(run_to_eval))
        measure_value = -1
    return float(measure_value)


def create_evaluate_ranking(suff, rel_docs_by_qry, sim_scores_by_qry, gt_file, prog_name,
                            output_folder=os.path.dirname(os.path.realpath(__file__)), measure='map', te_path=''):
    output_file = os.path.join(output_folder, prog_name + '_' + str(suff) + '.txt')
    out = open(output_file, 'w')
    for q, rd in rel_docs_by_qry.items():
        for i in range(len(rd)):
            dname = rd[i]
            sim_score = sim_scores_by_qry[q][i]
            line = str(q) + ' Q0 ' + str(dname) + ' ' + str(i) + ' ' + str(sim_score) + ' ' + prog_name + '\n'
            out.write(line)
    out.close()
    if len(te_path) > 0:
        trec_eval_path = te_path
    else:
        trec_eval_path = '/home/ims/albe/IR_Engines/trec_eval-master'
    map_v = run_trec_eval(trec_eval_path=trec_eval_path, run_to_eval=output_file,
                          qrels_file=gt_file, measure=measure)
    # print(output_file)
    return map_v, output_file


def get_file_paths_in_dir(main_dir):
    filepaths = set()
    for filename in tqdm(os.listdir(main_dir)):
        fp = os.path.join(main_dir, filename)
        fn = filename.split(r'.')[0]
        if os.path.isfile(fp):
            if fn[0] == '.':
                continue
        filepaths.add(fp)
    return filepaths


def eval_final_run(final_run, qrels_file):
    trec_eval_path = '../trec_eval-master'
    print()
    thresholds = [5, 10, 15, 20, 30, 100, 200, 500, 1000]
    for thr in thresholds:
        eval_v = run_trec_eval(trec_eval_path=trec_eval_path, run_to_eval=final_run, qrels_file=qrels_file,
                               measure='P_{}'.format(thr))
        print('p{}: {}'.format(thr, eval_v))

        eval_v = run_trec_eval(trec_eval_path=trec_eval_path, run_to_eval=final_run, qrels_file=qrels_file,
                               measure='recall_{}'.format(thr))
        print('recall{}: {}'.format(thr, eval_v))

        eval_v = run_trec_eval(trec_eval_path=trec_eval_path, run_to_eval=final_run, qrels_file=qrels_file,
                               measure='ndcg_cut_{}'.format(thr))
        print('ndcg{}: {}'.format(thr, eval_v))
        print()
