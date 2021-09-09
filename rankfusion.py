import numpy as np

import new_model
import read_data
from new_model import new_q_dist
from read_data import get_doc_scores_bq


def fuse_runs_w_sklearn_models(regressors, rdbq, test_qnames, run_paths, hidd_size, learning_rate, oracle=False):
    doc_scores_names_by_query_all = []
    retrieved_docs_by_query_all = []
    for run_path in run_paths:
        doc_scores_names_by_query_all.append(get_doc_scores_bq(run_path))
        retrieved_docs_by_query_all.append(get_doc_scores_bq(run_path))
    doc_scores_names_by_query_all = normalize_doc_scores_of_each_model_by_qry(doc_scores_names_by_query_all)
    sim_scores_by_qry = {}
    rel_docs_by_qry = {}
    qdists_by_run_id = {}
    qnames_by_run_id = {}
    true_rel_docs_all_runs_by_q = {}
    # print('preparing distributions to feed regression model')
    for qname in test_qnames:
        if qname not in true_rel_docs_all_runs_by_q.keys():
            true_rel_docs_all_runs_by_q[qname] = []
        skip_q = False
        for i in range(len(run_paths)):
            if i not in qdists_by_run_id.keys():
                qdists_by_run_id[i] = []
                qnames_by_run_id[i] = []

            if qname not in retrieved_docs_by_query_all[i].keys():
                skip_q = True
                break
            # run_path = run_paths[i]
            tdist, true_n_rel_docs = new_q_dist(qname, rdbq, retrieved_docs_by_query_all[i])
            qdists_by_run_id[i].append(tdist)
            qnames_by_run_id[i].append(qname)
            true_rel_docs_all_runs_by_q[qname].append(true_n_rel_docs)
            # nrel_d = pred_w_prob_reg_model(regressors[i], np.array(tdist))
        if skip_q:
            continue

    # print('predicting relevant documents number')
    true_scores_all = []
    pred_scores_all = []

    n_rel_docs_all_runs_by_q = {}
    for j in range(len(run_paths)):
        predicted_scores = []
        trues_scores = []

        dists = qdists_by_run_id[j]
        qnames = qnames_by_run_id[j]

        if not oracle:
            nrel_pred = regressors[j].predict(dists)
            vars = None
        else:
            nrel_pred = None
            vars = None

        for k in range(len(qnames)):
            if qnames[k] not in n_rel_docs_all_runs_by_q.keys():
                n_rel_docs_all_runs_by_q[qnames[k]] = []
            if oracle:
                true_n_rel_docs = true_rel_docs_all_runs_by_q[qnames[k]][j]
                n_rel_docs_all_runs_by_q[qnames[k]].append(true_n_rel_docs)
            else:
                n_rel_docs_all_runs_by_q[qnames[k]].append(nrel_pred[k])
            predicted_scores.append(nrel_pred[k])
            trues_scores.append(true_rel_docs_all_runs_by_q[qnames[k]][j])
        pred_scores_all.append(predicted_scores)
        true_scores_all.append(trues_scores)
    for qname in test_qnames:
        dscores_fusion_dict = {}
        for i in range(len(doc_scores_names_by_query_all)):
            nreld = n_rel_docs_all_runs_by_q[qname][i]
            dsdict = doc_scores_names_by_query_all[i]
            tdscores = []
            tdnames = []

            for dn, dscore in dsdict[qname].items():
                tdnames.append(dn)
                tdscores.append(dscore)

            tdnames = np.array(tdnames)
            tdscores = np.array(tdscores)
            # first select only the top presumed relevant documents from each run
            tdnames = tdnames[np.argsort(-tdscores)]  # [0:nreld]
            tdscores = tdscores[np.argsort(-tdscores)]  # [0:nreld]
            for j in range(len(tdnames)):
                dn = tdnames[j]
                dscore = tdscores[j]
                if dn not in dscores_fusion_dict.keys():
                    dscores_fusion_dict[dn] = []
                # dscores_fusion_dict[dn].append(dscore * nreld / (j + 1))
                dscores_fusion_dict[dn].append(dscore * nreld)

        dnames = []
        dscores = []
        for k, v in dscores_fusion_dict.items():
            dnames.append(k)
            dscores.append(sum(v))

        dnames = np.array(dnames)
        dscores = np.array(dscores)

        dnames = dnames[np.argsort(-dscores)]
        dscores = dscores[np.argsort(-dscores)]

        sim_scores_by_qry[qname] = dscores[0:50]
        rel_docs_by_qry[qname] = dnames[0:50]

    return sim_scores_by_qry, rel_docs_by_qry, pred_scores_all, true_scores_all


def fuse_runs(regressors, rdbq, test_qnames, run_paths, learning_rate, oracle=False):
    doc_scores_names_by_query_all = []
    retrieved_docs_by_query_all = []
    for run_path in run_paths:
        doc_scores_names_by_query_all.append(get_doc_scores_bq(run_path))
        retrieved_docs_by_query_all.append(get_doc_scores_bq(run_path))
    doc_scores_names_by_query_all = normalize_doc_scores_of_each_model_by_qry(doc_scores_names_by_query_all)
    sim_scores_by_qry = {}
    rel_docs_by_qry = {}
    qdists_by_run_id = {}
    qnames_by_run_id = {}
    true_rel_docs_all_runs_by_q = {}
    # print('preparing distributions to feed regression model')
    for qname in test_qnames:
        if qname not in true_rel_docs_all_runs_by_q.keys():
            true_rel_docs_all_runs_by_q[qname] = []
        skip_q = False
        for i in range(len(run_paths)):
            if i not in qdists_by_run_id.keys():
                qdists_by_run_id[i] = []
                qnames_by_run_id[i] = []

            if qname not in retrieved_docs_by_query_all[i].keys():
                skip_q = True
                break
            # run_path = run_paths[i]
            tdist, true_n_rel_docs = new_q_dist(qname, rdbq, retrieved_docs_by_query_all[i])
            qdists_by_run_id[i].append(tdist)
            qnames_by_run_id[i].append(qname)
            true_rel_docs_all_runs_by_q[qname].append(true_n_rel_docs)
            # nrel_d = pred_w_prob_reg_model(regressors[i], np.array(tdist))
        if skip_q:
            continue

    # print('predicting relevant documents number')
    true_scores_all = []
    pred_scores_all = []

    n_rel_docs_all_runs_by_q = {}
    for j in range(len(run_paths)):
        predicted_scores = []
        trues_scores = []

        dists = qdists_by_run_id[j]
        qnames = qnames_by_run_id[j]

        if not oracle:
            nrel_pred = new_model.pred_w_prob_reg_model_batch(regressors[j], dists, learning_rate)
        else:
            nrel_pred = None

        for k in range(len(qnames)):
            if qnames[k] not in n_rel_docs_all_runs_by_q.keys():
                n_rel_docs_all_runs_by_q[qnames[k]] = []
            if oracle:
                true_n_rel_docs = true_rel_docs_all_runs_by_q[qnames[k]][j]
                n_rel_docs_all_runs_by_q[qnames[k]].append(true_n_rel_docs)
            else:
                n_rel_docs_all_runs_by_q[qnames[k]].append(nrel_pred[k])
            if not oracle:
                predicted_scores.append(nrel_pred[k])
            else:
                predicted_scores.append(true_rel_docs_all_runs_by_q[qnames[k]][j])
            trues_scores.append(true_rel_docs_all_runs_by_q[qnames[k]][j])
        pred_scores_all.append(predicted_scores)
        true_scores_all.append(trues_scores)
    for qname in test_qnames:
        dscores_fusion_dict = {}
        for i in range(len(doc_scores_names_by_query_all)):
            nreld = n_rel_docs_all_runs_by_q[qname][i]
            dsdict = doc_scores_names_by_query_all[i]
            tdscores = []
            tdnames = []

            for dn, dscore in dsdict[qname].items():
                tdnames.append(dn)
                tdscores.append(dscore)

            tdnames = np.array(tdnames)
            tdscores = np.array(tdscores)
            # first select only the top presumed relevant documents from each run
            tdnames = tdnames[np.argsort(-tdscores)]  # [0:nreld]
            tdscores = tdscores[np.argsort(-tdscores)]  # [0:nreld]
            for j in range(len(tdnames)):
                dn = tdnames[j]
                dscore = tdscores[j]
                if dn not in dscores_fusion_dict.keys():
                    dscores_fusion_dict[dn] = []
                # dscores_fusion_dict[dn].append(dscore * nreld / (j + 1))
                dscores_fusion_dict[dn].append(dscore * nreld)

        dnames = []
        dscores = []
        for k, v in dscores_fusion_dict.items():
            dnames.append(k)
            # dscores.append(sum(v) * sum([1 for val in v if val != 0]))
            dscores.append(sum(v))

        dnames = np.array(dnames)
        dscores = np.array(dscores)

        dnames = dnames[np.argsort(-dscores)]
        dscores = dscores[np.argsort(-dscores)]

        sim_scores_by_qry[qname] = dscores[0:50]
        rel_docs_by_qry[qname] = dnames[0:50]

    return sim_scores_by_qry, rel_docs_by_qry, pred_scores_all, true_scores_all


def sel_best(regressors, rdbq, test_qnames, run_paths, learning_rate, oracle=False):
    topic_runs_scores_pair = {}
    doc_scores_names_by_query_all = []
    retrieved_docs_by_query_all = []
    for run_path in run_paths:
        doc_scores_names_by_query_all.append(get_doc_scores_bq(run_path))
        retrieved_docs_by_query_all.append(get_doc_scores_bq(run_path))
    doc_scores_names_by_query_all = normalize_doc_scores_of_each_model_by_qry(doc_scores_names_by_query_all)
    sim_scores_by_qry = {}
    rel_docs_by_qry = {}
    qdists_by_run_id = {}
    qnames_by_run_id = {}
    true_rel_docs_all_runs_by_q = {}
    # print('preparing distributions to feed regression model')
    for qname in test_qnames:
        if qname not in true_rel_docs_all_runs_by_q.keys():
            true_rel_docs_all_runs_by_q[qname] = []
        skip_q = False
        for i in range(len(run_paths)):
            if i not in qdists_by_run_id.keys():
                qdists_by_run_id[i] = []
                qnames_by_run_id[i] = []

            if qname not in retrieved_docs_by_query_all[i].keys():
                skip_q = True
                break
            # run_path = run_paths[i]
            tdist, true_n_rel_docs = new_q_dist(qname, rdbq, retrieved_docs_by_query_all[i])

            qdists_by_run_id[i].append(tdist)
            qnames_by_run_id[i].append(qname)
            true_rel_docs_all_runs_by_q[qname].append(true_n_rel_docs)
            # nrel_d = pred_w_prob_reg_model(regressors[i], np.array(tdist))
        if skip_q:
            continue

    # print('predicting relevant documents number')
    true_scores_all = []
    pred_scores_all = []

    n_rel_docs_all_runs_by_q = {}
    for j in range(len(run_paths)):
        predicted_scores = []
        trues_scores = []

        dists = qdists_by_run_id[j]
        qnames = qnames_by_run_id[j]

        if not oracle:
            nrel_pred = new_model.pred_w_prob_reg_model_batch(regressors[j], dists, learning_rate)
        else:
            nrel_pred = None

        for k in range(len(qnames)):
            if qnames[k] not in n_rel_docs_all_runs_by_q.keys():
                n_rel_docs_all_runs_by_q[qnames[k]] = []
            if oracle:
                true_n_rel_docs = true_rel_docs_all_runs_by_q[qnames[k]][j]
                n_rel_docs_all_runs_by_q[qnames[k]].append(true_n_rel_docs)
            else:
                n_rel_docs_all_runs_by_q[qnames[k]].append(nrel_pred[k])
            if not oracle:
                predicted_scores.append(nrel_pred[k])
            else:
                predicted_scores.append(true_rel_docs_all_runs_by_q[qnames[k]][j])
            trues_scores.append(true_rel_docs_all_runs_by_q[qnames[k]][j])
        pred_scores_all.append(predicted_scores)
        true_scores_all.append(trues_scores)
    for qname in test_qnames:
        dscores_fusion_dict = {}
        for i in range(len(doc_scores_names_by_query_all)):
            nreld = n_rel_docs_all_runs_by_q[qname][i]
            dsdict = doc_scores_names_by_query_all[i]
            tdscores = []
            tdnames = []

            for dn, dscore in dsdict[qname].items():
                tdnames.append(dn)
                tdscores.append(dscore)

            tdnames = np.array(tdnames)
            tdscores = np.array(tdscores)
            # first select only the top presumed relevant documents from each run
            tdnames = tdnames[np.argsort(-tdscores)]  # [0:nreld]
            tdscores = tdscores[np.argsort(-tdscores)]  # [0:nreld]
            for j in range(len(tdnames)):
                dn = tdnames[j]
                dscore = tdscores[j]
                if dn not in dscores_fusion_dict.keys():
                    dscores_fusion_dict[dn] = []
                # dscores_fusion_dict[dn].append(dscore * nreld / (j + 1))
                dscores_fusion_dict[dn].append((dscore, nreld, i))

        dnames = []
        dscores = []
        for k, v in dscores_fusion_dict.items():
            dnames.append(k)
            # dscores.append(sum(v) * sum([1 for val in v if val != 0]))
            max_val_index = np.argmax([pair[1] for pair in v])
            selected_run_index = v[max_val_index][2]
            dscores.append(v[max_val_index][0])

        dnames = np.array(dnames)
        dscores = np.array(dscores)

        dnames = dnames[np.argsort(-dscores)]
        dscores = dscores[np.argsort(-dscores)]

        sim_scores_by_qry[qname] = dscores[0:50]
        rel_docs_by_qry[qname] = dnames[0:50]

    return sim_scores_by_qry, rel_docs_by_qry, pred_scores_all, true_scores_all, n_rel_docs_all_runs_by_q


def normalize_doc_scores_of_each_model_by_qry(doc_scores_names_by_query_all):
    for i in range(len(doc_scores_names_by_query_all)):
        curr_model_scores = doc_scores_names_by_query_all[i]
        for k, doc_scores_by_name in curr_model_scores.items():
            if np.max(list(doc_scores_by_name.values())) < 0:
                curr_rl_len = len(doc_scores_by_name.items())
                for nk, v in doc_scores_by_name.items():
                    new_v = curr_rl_len + v  # v is always negative
                    doc_scores_by_name[nk] = new_v
            max_score = max(doc_scores_by_name.values())
            for dname in doc_scores_by_name.keys():
                doc_scores_by_name[dname] = doc_scores_by_name[dname] / max_score
            curr_model_scores[k] = doc_scores_by_name
        doc_scores_names_by_query_all[i] = curr_model_scores
    return doc_scores_names_by_query_all


def combsum(test_qnames, run_paths):
    doc_scores_names_by_query_all = []
    retrieved_docs_by_query_all = []
    for run_path in run_paths:
        doc_scores_names_by_query_all.append(get_doc_scores_bq(run_path))
        retrieved_docs_by_query_all.append(get_doc_scores_bq(run_path))
    doc_scores_names_by_query_all = normalize_doc_scores_of_each_model_by_qry(doc_scores_names_by_query_all)
    sim_scores_by_qry = {}
    rel_docs_by_qry = {}
    for qname in test_qnames:
        skip_q = False
        for i in range(len(run_paths)):
            if qname not in retrieved_docs_by_query_all[i].keys():
                skip_q = True
                break
        if skip_q:
            continue

        dscores_fusion_dict = {}
        for i in range(len(doc_scores_names_by_query_all)):
            dsdict = doc_scores_names_by_query_all[i]
            tdscores = []
            tdnames = []

            for dn, dscore in dsdict[qname].items():
                tdnames.append(dn)
                tdscores.append(dscore)

            tdnames = np.array(tdnames)
            tdscores = np.array(tdscores)
            # first select only the top presumed relevant documents from each run
            tdnames = tdnames[np.argsort(-tdscores)]
            tdscores = tdscores[np.argsort(-tdscores)]
            for j in range(len(tdnames)):
                dn = tdnames[j]
                dscore = tdscores[j]
                if dn not in dscores_fusion_dict.keys():
                    dscores_fusion_dict[dn] = []
                dscores_fusion_dict[dn].append(dscore)

        dnames = []
        dscores = []
        for k, v in dscores_fusion_dict.items():
            dnames.append(k)
            dscores.append(sum(v))

        dnames = np.array(dnames)
        dscores = np.array(dscores)

        dnames = dnames[np.argsort(-dscores)]
        dscores = dscores[np.argsort(-dscores)]

        sim_scores_by_qry[qname] = dscores
        rel_docs_by_qry[qname] = dnames

    return sim_scores_by_qry, rel_docs_by_qry


def fit_gaussian_to_data(rel_scores):
    from scipy.optimize import curve_fit
    from scipy import asarray as ar
    mean = np.mean(rel_scores)
    sigma = np.std(rel_scores)
    try:
        popt, _ = curve_fit(gauss, ar(range(len(rel_scores))), rel_scores, p0=[1, mean, sigma], maxfev=2000)
    except RuntimeError:
        popt = [1, mean, sigma]
    return popt[0], popt[1], popt[2]


def gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def exponential(x, mean):
    rval = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] > 0:
            rval[i] = mean * np.exp(-(mean * x[i]))
    return rval


def fit_exp_to_data(rel_scores):
    from scipy.optimize import curve_fit
    from scipy import asarray as ar
    mean = np.mean(rel_scores)
    try:
        popt, _ = curve_fit(exponential, ar(range(len(rel_scores))), rel_scores, p0=[mean], maxfev=2000)
    except RuntimeError:
        popt = [mean]
    return popt[0]


def meta(test_qnames, run_paths, train_qnames, rdbq, collection):
    doc_scores_names_by_query_all = []
    retrieved_docs_by_query_all = []
    for run_path in run_paths:
        doc_scores_names_by_query_all.append(get_doc_scores_bq(run_path))
        retrieved_docs_by_query_all.append(get_doc_scores_bq(run_path))

    # doc_scores_names_by_query_all = normalize_doc_scores_of_each_model_by_qry(doc_scores_names_by_query_all)
    sim_scores_by_qry = {}
    rel_docs_by_qry = {}
    # prel = 0
    # pnrel = 0
    for qname in test_qnames:
        skip_q = False
        for i in range(len(run_paths)):
            if qname not in retrieved_docs_by_query_all[i].keys():
                skip_q = True
                break
        if skip_q:
            continue

        dscores_fusion_dict = {}
        for i in range(len(doc_scores_names_by_query_all)):
            dsdict = doc_scores_names_by_query_all[i]
            tdscores = []
            tdnames = []
            x, y = new_model.compute_training_distributions(read_data.get_doc_scores_bq(run_paths[i]), rdbq,
                                                            train_qnames, collection)
            prel = np.mean(y)
            pnrel = 1 - prel
            for dn, dscore in dsdict[qname].items():
                tdnames.append(dn)
                tdscores.append(dscore)

            tdnames = np.array(tdnames)
            tdscores = np.array(tdscores)
            # first select only the top presumed relevant documents from each run
            tdnames = tdnames[np.argsort(-tdscores)]
            tdscores = tdscores[np.argsort(-tdscores)]

            gauss_params = fit_gaussian_to_data(tdscores)
            exp_param = fit_exp_to_data(tdscores)

            gauss_values = [gauss(score, gauss_params[0], gauss_params[1], gauss_params[2]) for score in tdscores]
            exp_values = [exponential([score], exp_param) for score in tdscores]
            posterior_values = [((gauss_values[i] * prel) / (gauss_values[i] * prel + exp_values[i] * pnrel))[0] for i in
                                range(len(tdscores))]
            assert len(posterior_values) == len(tdscores)
            tdscores *= posterior_values
            for j in range(len(tdnames)):
                dn = tdnames[j]
                dscore = tdscores[j]
                if dn not in dscores_fusion_dict.keys():
                    dscores_fusion_dict[dn] = []
                dscores_fusion_dict[dn].append(dscore)

        dnames = []
        dscores = []
        for k, v in dscores_fusion_dict.items():
            dnames.append(k)
            dscores.append(sum(v))

        dnames = np.array(dnames)
        dscores = np.array(dscores)

        dnames = dnames[np.argsort(-dscores)]
        dscores = dscores[np.argsort(-dscores)]

        sim_scores_by_qry[qname] = dscores
        rel_docs_by_qry[qname] = dnames

    return sim_scores_by_qry, rel_docs_by_qry


def combmnz(test_qnames, run_paths):
    n_systems = len(run_paths)
    doc_scores_names_by_query_all = []
    retrieved_docs_by_query_all = []
    for run_path in run_paths:
        doc_scores_names_by_query_all.append(get_doc_scores_bq(run_path))
        retrieved_docs_by_query_all.append(get_doc_scores_bq(run_path))
    doc_scores_names_by_query_all = normalize_doc_scores_of_each_model_by_qry(doc_scores_names_by_query_all)
    sim_scores_by_qry = {}
    rel_docs_by_qry = {}
    for qname in test_qnames:
        skip_q = False
        for i in range(len(run_paths)):
            if qname not in retrieved_docs_by_query_all[i].keys():
                skip_q = True
                break
        if skip_q:
            continue

        dscores_fusion_dict = {}
        for i in range(len(doc_scores_names_by_query_all)):
            dsdict = doc_scores_names_by_query_all[i]
            tdscores = []
            tdnames = []

            for dn, dscore in dsdict[qname].items():
                tdnames.append(dn)
                tdscores.append(dscore)

            tdnames = np.array(tdnames)
            tdscores = np.array(tdscores)
            # first select only the top presumed relevant documents from each run
            tdnames = tdnames[np.argsort(-tdscores)]
            tdscores = tdscores[np.argsort(-tdscores)]
            for j in range(len(tdnames)):
                dn = tdnames[j]
                dscore = tdscores[j]
                if dn not in dscores_fusion_dict.keys():
                    dscores_fusion_dict[dn] = []
                dscores_fusion_dict[dn].append(dscore)

        dnames = []
        dscores = []
        for k, v in dscores_fusion_dict.items():
            dnames.append(k)
            dscores.append(sum(v) * sum([1 for val in v if val != 0]))

        dnames = np.array(dnames)
        dscores = np.array(dscores)

        dnames = dnames[np.argsort(-dscores)]
        dscores = dscores[np.argsort(-dscores)]

        sim_scores_by_qry[qname] = dscores
        rel_docs_by_qry[qname] = dnames

    return sim_scores_by_qry, rel_docs_by_qry
