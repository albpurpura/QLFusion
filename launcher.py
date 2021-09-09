import os

import read_data
from main import perform_rankfusion

collections = ['Robust04']
hidden_sizes = [2, 4, 8]
learning_rates = [1e-3, 1e-2, 0.1]
batch_sizes = [1, 2, 4, 8]
epochs = [300]
test_types = ['QLFusion']

project_dir = '/nfsd/gracedata/purpura/QLRF'
# project_dir = '/Users/alberto/PycharmProjects/QLRF'
# python_path = '/Users/alberto/anaconda3/envs/tf/bin/python'
main_file_path = project_dir + '/main.py'
logs_dir = project_dir
python_path = '/nfsd/gracedata/purpura/venv/bin/python'

for collection in collections:
    for hidden_size in hidden_sizes:
        for learning_rate in learning_rates:
            for batch_size in batch_sizes:
                for n_epochs in epochs:
                    for test_type in test_types:
                        log = './log_coll={}_hidd_size={}_lr={}_batch_size={}_epochs={}_test={}.txt'. \
                            format(collection, hidden_size, learning_rate, batch_size, n_epochs, test_type)
                        command = 'cd {} & {} {} ' \
                                  '--collection {} ' \
                                  '--hidden_size {} ' \
                                  '--learning_rate {} ' \
                                  '--batch_size {} ' \
                                  '--epochs {} ' \
                                  '--test_type {} > {} &'.format(project_dir, python_path, main_file_path, collection,
                                                                 hidden_size, learning_rate,
                                                                 batch_size, n_epochs, test_type,
                                                                 os.path.join(logs_dir, log))
                        print(command)
                        print("sleep 5")
