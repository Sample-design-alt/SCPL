# -*- coding: utf-8 -*-

import json
import os
import datetime

def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(dictionary)
    """
    if not os.path.exists(json_file):
        return {"piece_size": 0.2, "class_type": "3C"}

    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    return config_dict

def log_file(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    file2print_detail_train = open("{}/train_detail.log".format(log_dir), 'a+')

    print(datetime.datetime.now(), file=file2print_detail_train)
    print("Dataset\tTrain\tTest\tDimension\tClass\tSeed\tAcc_label\tAcc_unlabel\tEpoch_max",
          file=file2print_detail_train)
    file2print_detail_train.flush()

    file2print = open("{}/test.log".format(log_dir), 'a+')
    print(datetime.datetime.now(), file=file2print)
    print("Dataset\tAcc_mean\tAcc_std\tEpoch_max",
          file=file2print)
    file2print.flush()

    file2print_detail = open("{}/test_detail.log".format(log_dir), 'a+')
    print(datetime.datetime.now(), file=file2print_detail)
    print("Dataset\tTrain\tTest\tDimension\tClass\tSeed\tAcc_max\tEpoch_max",
          file=file2print_detail)
    file2print_detail.flush()
    return file2print_detail_train,file2print,file2print_detail


def convert_to_tuner_config(config, trial):
    search_space = config['search_space']
    config = config.copy()

    # Find and replace the hyper parameters defined in search_space
    for section, params in config.items():
        if section == 'search_space':
            continue

        for hp_name in config[section].keys():
            if hp_name in search_space.keys():
                if hp_name == 'loss_weights':
                    low = search_space[hp_name]['low']
                    high = search_space[hp_name]['high']
                    suggested_losses = [
                        trial.suggest_loguniform("lw_{}".format(lw), low=low,
                                                 high=high) for lw in
                        range(search_space[hp_name]['length'])]
                    config[section][hp_name] = suggested_losses
                    continue

                tuner_hp_type = search_space[hp_name]['type']
                if tuner_hp_type == 'categorical':
                    config[section][hp_name] = trial.suggest_categorical(hp_name,
                                                                         choices=search_space[hp_name]['choices'])
                    continue

                low = search_space[hp_name]['low']
                high = search_space[hp_name]['high']
                if tuner_hp_type == 'float':
                    config[section][hp_name] = trial.suggest_float(hp_name, low=low, high=high,
                                                                   step=search_space[hp_name]['step'])
                elif tuner_hp_type == 'int':
                    config[section][hp_name] = trial.suggest_int(hp_name, low=low, high=high,
                                                                 step=search_space[hp_name]['step'])
                elif tuner_hp_type == 'log':
                    config[section][hp_name] = trial.suggest_loguniform(hp_name, low=low, high=high)
                else:
                    raise ValueError("Expected the tuner hyper parameter to have type int or float")
    return config