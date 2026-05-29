import configparser
import os
import shutil
import time


# ----------------------------------------------------
#          FILE CONTAINING UTILITY FUNCTIONS
# ----------------------------------------------------


def read_config(cfg_file):
    """
    Loads parameters for execution from a configuration file.
    :param cfg_file: the path to the configuration file
    :return: a dictionary
    """
    # Loading Conf file
    if not os.path.exists(cfg_file):
        return None
    config = configparser.ConfigParser()
    config.read(cfg_file)

    # Setting up variables
    cfg_info = {}
    for section in config.sections():
        cfg_info[section] = {}
        for option in config.options(section):
            cfg_info[option] = config.get(section, option)
            if "," in cfg_info[option]:
                cfg_info[option] = [x.strip() for x in cfg_info[option].split(',')]
            elif cfg_info[option] in ["True", "False", "true", "false", "t", "f", "T", "F"]:
                cfg_info[option] = True if cfg_info[option].upper().startswith("T") else False

    return cfg_info


def current_ms():
    """
    Reports the current time in milliseconds
    :return: long int
    """
    return round(time.time() * 1000)


def print_scores(to_print, analysis_tag:str, output_folder, filename):
    """
    Prints scores of test runs to file specified in the CFG
    :param to_print: data to print
    :param output_folder: folder in which outputs are placed
    :param filename: name of the output file
    :return:
    """
    output_folder = output_folder.replace('"', '')
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    csv_file = os.path.join(output_folder, filename).replace('"', '')
    if not os.path.exists(csv_file):
        with open(csv_file, 'w') as f:
            f.write("timeseries,supervised,analysis_tag,clf_name,tp,fp,fn,tn,mcc,acc,recall,test_time,test_time_per_item,dd_min,dd_max,dd_avg,fpr_min,fpr_max,fpr_avg,tpr_min,tpr_max,tpr_avg\n")
    for result in to_print:
        with open(csv_file, "a") as myfile:
            myfile.write(str(result['general']['use_timeseries']) + "," +
                         str(result['general']['supervised']) + "," +
                         analysis_tag + "," + result['general']['clf'] + "," +
                         str(result['point']['tp']) + "," + str(result['point']['fp']) + "," +
                         str(result['point']['fn']) + "," + str(result['point']['tn']) + "," +
                         str(result['point']['mcc']) + "," + str(result['point']['accuracy']) + "," +
                         str(result['point']['recall']) + ',' + str(result['general']['predict_time']) + ',' +
                         str(result['general']['predict_time_per_item']) + "," +
                         str(result['timeseries']['dd']['min']) + "," +
                         str(result['timeseries']['dd']['max']) + "," +
                         str(result['timeseries']['dd']['avg']) + "," +
                         str(result['timeseries']['fpr']['min']) + "," +
                         str(result['timeseries']['fpr']['max']) + "," +
                         str(result['timeseries']['fpr']['avg']) + "," +
                         str(result['timeseries']['tpr']['min']) + "," +
                         str(result['timeseries']['tpr']['max']) + "," +
                         str(result['timeseries']['tpr']['avg']) + "\n")


def clear_folder(folder_path):
    """
    Used to clear an existing folder
    :param folder_path:
    :return:
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))