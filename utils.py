"""
Small miscalaneous tools.
"""

import os
import shutil
import numpy as np
import pandas as pd

def delete_unfinished_logdirs(logs):
    logdirs = [os.path.join(logs, x) for x in os.listdir(logs)]
    finished = [os.path.isfile(os.path.join(logdir, "args.pickle")) for logdir in logdirs]
    unfinished_logdirs = [logdir for f, logdir in zip(finished, logdirs) if not f]
    # ask for confirmation
    if len(unfinished_logdirs) > 0:
        answer = ""
        while answer not in ["y", "n"]:
            answer = input("OK to delete {} folders in {} [Y/N]? ".format(
                len(unfinished_logdirs), logs)).lower()
        if answer == "y":
            print("Deleting...")
            for logdir in unfinished_logdirs: 
                try:
                    shutil.rmtree(logdir)
                except OSError as e:
                    print("Error: %s : %s" % (logdir, e.strerror))
            print("Finished.")
        else:
            print("Aborted. No deletion performed.")
    else: 
        print("There are no unfinished logdirs in {}.".format(logs))

def fix_folder_names(logs):
    cwd = os.getcwd()
    # Rename all Training folders from Training-some-date-here to Training
    # So that ray tune has no problem retreiving logs
    for p in [os.path.join(logs, f) for f in os.listdir(logs)]:
        os.chdir(p) 
        os.rename([f for f in os.listdir(p) if f.startswith('Training')][0], 'Training')
    os.chdir(cwd)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_orders(row):
    array = row.values
    temp = sorted(list(array))
    res = np.array([temp.index(i) for i in array])
    return pd.Series(res, row.index)

def get_parts(row, nparts):
    array = row.values
    temp = sorted(list(array))
    res = np.array([temp.index(i) for i in array])
    labels = list(reversed((np.arange(nparts)+1)*(30/nparts)))
    return pd.qcut(pd.Series(res, row.index).rank(method='first'),nparts,labels).astype(float)