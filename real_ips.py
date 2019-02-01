#
# Implements the HORUS probabilistic WiFi Localization system
# ( Cyber Physical System Group )
# Jan. 2019
# Chris Xiaoxuan Lu
#

import numpy as np
import os
import glob
import sys
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm
import argparse
import csv
import numpy as np


def load_real_csv(csv_file):
    # EXAMPLE: "2017-06-22_15-02-47_wifi.csv"

    # 1.498139656195E9,eduroam,00:81:c4:85:07:a0,2462,-60
    # timestamp, ssid, mac, channel (ignore), rss
    # Ignore ssid and use mac as location identifier.
    return np.array([(mac, int(rss)) for timestamp, ssid, mac, channel, rss in csv.reader(open(csv_file, 'r'))])


def load_real_data_train(path):
    locations = np.array(list(csv.reader(open('%s/training/location.txt' % path, 'r')))).astype(float)
    data = [load_real_csv('%s/training/wifi_signal/train_%s.csv' % (path, index + 1)) for index in range(len(locations))]
    return locations, data


def load_real_data_test(path):
    locations = np.array(list(csv.reader(open('%s/test/location.txt' % path, 'r')))).astype(float)
    data = [load_real_csv('%s/test/wifi_signal/test_%s.csv' % (path, index + 1)) for index in range(len(locations))]
    return locations, data


# TODO: Implement a fingerprinting algorithm that predicts the location given the testset
