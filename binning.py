# !/usr/bin/env python
# coding=utf-8

import json
import math
import random
import os
import sys
import numpy as np
import pandas as pd
from functools import partial
from pyspark import Broadcast, RDD
from pyspark.sql.types import *
from pyspark.sql import functions as F, DataFrame
from base_operator import SparkMLOperator, logger

NUMBER_TYPES = {
    # BooleanType: bool,
    ByteType: int,
    ShortType: int,
    IntegerType: int,
    LongType: int,
    FloatType: float,
    DoubleType: float,
    DecimalType: float,
    # StringType: str,
    # BinaryType: (bytearray, bytes),
    # DateType: (datetime.date, datetime.datetime),
    # TimestampType: (datetime.datetime,),
    # ArrayType: (list, tuple, array),
    # MapType: (dict,),
    # StructType: (tuple, list, dict),
}

result_schema = StructType([
    StructField("feature", StringType(), True), StructField("bin", StringType(), True),
    StructField("total", DoubleType(), True), StructField("total_pcnt", DoubleType(), True),
    StructField("good_num", DoubleType(), True), StructField("bad_num", DoubleType(), True),
    StructField("good_rate", DoubleType(), True), StructField("bad_rate", DoubleType(), True),
    StructField("good_pcnt", DoubleType(), True), StructField("bad_pcnt", DoubleType(), True),
    StructField("gini", DoubleType(), True), StructField("woe", DoubleType(), True),
    StructField("iv", DoubleType(), True), StructField("good_cum", DoubleType(), True),
    StructField("bad_cum", DoubleType(), True), StructField("ks", DoubleType(), True),
    StructField("iv_sum", DoubleType(), True), StructField("bad_mono", StringType(), True),
    StructField("ks_mono", StringType(), True)
])

GOOD, BAD = 0, 1
MAX_COLUMN_COUNT = 1 * 10000
MAX_SAMPLE_COUNT = 10000 * 10000
MAX_discrete_VALUES = 50
NUMBER_TYPE_NAMES = [x().typeName() for x in NUMBER_TYPES.keys()]


def parse_memory(s):
    units = {'g': 1024, 'm': 1, 't': 1 << 20, 'k': 1.0 / 1024}
    if s[-1].lower() not in units:
        raise ValueError("invalid format: " + s)
    return int(float(s[:-1]) * units[s[-1].lower()])

def get_local_dirs(sub):
    """ Get all the directories """
    path = os.environ.get("SPARK_LOCAL_DIRS", "/tmp")
    dirs = path.split(",")
    if len(dirs) > 1:
        # different order in different processes and instances
        rnd = random.Random(os.getpid() + id(dirs))
        random.shuffle(dirs, rnd.random)
    return [os.path.join(d, "python", str(os.getpid()), sub) for d in dirs]

def new_memmap_list(subdir, filename, dtype, shap):
    """ create a new list based on numpy memmap list """
    dirs = get_local_dirs(subdir)
    d = dirs[0]
    if not os.path.exists(d):
        os.makedirs(d)
    data_file = os.path.join(d, filename)
    data = np.memmap(data_file, mode='w+', dtype=dtype, shape=shap)
    return data_file, data

def new_list(memory_limit, subdir, filename, dtype, shap):
    """ create a new list based on memory """
    a = [dtype() for _ in range(1000)]
    a_memory = sys.getsizeof(a)/1024/1024
    data_memory = a_memory * shap[0] * shap[1] * (shap[0]/1000)
    if data_memory < memory_limit:
        return None, [[0]*shap[1] for _ in range(shap[0])]
    return new_memmap_list(subdir, filename, dtype, shap) 


class FeatBinOperator(SparkMLOperator):

    @staticmethod
    def is_monotonous(in_list):
        judgement = [in_list[i] < in_list[i + 1] for i in range(len(in_list) - 1)]
        Monotone = len(set(judgement))
        if Monotone == 1:
            return True
        else:
            return False

    @staticmethod
    def is_convex(in_list):
        in_list_new = np.asarray(in_list, dtype=np.double)
        diffs = np.diff(in_list_new)
        monotonous = FeatBinOperator.is_monotonous(diffs)
        return monotonous

    @staticmethod
    def bin_stat(data, label_acc, left, right):
        """compute bin stat of (left, right], include left if 
        left is the first index of one column.

        Args:
            data (list): value, label
            label_acc (list): good acc, bad acc
            left (int): left index of data
            right (int): right index of data

        Returns:
            tuple: (new left, init acc, good count, bad count)
        """
        new_left = left if left == 0 else left+1
        first_label = int(data[new_left][1])
        init_acc = list(label_acc[new_left]) # new list
        init_acc[first_label] = init_acc[first_label] - 1
        good_count = label_acc[right][GOOD] - init_acc[GOOD]
        bad_count = label_acc[right][BAD] - init_acc[BAD]
        return new_left, init_acc, good_count, bad_count

    @staticmethod
    def ks_binning(data, label_acc, left, right):
        """ compute ks bins of (left, right], include 
        left if left is the first index of one column.

        Args:
            data (list): value, label
            label_acc (list): good acc, bad acc
            left (int): left index of data
            right (int): right index of data

        Returns:
            tuple: best ks and bins
        """
        new_left, init_acc, good_count, bad_count = FeatBinOperator.bin_stat(data, label_acc, left, right)
        if good_count == 0 or bad_count == 0:
            return 1, []

        best_row_index, best_ks = -1, -1
        for i in range(new_left, right):
            acc = label_acc[i]
            tmp_good = acc[GOOD] - init_acc[GOOD]
            tmp_bad = acc[BAD] - init_acc[BAD]
            good_rate = tmp_good / good_count
            bad_rate = tmp_bad / bad_count
            ks = abs(good_rate - bad_rate)
            if ks > best_ks:
                best_ks = ks
                best_row_index = i

        bins = []
        bins.append((left, best_row_index))
        bins.append((best_row_index, right))

        return best_ks, bins

    @staticmethod
    def gini_binning(data, label_acc, left, right):
        """ compute gini bins of (left, right], include 
        left if left is the first index of one column.

        Args:
            data (list): value, label
            label_acc (list): good acc, bad acc
            left (int): left index of data
            right (int): right index of data

        Returns:
            tuple: best gini index and bins
        """
        new_left, init_acc, good_count, bad_count = FeatBinOperator.bin_stat(data, label_acc, left, right)
        if good_count == 0 or bad_count == 0:
            return 0, []

        count = good_count + bad_count
        best_row_index, best_gini_index = -1, 1
        for i in range(new_left, right):
            acc = label_acc[i]
            left_good, left_bad = acc[GOOD] - init_acc[GOOD], acc[BAD] - init_acc[BAD]
            right_good, right_bad = good_count - left_good, bad_count - left_bad
            left_count, right_count = left_good+left_bad, right_good+right_bad
            left_gini = 1 - (left_good/left_count)**2 - (left_bad/left_count)**2
            right_gini = 1 - (right_good/right_count)**2 - (right_bad/right_count)**2
            gini_index = (left_count/count)*left_gini + (right_count/count)*right_gini
            if gini_index < best_gini_index:
                best_gini_index = gini_index
                best_row_index = i

        bins = []
        bins.append((left, best_row_index))
        bins.append((best_row_index, right))

        return best_gini_index, bins

    @staticmethod
    def chi_binning(data, label_acc, left, right):
        """ compute chi-square bins of (left, right], include 
        left if left is the first index of one column.

        Args:
            data (list): value, label
            label_acc (list): good acc, bad acc
            left (int): left index of data
            right (int): right index of data

        Returns:
            tuple: best gini and bins
        """
        new_left, init_acc, good_count, bad_count = FeatBinOperator.bin_stat(data, label_acc, left, right)
        if good_count == 0 or bad_count == 0:
            return 0, []

        count = good_count + bad_count
        good_rate, bad_rate = good_count/count, bad_count/count
        best_row_index, best_chi = -1, -math.inf
        for i in range(new_left, right):
            acc = label_acc[i]
            left_good, left_bad = acc[GOOD] - init_acc[GOOD], acc[BAD] - init_acc[BAD]
            right_good, right_bad = good_count - left_good, bad_count - left_bad
            left_count, right_count = left_good+left_bad, right_good+right_bad
            left_good_expected, left_bad_expected = left_count * good_rate, left_count * bad_rate
            right_good_expected, right_bad_expected = right_count * good_rate, right_count * bad_rate
            a11 = (left_good-left_good_expected)**2 / left_good_expected
            a12 = (left_bad-left_bad_expected)**2 / left_bad_expected
            a21 = (right_good-right_good_expected)**2 / right_good_expected
            a22 = (right_bad-right_bad_expected)**2 / right_bad_expected
            chi = a11 + a12 + a21 + a22
            if chi > best_chi:
                best_chi = chi
                best_row_index = i

        bins = []
        bins.append((left, best_row_index))
        bins.append((best_row_index, right))

        return best_chi, bins

    @staticmethod
    def quantile_binning(data, count, left, right, init_bin_ratio):
        """ compute quantile bins of (left, right], include 
        left if left is the first index of one column.

        Args:
            data (list): column index, value, label
            count (int): count of data
            left (int): left index of data
            right (int): right index of data
            init_bin_ratio (float): init bin ratio

        Returns:
            list: bins
        """
        init_bin_count = int(1/init_bin_ratio)
        if count/init_bin_count < 1:
            init_bin_count = count

        poses = [round((i*init_bin_ratio)*count) for i in range(init_bin_count+1)]
        bins = [(poses[i-1], poses[i]-1) for i in range(1, init_bin_count+1)]
        bins = [bins[0]] + [(bins[i][0]-1, bins[i][1]) for i in range(1, len(bins))]

        return bins

    @staticmethod
    def bucket_binning(data, count, left, right, init_bin_num):
        """ compute bucket bins of (left, right].

        Args:
            data (list): column index, value, label
            count (int): count of data
            left (int): left index of data
            right (int): right index of data
            init_bin_num (int): init bin number

        Returns:
            list: bins
        """
        min, max = data[left][0], data[right][0]
        diff = (max - min) / init_bin_num
        split_values = [min+(i+1)*diff for i in range(init_bin_num)]
        split_values[-1] = max

        bins = []
        current_left, last_i = 0, 0
        current_value_i = 0
        current_value = split_values[current_value_i]
        for i in range(left+1, right+1):
            value = data[i][0]
            if value > current_value:
                bins.append((current_left, last_i))
                current_left = last_i
                current_value_i += 1
                current_value = split_values[current_value_i]
            last_i = i

        return bins

    @staticmethod
    def user_binning(data, left, right, user_bins):
        """ compute bins by user defined bins with value.

        Args:
            data (list): column index, value, label
            left (int): left index of data
            right (int): right index of data
            user_bins (list): user defined bins with value

        Returns:
            list: bins
        """
        user_bins_len = len(user_bins)
        if user_bins_len == 0:
            return []

        bins = []
        current_left, last_i = 0, 0
        user_bin_i = 0
        user_bin_value = user_bins[user_bin_i][1]
        for i in range(left+1, right+1):
            value = data[i][0]
            if value > user_bin_value:
                bins.append((current_left, last_i))
                current_left = last_i
                user_bin_i += 1
                if user_bin_i >= user_bins_len:
                    break
                user_bin_value = user_bins[user_bin_i][1]
            last_i = i

        return bins

    @staticmethod
    def merge_bins(data, label_acc, max_bin_num, col_bins):
        """ merge bins of continuous column by chi index

        Args:
            data (list): value, label
            label_acc (list): good acc, bad acc
            max_bin_num (int): maximum bins number
            col_bins (list): bins which is sorted by first index

        Returns:
            list: merged bins
        """
        if len(col_bins) <= 1:
            return col_bins

        bins_chi_map = {}
        while True:
            bins_chi = []
            bins_len = len(col_bins)
            for i in range(1, bins_len):
                bin1, bin2 = col_bins[i-1], col_bins[i]
                chi = bins_chi_map.get((bin1, bin2))
                if chi is None:
                    _, _, bin1_good, bin1_bad = FeatBinOperator.bin_stat(data, label_acc, bin1[0], bin1[1])
                    _, _, bin2_good, bin2_bad = FeatBinOperator.bin_stat(data, label_acc, bin2[0], bin2[1])
                    good_count, bad_count = bin1_good+bin2_good, bin1_bad+bin2_bad
                    count = good_count + bad_count
                    good_rate, bad_rate = good_count/count, bad_count/count
                    bin1_good_expected, bin1_bad_expected = bin1_good*good_rate, bin1_bad*bad_rate
                    bin2_good_expected, bin2_bad_expected = bin2_good*good_rate, bin2_bad*bad_rate
                    a11 = (bin1_good-bin1_good_expected)**2 / bin1_good_expected if bin1_good_expected > 0 else 0
                    a12 = (bin1_bad-bin1_bad_expected)**2 / bin1_bad_expected if bin1_bad_expected > 0 else 0
                    a21 = (bin2_good-bin2_good_expected)**2 / bin2_good_expected if bin2_good_expected > 0 else 0
                    a22 = (bin2_bad-bin2_bad_expected)**2 / bin2_bad_expected if bin2_bad_expected > 0 else 0
                    chi = a11 + a12 + a21 + a22
                    bins_chi_map[(bin1, bin2)] = chi
                bins_chi.append((chi, (i-1, i)))

            if bins_len <= max_bin_num:
                not_zero_chis = [item[0] for item in bins_chi if np.isclose(item[0], 0)]
                if len(not_zero_chis) == 0:
                    break

            bins_chi.sort(key=lambda x: x[0], reverse=True)
            merge_i, merge_j = bins_chi.pop()[1]  # minimum chi
            merge_bin1, merge_bin2 = col_bins[merge_i], col_bins[merge_j]

            # remove affected bins chi cache
            del bins_chi_map[(merge_bin1, merge_bin2)]
            if merge_i-1 >= 0:
                del bins_chi_map[(col_bins[merge_i-1], merge_bin1)]
            if merge_j+1 <= bins_len-1:
                del bins_chi_map[(merge_bin2, col_bins[merge_j+1])]

            new_bin = (merge_bin1[0], merge_bin2[1])
            col_bins[merge_j] = new_bin
            del col_bins[merge_i]
        return col_bins

    @staticmethod
    def compute_result(col, data, label_acc, good_count, bad_count, bins, bins_index={}):
        """ compute continuous column binning result """

        count = good_count + bad_count
        result = []
        for i in range(len(bins)):
            bin = bins[i]
            left, right = bin
            stat = [None] * 19
            """
            (0)feature, (1)bin, (2)total, (3)total_pcnt, (4)good_num, 
            (5)bad_num, (6)good_rate, (7)bad_rate, (8)good_pcnt, (9)bad_pcnt, 
            (10)gini, (11)woe, (12)iv, (13)good_cum, (14)bad_cum, 
            (15)ks, (16)iv_sum, (17)bad_mono, (18)ks_mono
            """
            stat[0] = col
            value1, value2 = data[left][0], data[right][0]
            if i == 0:
                stat[1] = '(-inf, {}]'.format(value2)
            elif i == len(bins)-1:
                stat[1] = '({}, inf)'.format(value1)
            else:
                stat[1] = '({}, {}]'.format(value1, value2)
            _, _, bin_good, bin_bad = FeatBinOperator.bin_stat(data, label_acc, left, right)
            stat[2] = float(bin_good + bin_bad)
            stat[3] = float(stat[2]) / float(count)
            stat[4] = float(bin_good)
            stat[5] = float(bin_bad)
            stat[6] = float(bin_good / stat[2])
            stat[7] = float(bin_bad / stat[2])
            stat[8] = float(bin_good) / float(good_count)
            stat[9] = float(bin_bad) / float(bad_count)
            stat[10] = 1.0-(stat[4]/stat[2])**2-(stat[5]/stat[2])**2
            if bin_good == 0 or bin_bad == 0:
                stat[11] = math.log((bin_good+0.5)/good_count) - math.log((bin_bad+0.5)/bad_count)
                stat[12] = float((bin_good+0.5)/good_count - (bin_bad+0.5)/bad_count) * stat[11]
            else:
                stat[11] = math.log(stat[8]) - math.log(stat[9])
                stat[12] = float(stat[8] - stat[9]) * stat[11]
            if bin in bins_index:
                stat[15] = bins_index[bin][0]
            else:
                stat[15], _ = FeatBinOperator.ks_binning(data, label_acc, left, right)
            stat[15] = float(stat[15])
            result.append(stat)

        bins_good = [stat[4] for stat in result]
        bins_bad = [stat[5] for stat in result]
        bad_rates = [stat[7] for stat in result]
        kss = [stat[15] for stat in result]
        iv_sum = sum([stat[12] for stat in result])

        cum_bins_good = [bins_good[0]]
        cum_bins_bad = [bins_bad[0]]
        for i in range(1, len(result)):
            cum_bins_good.append(cum_bins_good[i-1] + bins_good[i])
            cum_bins_bad.append(cum_bins_bad[i-1] + bins_bad[i])

        bad_rates_type = 'None'
        if FeatBinOperator.is_monotonous(bad_rates):
            bad_rates_type = 'monotonous'
        elif FeatBinOperator.is_convex(bad_rates):
            bad_rates_type = 'convex'

        kss_type = 'None'
        if FeatBinOperator.is_monotonous(kss):
            kss_type = 'monotonous'
        elif FeatBinOperator.is_convex(kss):
            kss_type = 'convex'

        for i in range(len(result)):
            stat = result[i]
            stat[13] = float(cum_bins_good[i])
            stat[14] = float(cum_bins_bad[i])
            stat[16] = float(iv_sum)
            stat[17] = bad_rates_type
            stat[18] = kss_type

        return result

    @staticmethod
    def compute_single_stat(col, single_data, single_good, single_bad):
        """ compute single value statistic"""

        stat = [None] * 19
        """
        (0)feature, (1)bin, (2)total, (3)total_pcnt, (4)good_num, 
        (5)bad_num, (6)good_rate, (7)bad_rate, (8)good_pcnt, (9)bad_pcnt, 
        (10)gini, (11)woe, (12)iv, (13)good_cum, (14)bad_cum, 
        (15)ks, (16)iv_sum, (17)bad_mono, (18)ks_mono
        """
        count = single_good + single_bad
        if count == 0:
            return []

        value = single_data[0][0]
        stat[0] = col
        stat[1] = '[{}, {}]'.format(value, value)
        stat[2] = float(count)
        stat[3] = float(stat[2]) / float(count)
        stat[4] = float(single_good)
        stat[5] = float(single_bad)
        stat[6] = float(single_good / stat[2])
        stat[7] = float(single_bad / stat[2])
        stat[8] = None
        stat[9] = None
        stat[10] = 1.0-(stat[4]/stat[2])**2-(stat[5]/stat[2])**2
        stat[11] = 0.0
        stat[12] = None
        stat[13] = None
        stat[14] = None
        stat[15] = None
        stat[16] = None
        stat[17] = None
        stat[18] = None
        return stat

    @staticmethod
    def best_binning(data, label_acc, data_count, max_bin_num, min_bin_count,
                     binning_method, sorted_method, bins_index):
        """ binning for best index

        Args:
            data (list): value, label
            label_acc (list): good and bad acc
            data_count (int): data count
            max_bin_num (int): max bins number
            min_bin_count (int): min count of one bin
            binning_method (function): binning method
            sorted_method (function): index sort method
            bins_index (dict): bins index cache
        """
        not_split_bins = []
        bin = (0, data_count-1)
        bins_index[bin] = binning_method(data, label_acc, bin[0], bin[1])
        while True:
            # filter bins for binning
            ks_bins = [(bins_index[bin][0], bin) for bin in bins_index]
            ks_bins = [item for item in ks_bins if item[1] not in not_split_bins]
            if len(ks_bins) == 0:
                break

            # get best index bin
            ks_bins = sorted_method(ks_bins)
            _, bin = ks_bins.pop()
            _, new_bins = bins_index[bin]

            can_split = True
            if len(new_bins) > 0:
                for new_bin in new_bins:
                    new_bin_count = new_bin[1] - new_bin[0]
                    new_bin_count = new_bin_count+1 if new_bin[0] == 0 else new_bin_count
                    if new_bin_count < min_bin_count:
                        can_split = False
                        break
            else:
                can_split = False

            if can_split:
                # compute index and binning
                for new_bin in new_bins:
                    bins_index[new_bin] = binning_method(data, label_acc, new_bin[0], new_bin[1])
                del bins_index[bin]
            else:
                not_split_bins.append(bin)

            # check bins count
            current_bins = set(not_split_bins).union(set([bin for bin in bins_index]))
            if len(current_bins) >= max_bin_num:
                break

        # sort by first index
        col_bins = list(set(not_split_bins).union(set([bin for bin in bins_index])))
        col_bins.sort(key=lambda x: x[0])

        # combine bins which have same start and end value
        continue_combine = True
        while continue_combine:
            if len(col_bins) < 2:
                continue_combine = False
                break
            for i in range(1, len(col_bins)):
                bin1 = col_bins[i-1]
                bin2 = col_bins[i]
                if data[bin1[0]][0] == data[bin2[0]][0] and \
                        data[bin1[1]][0] == data[bin2[1]][0]:
                    col_bins[i] = (bin1[0], bin2[1])
                    del col_bins[i-1]
                    continue_combine = True
                    break
                if i == len(col_bins) - 1:
                    continue_combine = False
                    break
        return col_bins

    @staticmethod
    def column_data_iter(partition_iter, memory, count, single_value):
        """ return a iter which is one column data.

        Args:
            partition_iter (iter): sorted by column index and value
            count (int): row count of one column
            single_value (float): single binning value

        Yields:
            iter: iter which is one column data
        """
        filename = str(id(partition_iter))
        data_file, data = new_memmap_list('binning_data', filename, np.float64, (count, 2))
        single_file, single_data = new_memmap_list('binning_single', filename, np.float64, (count, 2))
        acc_file, label_acc = new_list(memory*0.6, 'binning_acc', filename, np.int32, (count, 2))
        column_index = -1
        last_acc = [0, 0]
        data_count = 0
        single_count = 0
        for row in partition_iter:
            label = row[2]
            column_index = row[0] if column_index == -1 else column_index
            if row[0] == column_index:
                if row[1] == single_value:
                    single_data[single_count] = row[1:]
                    single_count += 1
                else:
                    data[data_count] = row[1:]
                    another = BAD if label == GOOD else GOOD
                    label_acc[data_count][label] = last_acc[label] + 1
                    label_acc[data_count][another] = last_acc[another] 
                    last_acc = label_acc[data_count]
                    data_count += 1
            else:
                yield column_index, data_count, data, single_data, label_acc

                column_index = row[0]
                last_acc = [0, 0]
                data_count = 0
                single_count = 0
                if row[1] == single_value:
                    single_data[single_count] = row[1:]
                    single_count += 1
                else:
                    data[data_count] = row[1:]
                    another = BAD if label == GOOD else GOOD
                    label_acc[data_count][label] = last_acc[label] + 1
                    label_acc[data_count][another] = last_acc[another] 
                    last_acc = label_acc[data_count]
                    data_count += 1
                
        # last row
        yield column_index, data_count, data, single_data, label_acc

        del data
        del single_data
        del label_acc
        os.remove(data_file)
        os.remove(single_file)
        if acc_file is not None:
            os.remove(acc_file)

    @staticmethod
    def continuous_binning_f(partition_iter, bc: Broadcast):
        """ all records of one column must be in the same partition

        Args:
            partition_iter (iter): sorted by column index and value
            bc (Broadcast): bc value
        """
        cut_method = bc.value['cut_method']
        continuous_cols = bc.value['continuous_cols']
        count = bc.value['count']
        good_count = bc.value['good_count']
        bad_count = bc.value['bad_count']
        max_bin_num = bc.value['max_bin_num']
        init_bin_ratio = bc.value['init_bin_ratio']
        min_bin_ratio = bc.value['min_bin_ratio']
        single_value = bc.value['single_value']
        user_defined_bins = bc.value['user_defined_bins']
        python_memory = bc.value['python_memory']
        cut_method_map = {
            'bestks': FeatBinOperator.ks_binning,
            'bestgini': FeatBinOperator.gini_binning,
            'bestchisq': FeatBinOperator.chi_binning,
            'quantile': FeatBinOperator.quantile_binning,
            'bucket': FeatBinOperator.bucket_binning,
            'user': FeatBinOperator.user_binning,
        }
        binning_method = cut_method_map.get(cut_method)
        sort_method_map = {
            'bestks': lambda x: sorted(x, key=lambda y: y[0], reverse=False),
            'bestgini': lambda x: sorted(x, key=lambda y: y[0], reverse=True),
            'bestchisq': lambda x: sorted(x, key=lambda y: y[0], reverse=True),
        }
        sorted_method = sort_method_map.get(cut_method)

        for column_index, data_count, data, single_data, label_acc in \
            FeatBinOperator.column_data_iter(partition_iter, python_memory, count, single_value):
            result = []
            column_name = continuous_cols[column_index]
            data_good, data_bad = 0, 0
            # none single value binning
            if data_count > 0: 
                data_good, data_bad = label_acc[data_count-1]
                min_bin_count = math.ceil(min_bin_ratio*data_count)
                bins_index = {}
                if cut_method in ['bestks', 'bestgini', 'bestchisq']:
                    col_bins = FeatBinOperator.best_binning(
                        data, label_acc, data_count, max_bin_num, min_bin_count,
                        binning_method, sorted_method, bins_index)
                elif cut_method in ['quantile', 'bucket']:
                    new_bin_count = int(1 / init_bin_ratio)
                    new_bin_count = max_bin_num if new_bin_count < max_bin_num else new_bin_count
                    init_bin_params = init_bin_ratio if cut_method == 'quantile' else new_bin_count
                    col_bins = binning_method(data, data_count, 0, data_count-1, init_bin_params)
                    col_bins.sort(key=lambda x: x[0])
                    col_bins = FeatBinOperator.merge_bins(data, label_acc, max_bin_num, col_bins)
                elif cut_method == 'user':
                    user_bins = user_defined_bins[column_index]
                    col_bins = binning_method((data, 0, data_count-1, user_bins))
                result = FeatBinOperator.compute_result(
                    column_name, data, label_acc, data_good,
                    data_bad, col_bins, bins_index)
            # single value binning
            if count-data_count > 0: 
                single_stat = FeatBinOperator.compute_single_stat(
                    column_name, single_data,
                    good_count-data_good, bad_count-data_bad)
                result.insert(0, single_stat)
            
            for stat in result:
                yield stat

    def repartition(self, df: DataFrame):
        """ repartition if partition number is too low
        """
        executor_num = self.session._sc._conf.get('spark.executor.instances', '1')
        executor_num = float(executor_num)
        executor_cores = self.session._sc._conf.get('spark.executor.cores', '1')
        executor_cores = float(executor_cores)
        minimum_partitions = math.ceil(executor_num*executor_cores)
        if df.rdd.getNumPartitions() < minimum_partitions:
            logger.info('数据分区太少，优化数据分区为: {}'.format(minimum_partitions))
            df = df.repartition(minimum_partitions)
        return df

    def compute_stat(self, df: DataFrame, continuous_cols, discrete_cols, y_col):
        """ compute table statistics

        Args:
            df (DataFrame): data frame: continuous_cols, discrete_cols and then y_col
            continuous_cols (_type_): number cols
            discrete_cols (_type_): discrete cols
            y_col (_type_): y_col
        """
        def seq_op(acc, row, bc: Broadcast):
            continuous_cols_len = bc.value['continuous_cols_len']
            discrete_cols_len = bc.value['discrete_cols_len']
            label = row[-1]
            if label in acc:
                acc[label] += 1
            else:
                raise Exception('标签列必须为: 0或1')

            for i in range(continuous_cols_len):
                if row[i] is None:
                    raise Exception('数值列{}存在Null值，请先进行缺失值填充'.format(continuous_cols[i]))

            for i in range(discrete_cols_len):
                idx = continuous_cols_len + i
                value = row[idx]
                stat = acc['discrete_stat'][idx]
                if value not in stat:
                    stat[value] = {GOOD: 0, BAD: 0}
                stat[value][label] += 1
                if len(stat) > MAX_discrete_VALUES:
                    discrete_cols = bc.value['discrete_cols']
                    raise Exception('离散列{}的分组数超过最大限制({})'.format(discrete_cols[i], MAX_discrete_VALUES))
            return acc

        def com_op(acc1, acc2, bc: Broadcast):
            continuous_cols_len = bc.value['continuous_cols_len']
            discrete_cols_len = bc.value['discrete_cols_len']

            acc1[GOOD] += acc2[GOOD]
            acc1[BAD] += acc2[BAD]

            for i in range(discrete_cols_len):
                idx = continuous_cols_len + i
                stat1 = acc1['discrete_stat'][idx]
                stat2 = acc2['discrete_stat'][idx]
                for value in stat2:
                    if value in stat1:
                        stat1[value][GOOD] += stat2[value][GOOD]
                        stat1[value][BAD] += stat2[value][BAD]
                    else:
                        stat1[value] = stat2[value]
            return acc1

        continuous_cols_len, discrete_cols_len = len(continuous_cols), len(discrete_cols)
        bc = self.session.sparkContext.broadcast({
            'discrete_cols': discrete_cols,
            'continuous_cols_len': continuous_cols_len,
            'discrete_cols_len': discrete_cols_len
        })
        discrete_stat = dict([(idx, {}) for idx in range(continuous_cols_len, continuous_cols_len+discrete_cols_len)])
        zero_value = {GOOD: 0, BAD: 0, 'discrete_stat': discrete_stat}
        stat = df.rdd.treeAggregate(zero_value, partial(seq_op, bc=bc), partial(com_op, bc=bc))
        stat['count'] = stat[GOOD] + stat[BAD]
        return stat

    def unpivot_df(self, df: DataFrame, y_col):
        """ unpivot continuous df

        Args:
            df (DataFrame): continuous cols and y col
            y_col (str): y col
        """
        cols_len = len(df.columns) - 1
        cols_expr = []
        for i in range(cols_len):
            cols_expr.append('{}'.format(i))
            cols_expr.append('cast({} as double)'.format(df.columns[i]))

        unpivot_expr = 'stack({},{}) as (col,value)'.format(cols_len, ','.join(cols_expr))
        unpivot_df = df.select(F.expr(unpivot_expr), y_col)
        return unpivot_df

    def discrete_bin_stat(self, discrete_stat, index, bin):
        """ compute bin statistics of one discrete column

        Args:
            discrete_stat (dict): discrete columns statistics
            index (int): column index
            bin (list): bin

        Returns:
            tuple: bin statistics
        """
        good_count, bad_count = 0, 0
        stat = discrete_stat[index]
        for value in bin:
            value_good, value_bad = stat[value][GOOD], stat[value][BAD]
            good_count += value_good
            bad_count += value_bad
        return good_count, bad_count

    def discrete_bin_chi(self, discrete_stat, index, bin1, bin2):
        """ compute chi index of bin1 and bin2

        Args:
            discrete_stat (dict): discrete columns statistics
            index (int): column index
            bin1 (list): bin1
            bin2 (list): bin2

        Returns:
            float: chi index
        """
        bin1_good, bin1_bad = self.discrete_bin_stat(discrete_stat, index, bin1)
        bin2_good, bin2_bad = self.discrete_bin_stat(discrete_stat, index, bin2)
        bin1_count, bin2_count = bin1_good+bin1_bad, bin2_good+bin2_bad
        good_count, bad_count = bin1_good+bin2_good, bin1_bad+bin2_bad
        count = good_count + bad_count
        good_rate = good_count/count
        bad_rate = bad_count/count
        bin1_good_expected, bin1_bad_expected = bin1_count*good_rate, bin1_count*bad_rate
        bin2_good_expected, bin2_bad_expected = bin2_count*good_rate, bin2_count*bad_rate
        a11 = (bin1_good-bin1_good_expected)**2 / bin1_good_expected if bin1_good_expected > 0 else 0
        a12 = (bin1_bad-bin1_bad_expected)**2 / bin1_bad_expected if bin1_bad_expected > 0 else 0
        a21 = (bin2_good-bin2_good_expected)**2 / bin2_good_expected if bin2_good_expected > 0 else 0
        a22 = (bin2_bad-bin2_bad_expected)**2 / bin2_bad_expected if bin2_bad_expected > 0 else 0
        chi = a11 + a12 + a21 + a22
        return chi

    def discrete_bin_result(self, index, col, discrete_stat, count, good_count, bad_count, bins):
        """ compute discrete column binning result """

        result = []
        for i in range(len(bins)):
            bin = bins[i]
            stat = [None] * 19
            """
            (0)feature, (1)bin, (2)total, (3)total_pcnt, (4)good_num, 
            (5)bad_num, (6)good_rate, (7)bad_rate, (8)good_pcnt, (9)bad_pcnt, 
            (10)gini, (11)woe, (12)iv, (13)good_cum, (14)bad_cum, 
            (15)ks, (16)iv_sum, (17)bad_mono, (18)ks_mono
            """
            stat[0] = col
            format_str = ','.join(['{}' for _ in range(len(bin))])
            stat[1] = format_str.format(*bin)
            bin_good, bin_bad = self.discrete_bin_stat(discrete_stat, index, bin)
            stat[2] = float(bin_good + bin_bad)
            stat[3] = float(stat[2]) / float(count)
            stat[4] = float(bin_good)
            stat[5] = float(bin_bad)
            stat[6] = float(bin_good / stat[2])
            stat[7] = float(bin_bad / stat[2])
            stat[8] = float(bin_good) / float(good_count)
            stat[9] = float(bin_bad) / float(bad_count)
            stat[10] = 1.0-(stat[4]/stat[2])**2-(stat[5]/stat[2])**2
            if bin_good == 0 or bin_bad == 0:
                stat[11] = math.log((bin_good+0.5)/float(good_count)) - math.log((bin_bad+0.5)/float(bad_count))
                stat[12] = float((bin_good+0.5)/good_count - (bin_bad+0.5)/bad_count) * stat[11]
            else:
                stat[11] = math.log(stat[8]) - math.log(stat[9])
                stat[12] = float(stat[8] - stat[9]) * stat[11]
            stat[15] = None
            stat[17] = None
            stat[18] = None
            result.append(stat)

        # sort result by bad rate
        result.sort(key=lambda x: x[7])

        bins_good = [stat[4] for stat in result]
        bins_bad = [stat[5] for stat in result]
        iv_sum = sum([stat[12] for stat in result])

        cum_bins_good = [bins_good[0]]
        cum_bins_bad = [bins_bad[0]]
        for i in range(1, len(result)):
            cum_bins_good.append(cum_bins_good[i-1] + bins_good[i])
            cum_bins_bad.append(cum_bins_bad[i-1] + bins_bad[i])

        for i in range(len(result)):
            stat = result[i]
            stat[13] = float(cum_bins_good[i])
            stat[14] = float(cum_bins_bad[i])
            stat[16] = iv_sum

        return result

    def discrete_user_binning(self, table_stat, continuous_cols, discrete_cols):
        """ compute user defined bins of discrete columns

        Args:
            table_stat (dict): table statistics
            continuous_cols (list): continuous columns
            discrete_cols (list): discrete columns

        Returns:
            list: result
        """
        count = table_stat['count']
        good_count = table_stat[GOOD]
        bad_count = table_stat[BAD]
        discrete_stat = table_stat['discrete_stat']
        user_defined = self.context.get("user_defined")
        continuous_cols_len = len(continuous_cols)

        user_defined_bins = {}
        for col, bins_str in user_defined.items():
            bins = []
            for bin_str in bins_str:
                bin_str = bin_str.replace('(', '[')
                bin_str = bin_str.replace(')', ']')
                bins.append(eval(bin_str))
            index = discrete_cols.index(col)
            index = continuous_cols_len + index
            user_defined_bins[index] = bins

        result = []
        for index in discrete_stat:
            col_bins = user_defined_bins[index]
            col = discrete_cols[index - continuous_cols_len]
            col_result = self.discrete_bin_result(index, col, discrete_stat, count, good_count, bad_count, col_bins)
            result.extend(col_result)
        return result

    def discrete_binning(self, table_stat, continuous_cols, discrete_cols):
        """ compute discrete column bins

        Args:
            table_stat (dict): table statistics
            continuous_cols (list): continuous columns
            discrete_cols (list): discrete columns

        Returns:
            list: result
        """
        count = table_stat['count']
        good_count = table_stat[GOOD]
        bad_count = table_stat[BAD]
        max_bin_num = self.context.get('piece')
        discrete_stat = table_stat['discrete_stat']
        continuous_cols_len = len(continuous_cols)

        result = []
        for index in discrete_stat:
            bins_chi_map = {}
            col_bins = [[value] for value in discrete_stat[index]]
            while True:
                bins_len = len(col_bins)
                bins_chi = []
                for i in range(0, bins_len-1):
                    for j in range(1, bins_len):
                        sorted_values = sorted(col_bins[i]+col_bins[j], key=lambda x: (x is None, x))
                        values_key = tuple(sorted_values)
                        chi = bins_chi_map.get(values_key)
                        if chi is None:
                            chi = self.discrete_bin_chi(discrete_stat, index, col_bins[i], col_bins[j])
                            bins_chi_map[values_key] = chi
                        bins_chi.append((chi, (i, j)))

                if bins_len <= max_bin_num:
                    not_zero_chis = [item[0] for item in bins_chi if np.isclose(item[0], 0)]
                    if len(not_zero_chis) == 0:
                        break

                bins_chi.sort(key=lambda x: x[0], reverse=True)
                merge_i, merge_j = bins_chi.pop()[1]  # minimum chi
                merge_bin1, merge_bin2 = col_bins[merge_i], col_bins[merge_j]
                new_bin = sorted(merge_bin1 + merge_bin2, key=lambda x: (x is None, x))

                # remove affected chi cache
                affected_keys = []
                for key in bins_chi_map:
                    if not set(key).intersection(set(merge_bin1)) or not set(key).intersection(set(merge_bin2)):
                        affected_keys.append(key)
                for key in affected_keys:
                    del bins_chi_map[key]

                col_bins[merge_j] = new_bin
                del col_bins[merge_i]

            col = discrete_cols[index - continuous_cols_len]
            col_result = self.discrete_bin_result(index, col, discrete_stat, count, good_count, bad_count, col_bins)
            result.extend(col_result)
        return result

    def continuous_binning(self, cut_method, sorted_rdd: RDD, continuous_cols, table_stat):
        """ compute continuous column bins

        Args:
            cut_method (str): cut method
            sorted_rdd (RDD): spark rdd partitioned by column index and sorted by column index and value
            continuous_cols (list): continuous columns
            table_stat (dict): table statistics

        Returns:
            RDD: result
        """
        count = table_stat['count']
        good_count = table_stat[GOOD]
        bad_count = table_stat[BAD]
        max_bin_num = self.context.get('piece')
        min_bin_ratio = self.context.get('ratio')
        init_bin_ratio = self.context.get('init_ratio')
        single_value = self.context.get('single_value')
        single_value = None if single_value in ['None', ''] else float(single_value)
        is_user_defined = self.context.get('is_user_defined')
        user_defined = self.context.get("user_defined")
        python_memory = self.session._sc._conf.get('spark.python.worker.memory', '512m')
        python_memory = parse_memory(python_memory)

        user_defined_bins = {}
        if is_user_defined:
            for col, bins_str in user_defined.items():
                bins = []
                for bin_str in bins_str:
                    bin_str = bin_str.replace('(', '[')
                    bin_str = bin_str.replace(')', ']')
                    bins.append(eval(bin_str))
                bins.sort(key=lambda x: x[0])
                index = continuous_cols.index(col)
                user_defined_bins[index] = bins

        bc = self.session.sparkContext.broadcast({
            'cut_method': cut_method,
            'continuous_cols': continuous_cols,
            'count': count,
            'good_count': good_count,
            'bad_count': bad_count,
            'max_bin_num': max_bin_num,
            'min_bin_ratio': min_bin_ratio,
            'single_value': single_value,
            'init_bin_ratio': init_bin_ratio,
            'user_defined_bins': user_defined_bins,
            'python_memory': python_memory,
        })
        result = sorted_rdd.mapPartitions(partial(FeatBinOperator.continuous_binning_f, bc=bc))
        return result

    def saveRuntime(self, pd_df):
        rule = {
            'bin': 'hdfs://' + self.outputs[1] + '.json',
            'x_cols':  self.context.get('x_cols'),
            'discrete_cols':  self.context.get('discrete_cols'),
            'global_params':  self.context.get('global_params'),
            'cut_method': self.context.get('cut_method'),
            'piece':  self.context.get('piece'),
            'ratio':  self.context.get('ratio'),
            'init_ratio':  self.context.get('init_ratio'),
            'single_value':  self.context.get('single_value'),
            'is_user_defined':  self.context.get('is_user_defined'),
            'user_defined':  self.context.get('user_defined'),
        }

        bin_json = {}
        for feature in list(set(pd_df['feature'].tolist())):
            bin_json[feature] = pd_df[pd_df["feature"] == feature]['bin'].tolist()

        self.runtime.add(rule)
        self.saveFile(bin_json, self.outputs[1] + '.json')

    def execute(self, read_data):
        df: DataFrame = read_data[0]

        # check y_col
        global_params = self.context.get("global_params")
        y_cols = global_params.get('y_col')
        assert y_cols is not None and len(y_cols) == 1, logger.info("标签列为空或者多于一个，请重新选择！")
        y_col = y_cols[0]
        x_cols = self.context.get("x_cols")
        discrete_cols = self.context.get('discrete_cols')
        cut_method = self.context.get('cut_method')

        # process user defined binning: used for interactive binning
        is_user_defined = self.context.get('is_user_defined')
        if is_user_defined:
            user_defined = self.context.get("user_defined")
            assert user_defined is not None, logger.error('自定义分箱数据不能为空')
            # change some settings for user defined binning
            x_cols = list(user_defined.keys())
            discrete_cols = [col for col in x_cols if col in discrete_cols]
            cut_method = 'user'

        # process x_cols
        if x_cols is None or len(x_cols) == 0:
            logger.info('因未选择字段，故默认选择全部字段'.format())
            x_cols = df.columns
        else:
            unexpected_cols = set(x_cols) - set(df.columns)
            if len(unexpected_cols) > 0:
                logger.error('选择的列不在当前数据集中：{}'.format(','.join(unexpected_cols)))
        x_cols = list(set(x_cols) - set([y_col]))

        # process discrete cols
        if discrete_cols is None:
            discrete_cols = []
        for col in discrete_cols:
            if col not in x_cols:
                logger.info('离散列{}不在分箱列中，自动添加到分箱列中'.format(col))
                x_cols.append(col)
        data_types = dict([(field.name, field.dataType.typeName()) for field in df.schema.fields])
        for col in x_cols:
            if data_types[col] not in NUMBER_TYPE_NAMES:
                logger.info('列{}为非数值列，自动添加到离散列'.format(col))
                discrete_cols.append(col)

        if len(x_cols) > MAX_COLUMN_COUNT:
            logger.error('所要计算的列超过最大限制，当前为: {}，最大为: {}'.format(len(x_cols), MAX_COLUMN_COUNT))

        # continuous cols and convert y col to integer
        continuous_cols = list(set(x_cols) - set(discrete_cols))
        if data_types[y_col] != IntegerType().typeName():
            df = df.withColumn(y_col, F.col(y_col).cast('int'))
        df = df.select(continuous_cols + discrete_cols + [y_col])

        # repartition if needed
        df = self.repartition(df)

        # compute table statistics
        table_stat = self.compute_stat(df, continuous_cols, discrete_cols, y_col)

        # check GOOD and BAD
        if table_stat[GOOD] == 0 or table_stat[BAD] == 0:
            logger.error('当前样本只有一个分类，无需分箱')

        # check max sample count
        if table_stat['count'] > MAX_SAMPLE_COUNT:
            logger.error('当前样本数超过最大限制(最大{})，请先进行采样'.format(MAX_SAMPLE_COUNT))

        result_pdf: pd.DataFrame = None

        # discrete cols binning
        if len(discrete_cols):
            if cut_method == 'user':
                result = self.discrete_user_binning(table_stat, continuous_cols, discrete_cols)
            else:
                result = self.discrete_binning(table_stat, continuous_cols, discrete_cols)
            result_pdf = self.session.createDataFrame(result, schema=result_schema)
            result_pdf = result_pdf.toPandas()

        # continuous cols binning
        if len(continuous_cols) > 0: 
            if cut_method in ['bestks', 'bestgini', 'bestchisq', 'quantile', 'bucket', 'user']:
                continuous_df = df.select(continuous_cols + [y_col])
                unpivot_df = self.unpivot_df(continuous_df, y_col)
                sorted_df = unpivot_df.repartition('col').sortWithinPartitions('col', 'value')
                binning_rdd = self.continuous_binning(cut_method, sorted_df.rdd, continuous_cols, table_stat)
                binning_df = binning_rdd.toDF(schema=result_schema)
                binning_pdf = binning_df.toPandas()
                result_pdf = binning_pdf if result_pdf is None else pd.concat([result_pdf, binning_pdf])
            else:
                logger.error('unknown cut method:{}'.format(cut_method))

        self.saveRuntime(result_pdf)
        return [self.session.createDataFrame(result_pdf)]


def main(session, *args, **kwargs):
    inputs = json.loads('${inputs}')
    outputs = json.loads('${outputs}')
    context = json.loads('${config_param}')

    operator = FeatBinOperator(session=session, context=context, inputs=inputs, outputs=outputs, *args, **kwargs)
    return operator.run()
    
