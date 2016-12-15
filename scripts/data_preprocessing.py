import os
import sys

sys.path.append(os.path.join(os.getcwd(), "../data"))
sys.path.append(os.path.join(os.getcwd(), "../common"))
import json
import datetime
import re
import itertools
import operator
import numpy as np
import pandas as pd
import path_constants
import csv
import data_constants
from sklearn.feature_extraction import DictVectorizer
delimiter="\t"

def clean_wiki_omdb_raw_data(input_file_path, output_file_path):
    """
    Clean the initial dump of wikipedia and omdb details on various params and save it an a file
    :param input_file_path:
    :param output_file_path:
    :return:
    """
    lines = open(input_file_path, 'r').read().splitlines()
    out_file = open(output_file_path, 'w')
    out_file.write("title\tyear\ttype\tdetails_json\tpage_view_data\n")
    try:
        num_lines = len(lines)
        for i in range(0, num_lines - 1, 2):
            combined_line = lines[i].strip() + "\t" + lines[i + 1].strip() + "\n"
            out_file.write(combined_line)
    except Exception as e:
        print e.message
    finally:
        out_file.close()


def parse_details_json(details_json):
    """
    A function to parse the OMDB data json to return essential details
    :param details_json:
    :return:
    """
    details_dict = json.loads(details_json)
    dict_genre = dict(
        map(lambda x: (x.strip(), 1), details_dict["Genre"].encode('utf-8', 'ignore').split(",")))
    dict_directors = dict(map(lambda x: (x.strip(), 1),
                              details_dict["Director"].encode('utf-8', 'ignore').split(",")))
    dict_actors = dict(
        map(lambda x: (x.strip(), 1), details_dict["Actors"].encode('utf-8', 'ignore').split(",")))
    dict_writers = dict(map(lambda x: (x[0].strip(), 1), re.findall(r"([A-Z]((\w|\.)+ ?\'?-?)+)\(?,?",
                                                                    details_dict["Writer"].encode(
                                                                        'utf-8',
                                                                        'ignore'))))
    return dict_genre, dict_directors, dict_actors, dict_writers


def get_revenue_amount(revenue_string):
    """
    Convert Revenue strint into revenue amount
    :param revenue_string:
    :return:
    """
    filtered_revenue_str1 = re.sub("\(Estimate\)", "", revenue_string)
    filtered_revenue_str2 = re.sub(",", "", filtered_revenue_str1)
    filtered_revenue_str3 = re.sub("\\$", "", filtered_revenue_str2)
    return int(filtered_revenue_str3)


def check_correct_date(date_str):
    try:
        datetime.datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False
    return False


def get_page_view_data(pv_str, release_date_str):
    """
    Get average page view data for last 8 weeks as a list of length 8
    :param pv_str:
    :param release_date_str:
    :return:
    """
    release_date = datetime.datetime.strptime(release_date_str, "%Y-%m-%d")
    split_data = [element.split(":") for element in pv_str.split(",")]
    page_view_daily_data = []
    for dt, pv in split_data:
        if check_correct_date(dt) is False:
            continue
        date_diff = (release_date - datetime.datetime.strptime(dt, "%Y-%m-%d")).days
        if date_diff >= 0:
            page_view_daily_data.append((date_diff, int(pv)))
    page_view_weekly_data = filter(lambda x: x[0] <= 8 and x[1] > 0,
                                   map(lambda x: (x[0] / 7 + 1, x[1]), page_view_daily_data))
    page_view_weekly_data.sort(key=lambda x: x[0])
    it = itertools.groupby(page_view_weekly_data, operator.itemgetter(0))
    pv_dict = {}
    for key, subiter in it:
        arr = [item[1] for item in subiter]
        pv_dict[key] = np.mean(arr)
    out = []
    for week in range(1, 9):
        if week in pv_dict:
            out.append(str(pv_dict[week]))
        else:
            out.append('0')
    return out


def merge_all_tables(details_table_path, revenue_table_path, out_file):
    file_details = pd.read_csv(details_table_path, sep="\t")
    file_details_complete = file_details.dropna()
    file_revenue = pd.read_csv(revenue_table_path)
    file_revenue_filtered = file_revenue[file_revenue['weekendRevenue'] != '-1']
    file_revenue_filtered = file_revenue_filtered[file_revenue_filtered['weekendRevenue'] != 'n/a']
    merged_table = pd.merge(file_revenue_filtered, file_details_complete, how='inner', on='title')
    merged_table.to_csv(out_file, sep='\t', quoting=csv.QUOTE_NONE)


def transform_raw_data(merged_file, processed_file_path):
    lines =open(merged_file).read().splitlines()
    list_director_dict=[]
    list_writer_dict=[]
    list_actor_dict=[]
    list_genre_dict=[]
    other_details_list=[]
    list_revenue=[]
    v = DictVectorizer(sparse=False)
    for line in lines:
        details = line.strip().split("\t")
        details_json = details[11]
        date_elements = details[2].strip().split("/")
        if len(date_elements) != 3 :
            continue
        month,day,year=date_elements
        month = data_constants.get_zero_padded_number_string(int(month),2)
        day = data_constants.get_zero_padded_number_string(int(day), 2)
        date="%s-%s-%s"%(year,month,day)
        if check_correct_date(date) is False:
            continue
        dict_genre, dict_directors, dict_actors, dict_writers = parse_details_json(details_json)
        list_director_dict.append(dict_directors)
        list_writer_dict.append(dict_writers)
        list_actor_dict.append(dict_actors)
        list_genre_dict.append(dict_genre)
        page_view_str=details[12]
        pv_weekly_data_str = "\t".join(get_page_view_data(page_view_str,date))
        title=details[1]
        type=details[10]
        year_vector=[0]*10
        month_vector=[0]*12
        year_vector[int(year)-2007]=1
        month_vector[int(month) - 1] = 1
        month_str="\t".join(map(lambda x : str(x),month_vector))
        year_str = "\t".join(map(lambda x : str(x),year_vector))
        list_revenue.append(get_revenue_amount(details[4]))
        other_details_list.append("%s\t%s\t%s\t%s\t%s"%(type,title,year_str,month_str,pv_weekly_data_str))
    list_director_vector=v.fit_transform(list_director_dict)
    director_columns="\t".join(map(lambda x : "Director_%s"%x,v.get_feature_names()))
    list_actor_vector = v.fit_transform(list_actor_dict)
    actor_columns="\t".join(map(lambda x : "Actor_%s"%x,v.get_feature_names()))
    list_writer_vector = v.fit_transform(list_writer_dict)
    writer_columns="\t".join(map(lambda x : "Writer_%s"%x,v.get_feature_names()))
    list_genre_vector = v.fit_transform(list_genre_dict)
    genre_columns="\t".join(map(lambda x : "Genre_%s"%x,v.get_feature_names()))
    year_columns = "\t".join(map(lambda x : "Year_%d"%x,range(2007,2017)))
    month_columns = "\t".join(map(lambda x: "Month_%d" %x, range(1, 13)))
    page_view_columns = "\t".join(map(lambda x : "Mean_PV_For_PrevWeek_%d"%x,range(1,9)))
    out_file = open(processed_file_path, 'w')
    out_file.write("Type\tTitle\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tWeekend_Revenue\n" %(year_columns,month_columns,page_view_columns,director_columns,actor_columns,writer_columns,genre_columns))
    for i in range(len(list_director_vector)):
        director_vector_str="\t".join(map(lambda x : str(x),list_director_vector[i]))
        actor_vector_str = "\t".join(map(lambda x : str(x),list_actor_vector[i]))
        writer_vector_str = "\t".join(map(lambda x : str(x),list_writer_vector[i]))
        genre_vector_str = "\t".join(map(lambda x : str(x),list_genre_vector[i]))
        out_file.write("%s\t%s\t%s\t%s\t%s\t%d\n"%(other_details_list[i],director_vector_str,actor_vector_str,writer_vector_str,genre_vector_str,list_revenue[i]))
    out_file.close()


if __name__ == '__main__':
    clean_wiki_omdb_raw_data(os.path.join(path_constants.DATA_FOLDER_PATH, path_constants.MOVIES_DETAILS_WIKI),os.path.join(path_constants.DATA_FOLDER_PATH, path_constants.MOVIES_DETAILS_WIKI_FILTERED))
    merge_all_tables(
        details_table_path=os.path.join(path_constants.DATA_FOLDER_PATH, path_constants.MOVIES_DETAILS_WIKI_FILTERED),
        revenue_table_path=os.path.join(path_constants.DATA_FOLDER_PATH, path_constants.MOVIE_REVENUE_DATA),
        out_file=os.path.join(path_constants.DATA_FOLDER_PATH, path_constants.MOVIE_MERGED_RAW_DATA_FILE))
    #using grep : Filter data for USA and English
    transform_raw_data(os.path.join(path_constants.DATA_FOLDER_PATH,"raw_data_usa_eng.txt"),os.path.join(path_constants.DATA_FOLDER_PATH,"processed_training_data.txt"))