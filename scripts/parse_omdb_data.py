import json
import os
import re
import sys

sys.path.append(os.path.join(os.getcwd(), '../common'))
sys.path.append(os.path.join(os.getcwd(), '../utils'))
import data_constants
import path_constants
import requests
import csv

movies_set = set([])


def check_braces(movie_title):
    if len(re.findall(r'{.*}', movie_title)) > 0:
        return True
    else:
        return False


def check_tv(movie_title):
    if len(re.findall(r'\(T?VG?\)', movie_title)) > 0:
        return True
    else:
        return False


def check_hash_tag(movie_title):
    if len(re.findall(r'^"#?', movie_title)) > 0 or movie_title.startswith("#"):
        return True
    else:
        return False


def check_omdb_response(response_dict):
    if response_dict["Response"] == 'False':
        return False
    elif response_dict["Type"] != "movie":
        return False
    # elif 'USA' in response_dict["Country"]:
    #     return False
    else:
        return True


def get_data_type(year):
    if year <= data_constants.TRAINING_END_YEAR and year >= data_constants.TRAINING_START_YEAR:
        return "Train"
    elif year <= data_constants.TEST_END_YEAR and year >= data_constants.TEST_START_YEAR:
        return "Test"
    else:
        return None


def get_movie_name_year(movie_title):
    year_array = re.findall(r'\(\d+/?I?\)', movie_title)
    if len(year_array) == 0:
        print "No year in : %s" % movie_title
        return movie_title, 0
    else:
        year = year_array[0].lstrip("(").rstrip(")")
        if year.endswith("I"):
            year = year[0:year.find("/")]
        movie_name = re.split(r'\(\d+\)', movie_title)[0].strip()
        return movie_name, year


def get_omdb_data(file_path, out_file_path):
    out_file = open(out_file_path, "w+")
    with open(file_path) as f:
        for line in f:
            line = re.sub(r'\t+', '\t', line)
            details = line.split("\t")
            if len(details) != 2:
                continue
            raw_movie_title = details[0]
            if check_braces(raw_movie_title) is False and check_tv(raw_movie_title) is False and check_hash_tag(
                    raw_movie_title) is False:
                movie_name, year = get_movie_name_year(raw_movie_title)
                data_type = get_data_type(int(year))
                if data_type is None:
                    continue
                content = get_omdb_movie_data(movie_name, year)
                if content is None:
                    continue
                out_file.write("%s\t%s\t%s\t%s\n" % (movie_name, year, data_type, content))
                out_file.flush()
    out_file.close()


def get_omdb_data_box_office_mojo(infile_path, out_file_path):
    set_parsed_files =set(map(lambda x: x.split("\t")[0],open(out_file_path,'r').read().splitlines()))
    out_file = open(out_file_path, "a+")
    with open(infile_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
        for details in reader:
            raw_movie_title = details[0]
            if check_braces(raw_movie_title) is False and check_tv(raw_movie_title) is False and check_hash_tag(
                    raw_movie_title) is False:
                if len(details[1].split("/")) > 1:
                    movie_name, year = raw_movie_title, details[1].split("/")[2]
                else:
                    movie_name, year = raw_movie_title, details[1]
                if movie_name in set_parsed_files:
                    continue
                data_type = get_data_type(int(year))
                if data_type is None:
                    continue
                content = get_omdb_movie_data(movie_name, year)
                if content is None:
                    continue
                if check_content(content) is False:
                    content = get_omdb_movie_data(movie_name,str(int(year)-1))
                out_file.write("%s\t%s\t%s\t%s\n" % (movie_name, year, data_type, content))
                out_file.flush()
    out_file.close()


def check_content(content):
    content_dict = json.loads(content)
    if content_dict["Released"] == "N/A":
        return False
    else:
        return True

def get_omdb_movie_data(movie_title, year):
    url = "http://www.omdbapi.com/?t=%s&y=%s&plot=short&r=json" % (movie_title, year)
    try:
        movie_title.decode('utf-8')
    except UnicodeDecodeError:
        return None
    print "Getting details for movie title : ", movie_title
    try:
        content = requests.get(url).content
        json_output = json.loads(content)
        if check_omdb_response(json_output) is True:
            return content
        else:
            return None
    except Exception:
        return None


if __name__ == '__main__':
    file_path = path_constants.MOVIES_DATA_PATH + "/" + path_constants.MOVIES_DATA_BOX_OFFICE_MOJO
    out_file_path = path_constants.MOVIES_DATA_PATH + "/movie_title_details_box_office_mojo.txt"
    get_omdb_data_box_office_mojo(file_path, out_file_path)
