import datetime
import os
import re
import sys

import requests
from bs4 import BeautifulSoup
sys.path.append(os.path.join(os.getcwd(), '../common'))
sys.path.append(os.path.join(os.getcwd(), '../utils'))
import data_constants
import path_constants
import url_constants
import json



def get_monthly_page_views(movie_title, year, month):
    """
    Getting a page view data for the given movie title on the given year and month
    :param movie_title:
    :param year:
    :param month:
    :return:
    """
    url = url_constants.wiki_data_dump_base_url + "/" + data_constants.get_zero_padded_number_string(year,
                                                                                                     4) + data_constants.get_zero_padded_number_string(
        month, 2) + "/" + movie_title
    content = requests.get(url).content
    soup = BeautifulSoup(content, 'html.parser')
    out = soup.find_all("script")
    page_view_line = ""
    for element in out:
        content = element.contents
        if content is not None and len(content) > 0 and "line1" in content[0]:
            lines = content[0].split("\n")
            for line in lines:
                if line.strip().startswith("line1"):
                    page_view_line = line
    return parse_page_views_from_line(page_view_line)


def parse_page_views_from_line(line):
    """
    Extract date and page view from the string using Regular Expressions
    :param line:
    :return:
    """
    date_pv = map(lambda x: (x[0], int(x[1])), re.findall("'(\d{4}-\d{2}-\d{2}) *\d{1,2} *AM', *(\d+)", line))
    valid_result = any(x[1] > 0 for x in date_pv)
    if valid_result is True:
        return sorted(date_pv, key=lambda x: x[0])
    else:
        return []


def get_date(date_str):
    dt = datetime.datetime.strptime(date_str, "%d %b %Y").date()
    return dt


def older_month_year(date, num_months):
    out = set([])
    for i in range(num_months+1):
        old_date= (date + datetime.timedelta(days=-i * 31))
        out.add((old_date.year,old_date.month))
    return out



def get_wiki_format_names(movie_title, year):
    edited_movie_title = "%s_(%d_film)" % (movie_title, year)
    return edited_movie_title


if __name__ == '__main__':
    file_path = path_constants.MOVIES_DATA_PATH + "/" + path_constants.MOVIES_DETAILS_OMDB
    out_file_path = path_constants.MOVIES_DATA_PATH + "/" +path_constants.MOVIES_DETAILS_WIKI
    out_file = open(out_file_path, "w+")
    with open(file_path) as f:
        for line in f:
            movie_title,year,data_type,content = line.split("\t")
            if content.strip() == "None":
                continue
            content_dict= json.loads(content)
            if "Released" not in content_dict or content_dict["Released"] == "N/A":
                continue
            date_release = get_date(content_dict["Released"])
            prev_3_months_data=older_month_year(date_release,3)
            page_view_data = []
            for yr,mth in prev_3_months_data:
                pv_data = get_monthly_page_views(get_wiki_format_names(movie_title, yr),yr,mth)
                if len(pv_data) == 0:
                    pv_data = get_monthly_page_views(movie_title,yr,mth)
                page_view_data += pv_data
            page_view_data_str = (",".join(map(lambda x: x[0]+":"+str(x[1]),page_view_data))).encode("utf-8")
            print "Wiki PV Output for Movie :%s" %movie_title
            out_file.write("%s\t%s\t%s\t%s\t%s\n" % (movie_title, year, data_type,content,page_view_data_str))
            out_file.flush()
    out_file.close()