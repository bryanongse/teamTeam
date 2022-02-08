# FOR SECURITYWEEK WEBSITE


import requests
from bs4 import BeautifulSoup
import pdfkit
import os
import re
import datetime
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Required download: wkhtmltopdf
path_wkhtmltopdf = 'C:\\Users\\daniel\\PycharmProjects\\pythonProject\\wkhtmltopdf\\bin\\wkhtmltopdf.exe'  # change path as required
config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

# Original Working Folder
owd = os.getcwd()
folder = os.path.join(owd, "SecurityWeek")
if not os.path.exists(folder):
    os.mkdir(folder)
os.chdir(folder)

# PDF Folder
pdf_folder = os.path.join(folder, "PDF_files")
if not os.path.exists(pdf_folder):
    os.mkdir(pdf_folder)

# HTML Folder
html_folder = os.path.join(folder, "HTML_files")
if not os.path.exists(html_folder):
    os.mkdir(html_folder)


def make_soup(url):
    data = requests.get(url).content
    soup = BeautifulSoup(data, 'html.parser')
    return soup


def get_title(soup):
    title = soup.find('h2', {'class': 'page-title'})
    return str(title)


def get_formatted_title(soup):
    title = soup.find('h2', {'class': 'page-title'}).get_text()
    title_str = title.replace("\n", " ")
    return title_str


def get_file_name(soup):
    title = soup.find('h2', {'class': 'page-title'}).get_text()
    title = re.sub("[',?!|/\"\"$&:—	]", '', title)
    file_name = title + ".pdf"
    duplicates = 1
    if os.path.exists(file_name):
        file_name = file_name[:-4] + "-" + str(duplicates) + file_name[-4:]
    return file_name


def get_date(soup):
    date = soup.find('div', {'class': 'submitted'}).find('div').get_text()
    date = re.sub('\n', '', date)
    date_str = str(date)
    return date_str


def get_formatted_date(soup):
    date = soup.find('div', {'class': 'submitted'}).find('div').get_text()
    date = re.sub('\n', '', date)
    date_str = str(date)
    date_str = date_str.split("on")[-1].strip()
    date_formatted = datetime.datetime.strptime(date_str, "%B %d, %Y").strftime("%m_%d_%Y")
    return date_formatted


def get_body(soup):
    body = soup.find('div', {'class': 'content clear-block'}).find_all('p')
    content = str(body)

    # Remove images
    images = soup.find('div', {'class': 'content clear-block'}).find_all('img')
    for i in images:
        string = str(i)
        content = content.replace(string, "")

    # Remove ads
    ads = soup.find('div', {'class': 'content clear-block'}).find_all('div', {'class': 'html-advertisement'})
    for h in ads:
        string = str(h.find('a'))
        content = content.replace(string, "")

    # Remove videos
    if "<iframe" in str(soup):
        videos = soup.find_all('iframe')
        for video in videos:
            string = str(video)
            content = content.replace(string, "")

    split = content.split("Related:")
    content = split[0]  # remove related article links
    content = content[1:]  # remove leading square bracket
    if content[-1] == "]":
        content = content[:-1]  # remove trailing square bracket
    content = '<head><meta charset="utf-8"></head>' + content  # allow non-ascii characters
    content = re.sub('</p>, ', '</p>', content)  # remove commas between paragraphs
    return content


def generate_pdf(url, date, title, body, file_name):
    os.chdir(pdf_folder)
    final_html = url + "<br><br>" + date + "<br>" + title + "<br>" + body
    pdfkit.from_string(final_html, file_name, configuration=config, options={'quiet': ''})
    os.chdir(folder)


def generate_html(url, date, title, body, html_name):
    html_file_name = os.path.join(html_folder, html_name)
    final_html = url + "<br><br>" + date + "<br>" + title + "<br>" + body
    # open file with *.html* extension to write html
    file = open(html_file_name, "w", encoding='utf-8')
    # write then close file
    file.write(final_html)
    file.close()


def crawl_sw(inputUrl, totalPages, jsonName):  # totalpages = no. of pages to crawl
    error_list = []
    url_table = pd.DataFrame(columns=["Title", "Date", "FileName", "HTMLName", "URL"])

    # Iterate each page
    for i in range(totalPages):  # page numbering starts from 0
        page_url = inputUrl + "?page=" + str(i)
        page_soup = make_soup(page_url)
        content_body = page_soup.find('div', {'class': 'panel-pane pane-block pane-views-news-industry-block-1'})
        articles = content_body.find_all('span', {'class': 'field-content'})


        # Iterate each article on page
        for line in articles:
            href = line.find('a').get('href')
            url = "https://www.securityweek.com" + href
            soup = make_soup(url)
            try:
                title = get_title(soup)
                date = get_date(soup)
                body = get_body(soup)
                file_name = get_file_name(soup)
                html_name = file_name[:-4] + ".html"
                generate_pdf(url, date, title, body, file_name)  # download pdf file
                generate_html(url, date, title, body, html_name)  # download html file
                logger.info("Crawled: {file_name}".format(file_name=file_name))

                # Append article details to json file
                title_text = get_formatted_title(soup)
                formatted_date = get_formatted_date(soup)
                try:
                    url_table = url_table.append({'Title': title_text, 'Date': formatted_date,
                                                  'FileName': file_name, "HTMLName": html_name,
                                                  'URL': url}, ignore_index=True)
                except:
                    os.remove(file_name)
            except:
                error_list.append(url)
                logger.error("Error: {url}".format(url=url))

    # Print no. of errors (if any)
    no_errors = len(error_list)
    logger.info('There were {0} errors.'.format(no_errors))

    # Convert url table to json file
    json_file_name = os.path.join(folder, jsonName)
    url_table.to_json(json_file_name, orient='records')


# # Vulnerabilities
# crawl_sw(inputUrl="https://www.securityweek.com/virus-threats/vulnerabilities",
#          totalPages=30,  # 1 page = 10 articles, max 459
#          jsonName="JSON_SecurityWeek_vulnerabilities.json")
#
# # Email Security
# crawl_sw(inputUrl="https://www.securityweek.com/virus-threats/email-security",
#          totalPages=15,  # 1 page = 10 articles, max 33
#          jsonName="JSON_SecurityWeek_emailsecurity.json")
#
# # Virus and Malware
# crawl_sw(inputUrl="https://www.securityweek.com/virus-threats/virus-malware",
#          totalPages=20,  # 1 page = 10 articles, max 126
#          jsonName="JSON_SecurityWeek_virusmalware.json")
#
# # IoT Security
# crawl_sw(inputUrl="https://www.securityweek.com/virus-threats/white-papers",
#          totalPages=3,  # 1 page = 10 articles, max 3
#          jsonName="JSON_SecurityWeek_iotsecurity.json")
#
# Threat Intelligence
crawl_sw(inputUrl="https://www.securityweek.com/cybercrime/whitepapers",
         totalPages=1,  # 1 page = 10 articles, max 2
         jsonName="JSON_SecurityWeek_threatintelligence.json")

# # Endpoint Security
# crawl_sw(inputUrl="https://www.securityweek.com/virus-threats/endpoint-security",
#          totalPages=15,  # 1 page = 10 articles, max 60
#          jsonName="JSON_SecurityWeek_endpointsecurity.json")



