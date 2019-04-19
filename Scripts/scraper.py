'''
This script scrapes malware-traffic-analysis.net for pcap zip files
and downloads it in a folder
'''
from bs4 import BeautifulSoup
import pickle
from tqdm import tqdm
import requests
import time
import os

PICKLE_PATH = 'pickles/'
ZIP_DOWNLOAD_PATH = 'zips_from_MTA/'

def downloadZips():
    print("Downloading the zip files")

    pkl_file = open(PICKLE_PATH+'downloadLinks.pkl', 'rb')
    downloadLinks = pickle.load(pkl_file)

    # print(len(downloadLinks))

    for i, link in enumerate(downloadLinks):

        if i < 1433:
            continue

        file_name = link.split('/')[-1]
        print(i, file_name,link)

        try:
            response = requests.get(link, stream=True)
            # print(response.status_code)
            if response.status_code == 200:
                with open(ZIP_DOWNLOAD_PATH + file_name, 'wb') as handle:
                    for data in tqdm(response.iter_content()):
                        handle.write(data)
            time.sleep(3)
        except Exception as ex:
            print(ex)
            pass

def crawl_links():

    print("Crawling the pages to collect download links")

    baseLink = 'https://malware-traffic-analysis.net/2013/index.html'
    listOfLinks = ['https://malware-traffic-analysis.net/2013/index.html',
                   'https://malware-traffic-analysis.net/2014/index.html',
                   'https://malware-traffic-analysis.net/2015/index.html',
                   'https://malware-traffic-analysis.net/2016/index.html',
                   'https://malware-traffic-analysis.net/2017/index.html',
                   'https://malware-traffic-analysis.net/2018/index.html']

    # listOfLinks = ['https://malware-traffic-analysis.net/2014/index.html']
    downloadLinks = []

    for link in listOfLinks:
        baseLink = link.split('index')[0]
        # print(link.split('.net/')[1])
        r = requests.get(link)
        soup = BeautifulSoup(r.content)
        individualLinkTags = soup.find_all('li')
        individualLinks = []
        for descLink in individualLinkTags:
            individualLinks.append(baseLink + descLink.find('a').get('href'))
        # print(individualLinks)
        for link2 in individualLinks:

            r2 = requests.get(link2)
            soup = BeautifulSoup(r2.content)
            # print(link2)
            try:
                downloadLinkTags = soup.find('div', class_='blog_entry').find_all('li')
                for pcapLink in downloadLinkTags:
                    if 'pcap' in pcapLink.text.lower() and 'unfortunately' not in pcapLink.text.lower():
                        # print(pcapLink.text)
                        # print(link2.split('index')[0] + pcapLink.find('a').get('href'))
                        downloadLinks.append(link2.split('index')[0] + pcapLink.find('a').get('href'))
                        # with pkl_file = open('downloadLinks.pkl', 'wb')
            except:
                pass
    with open(PICKLE_PATH+'downloadLinks.pkl', 'wb') as pkl_file:
        pickle.dump(downloadLinks, pkl_file)
    downloadLinks = list(set(downloadLinks))
    # print(len(downloadLinks))

    # print(downloadLinkTag)

def main():
    os.makedirs(PICKLE_PATH, exist_ok=True)
    os.makedirs(ZIP_DOWNLOAD_PATH, exist_ok=True)

    user_input = str(input("Press \"y\" to download all the zip files. Press any other key to proceed further"))
    if user_input == 'y':
        crawl_links()
        downloadZips()
    else:
        return

if __name__ == '__main__':
    main()


'''
Next steps --

'''