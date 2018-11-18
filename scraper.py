"""
This program scrapes news articles from NewsAPI and saves them to the 
data folder. We obtain the 1000 most recent articles from the following
providers:
- NYT
- WaPost
- Politico
- The Hill
- National Review
- Fox News

We save the articles as json objects.
"""

from newsapi import NewsApiClient
import json

API_KEY = 'ea9f3ca698cb4cad9277ef52d085111d'
SOURCES = ['the-new-york-times', 'politico', 'the-washington-post', 'the-hill', 'national-review', 'fox-news']
PAGE_SIZE = 100
NUM_PAGES = 10

newsapi = NewsApiClient(api_key=API_KEY)

for source in SOURCES:
	article_list = []
	for i in range (1, NUM_PAGES + 1):
		res = newsapi.get_everything(sources=source, page_size=PAGE_SIZE, page=i)
		articles = res['articles']
		for page in articles:
			article_ts = {
				"source": page['source']['id'],
				"title": page['title'],
				"desc": page['description'],
				"content": page['content'],
				"url": page['url'],
				"date": page['publishedAt']
			}
			article_list.append (article_ts)
	print("Number of articles for " + source + ": " + str(len(article_list)))
	with open('data/' + source + '.json', 'w') as fout:
		json.dump(article_list, fout)