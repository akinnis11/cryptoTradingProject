
import requests
import json
import pandas as pd
from pytrends.request import TrendReq

def getPriceData(url):

	response = requests.get(url)
	data = response.json()
	json_string = json.dumps(data)
	datastore = json.loads(json_string)
	df = pd.DataFrame.from_dict(datastore, orient='columns', dtype=None)
	df = df.drop(['disclaimer','time'],axis=1)
	df = df.drop(['updated','updatedISO'])
	df.columns = ['price']
	df['price'] = 100*(df['price'] / df['price'].max()) # normalize price data to 0-100 to match good trend data

	return df


def trendData(term_list,pytrends):


	#Use pytrends to get data from google trends API about interest in bitcoin over time
	# Only need to run this once, the rest of requests will use the same session.
	if pytrends:
		print("Already started session")
	else:
		print("New session")
		pytrends = TrendReq()

	# Create payload and capture API tokens. Needed for interest_over_time() & related_queries()
	pytrends.build_payload(kw_list=term_list)

	# Interest Over Time
	df = pytrends.interest_over_time()
	df.drop(['isPartial'],axis=1,inplace=True)

	return df, pytrends

def relatedData(currency,nterms,pytrends):

	related_queries_df = pytrends.related_queries()
	rlist_top = related_queries_df[currency]['top']['query'].tolist() # get list of related query terms
	rlist_rising = related_queries_df[currency]['rising']['query'].tolist() # get list of related query terms

	#rlist = rlist.tolist()
	
	df_top = pd.DataFrame()
	df_rising = pd.DataFrame()

	if nterms <= 5:
		
		df_top, p = trendData(rlist_top[0:nterms],pytrends) # if only getting top 5 terms, can get with one call
		df_rising, p = trendData(rlist_rising[0:nterms],pytrends) # if only getting top 5 terms, can get with one call
	
	else:
		
		for term in rlist_top[0:nterms]: # if more than 5, loop through and get each individually
			temp, p = trendData([term],pytrends)
			df_top = pd.concat([df_top,temp],axis=1)

		for term in rlist_rising[0:nterms]: # if more than 5, loop through and get each individually
			temp, p = trendData([term],pytrends)
			df_rising = pd.concat([df_rising,temp],axis=1)
	
	# Add +1 to all google trend data columns but keep largest value at 100 (necessary for normalizing later) 
	df_rising = df_rising + 1
	df_rising[df_rising > 100] = 100

	df_top = df_top + 1
	df_top[df_top > 100] = 100


	return df_top, df_rising


	
