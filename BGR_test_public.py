from datetime import datetime
from gspread_dataframe import set_with_dataframe
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from bs4 import BeautifulSoup
import gspread
import pandas as pd
import requests
import re
import numpy as np

#####

# The main objective for the program is to automate the addition of flights to an existing spreadsheet.
# Previously, there were nine (reduced to seven with datetime and ML, respectively) fields that had to be filled out per observation:
# 
# 	# Date (if none, current datetime date entered)
# 	# Airline (if applicable)
#	# Flight # (if applicable)
# 	# Aircraft type (required)
# 	# Origin IATA code
# 	# Origin Country
# 	# Destination IATA code
# 	# Destination Country
# 	# Direction (predicted by ML)
# 
# With this program, we will scrape the FlightAware API to see if there are any new transatlantic flights to add to our spreadsheet.
# If there are no flights to add, the program exits and nothing changes.
# If there are flights to add, it will format them properly by (future plans after "EVENTUALLY" if applicable):
#    	
# 	# Reducing the departure/arrival times to strf dates
#	# Splitting the ICAO identifier into an Airline and Flight # (if applicable) ||||| EVENTUALLY: converting airline via dictionary
#  	# Recording aircraft type ||||| EVENTUALLY: converting aircraft type via dictionary
#  	# Converting ICAO codes to IATA codes via dictionary
# 	# Adding origin and destination countries (derived from IATA codes via dictionary)
# 	# Predicting direction based on origin country (e.g. If flight originates in USA, it's "E" for East; otherwise, "W" for West.)

##### This is where the new code comes in 19:42 12/8/2021

url = "https://flightxml.flightaware.com/json/FlightXML2/"
user = "user"
key = "key"
payload = {"airport":"KBGR", "howMany":15}

common_cols = ['ident','aircrafttype','actualdeparturetime','actualarrivaltime','origin','destination',]

##### Arrival and Departure class creation 20:27 12/18/2021

# DataFrames are constructed slightly differently based on whether we're dealing with an arrival or departure, so two classes makes the most sense.
# Even so, they retain a lot of the same properties.

class Arrivals:

	def __init__(self):
		self.url = url
		self.user = user
		self.key = key
		self.payload = payload
		self.req = requests.get(self.url + "Arrived", params=self.payload, auth=(self.user, self.key)).json()
		self.df = pd.DataFrame(self.req['ArrivedResult']['arrivals'])

class Departures:

	def __init__(self):
		self.url = url
		self.user = user
		self.key = key
		self.payload = payload
		self.req = requests.get(self.url + "Departed", params=self.payload, auth=(self.user, self.key)).json()
		self.df = pd.DataFrame(self.req['DepartedResult']['departures'])

arr = Arrivals().df
dep = Departures().df

# We append the arrival dataframe to the departure dataframe as new_df.
new_df = arr.append(dep).sort_values(by="actualdeparturetime")

# estimatedarrival time assumes arrivaltime as its column name 20:29 12/18/2021
new_df['arrivaltime'] = [new_df['estimatedarrivaltime'] if 0 else i for i in new_df['actualarrivaltime']]

# Getting rid of estimated and actual arrival times and replacing them with a single arrival time.
new_df = new_df.drop(['estimatedarrivaltime','actualarrivaltime'], axis=1)

# Strftime formatting added 03:02 12/12/2021.
new_df["actualdeparturetime"] = new_df["actualdeparturetime"].apply(lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d"))
new_df["arrivaltime"] = new_df["arrivaltime"].apply(lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d"))

# Sort by actual_departure_time DESCENDING 08:52 12/11/2021
new_df = new_df.sort_values(by=['actualdeparturetime'], ascending=False)

# If there are no flights, we exit the program and the Google API does not run.
if len(new_df) == 0:
	print("No flights added.")
	exit()
else:
	pass

# Ident lists created 19:52 12/16/2021
idents_a = []
idents_b = []

# Here we want to parse the airline and flight number from the identifier.
# We get all the first parts of the identifiers here, so it will separate the text first part (denoting the airline) from the flt # (if applicable).
for col in new_df['ident']:
	idents_a.append(re.split("(\d+)", col)[0])

# Reworked idents_b to handle IndexError 17:11 12/17/2021
# Some of the identifiers are solely letters (e.g. a German plane with the registration "DABVA") meaning there is no second part so we need to handle these errors.
for col in new_df['ident']:
	try:
		idents_b.append(re.split("(\d+)", col)[1])
	except IndexError:
		idents_b.append("None")

# Let's create the new dataframe! This is the dataframe for the flights to be added 21:32 12/16/2021
bgr = pd.DataFrame()
bgr['Date'] = new_df['actualdeparturetime'].str.strip()

# Splitting idents and and b into columns 21:36 12/16/2021
bgr['Airline'] = idents_a
bgr['Flight'] = idents_b

# Creating more columns 16:51 12/17/2021
bgr['Type'] = new_df['aircrafttype'].str.strip()

# Import the existing DataFrame 21:19 12/19/2021
gc = gspread.service_account()
sh = gc.open("#sheet#")
ws = sh.get_worksheet(0)
df = pd.DataFrame(ws.get_all_records())

# Dictionary creation for origins/destinations and their countries. 20:35 12/18/2021
origins = dict(zip(df['Origin'],df['Origin Country']))
destinations = dict(zip(df['Destination'],df['Destination Country']))

# Stackoverflow Code 20:57 12/18/2021
# We only want one dictionary with no duplicates so we'll just merge them.
def Merge(dict_1, dict_2):
	result = dict_1 | dict_2
	return result

# Implementation. 20:57 12/18/2021
combined_dict = Merge(origins, destinations)

# Adding origins and destinations 15:24 12/19/2021
bgr['Origin'] = new_df['origin'].str.strip()
bgr['Destination'] = new_df['destination'].str.strip()

# Cutdown 19:26 12/19/2021
# When we import the FA query, we want to be sure it has only true European airports. 
# This means no ICAO identifiers starting with "K" (USA), "C" (CAN), or "M" (MEX), and no flights with solely lat/long listed as O or D.
bgr = bgr[((bgr['Origin'].str[0] != "K") & (bgr['Origin'].str[0] != "C") & (bgr['Origin'].str[1] != " ") & (bgr['Origin'].str[0] != "M")) | ((bgr['Destination'].str[0] != "K") & (bgr['Destination'].str[0] != "C") & (bgr['Destination'].str[1] != " ") & (bgr['Destination'].str[0] != "M"))]

# Get rid of any stray medical flights, too.
# They seem to sneak in if there is no destination but they all fall under the same ident # of 901.
bgr = bgr[(bgr['Flight'] != "901") | (bgr['Airline'] != "N")]

# The airport codes in the spreadsheet are IATA (3-letter) whereas the pulls from FA are ICAO (4-letter). We will use a table to create a dictionary that maps the ICAO to the IATA.
# Scraping 15:39 12/19/2021

list_url = "http://www.flugzeuginfo.net/table_airportcodes_country-location_en.php"
list_req = requests.get(list_url)
soup = BeautifulSoup(list_req.content, "html.parser")

sfa = soup.find_all('table')

A = []
B = []

# There are a crazy number of tables on the single page, so we go the entirety of the tables when creating our ICAO:IATA pairs.
for i in range(len(sfa)):
	for row in sfa[i].findAll('tr'):
		cells = row.findAll('td')
		if len(cells) == 5:
			A.append(cells[0].text.strip())
			B.append(cells[1].text.strip())
		else:
			pass

# Creating our dictionary
icao_iata_dict = dict(zip(B, A))

# Successfully mapped to IATAs from ICAO 19:08 12/19/2021
bgr['Origin'] = bgr['Origin'].map(icao_iata_dict)
bgr['Destination'] = bgr['Destination'].map(icao_iata_dict)
bgr['Origin Country'] = bgr['Origin'].map(combined_dict)
bgr['Destination Country'] = bgr['Destination'].map(combined_dict)

bgr = bgr[['Date','Airline','Flight','Type','Origin','Origin Country','Destination','Destination Country']]

# ML model to predict direction column based on Origin Country.
y_final = df[['Direction']]

# Instantiating LabelEncoder
le = LabelEncoder()

# We only need one predictor variable, really, so Origin Country will do
X_final = le.fit_transform(df['Origin Country'].astype(str)).reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2)

# The Scikit-learn RF and GB classifiers tend to undersample East flights, so we use the Balanced RF classifier from imbalanced-learn.
model = BalancedRandomForestClassifier()
model.fit(X_train, y_train.values.ravel())

# Must transform the origin countries in the new flights to the LE scheme that currently exists
new_predicts = le.transform(bgr['Origin Country']).astype(str).reshape(-1,1)

# New flights direction predicted
bgr['Direction'] = model.predict(new_predicts)

# Combining the existing df with the new flights.
df = df.append(bgr).sort_values(by=['Date']).reset_index(drop=True)

# Setting with dataframe 20:34 12/19/2021
set_with_dataframe(ws, df)

print(f"{len(bgr)} new flights added. {len(df)} flights total.")

exit()



