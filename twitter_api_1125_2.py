#!/usr/bin/env python
# coding: utf-8

import json
import tweepy
from tweepy import Stream
from tweepy.streaming import StreamListener
import time
from datetime import timedelta
import pandas as pd
key = #key#
secret = #secret#
at = #at#
ats = #ats#
auth  = tweepy.OAuthHandler(key, secret)
auth.set_access_token(at, ats)
tracked = input("Enter Search Term: ")
timez = int(input("Enter Time: "))

class MyStreamListener(tweepy.StreamListener):
    def __init__(self, time_limit=timez):
        self.start_time = time.time()
        self.limit = time_limit
        self.saveFile = open('tw33ts.json', 'a')
        super(MyStreamListener, self).__init__()

    def on_data(self, data):
        if (time.time() - self.start_time) < self.limit:
            self.saveFile.write(data)
            self.saveFile.write('\n')
            return True
        else:
            self.saveFile.close()
            return False

myStream = tweepy.Stream(auth=auth, listener=MyStreamListener(time_limit=timez))
myStream.filter(track=[tracked])

tweet_stream = pd.read_json('tw33ts.json', lines=True)

print(tweet_stream)

exit()

