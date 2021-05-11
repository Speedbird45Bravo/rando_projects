#!/usr/bin/env python
# coding: utf-8

import json
import tweepy
from tweepy import Stream
from tweepy.streaming import StreamListener
import time
from datetime import timedelta
import pandas as pd
key = 'mSBosjhTDLJd8hJhabBBNZxeR'
secret = '2X6VH6sG4JKbrk3EM986I8idmpbEa03nPI6qiTLLf8ibSz1sjJ'
at = '1134935674803691526-SUXYhZZQabanxojKI2EFwvojn4Qs14'
ats = 'jxNiesNnhhM46PfqEyRIiR1YDq8r82zN6VX8SG8cX2gLB'
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



