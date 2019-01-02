import os
import json
import codecs
from datetime import datetime
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


with open('food.json', encoding='utf-8', errors='ignore') as a, open('hotel.json', encoding='utf-8', errors='ignore') as b, open('travel.json', encoding='utf-8', errors='ignore') as c:
  data1 = json.load(a)
  data2 = json.load(b)
  data3 = json.load(c)

#create stopword removal
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

#create stemmer
stem = StemmerFactory()
stemmer = stem.create_stemmer()

def prep(text):
  text = stopword.remove(text)
  text = stemmer.stem(text)
  return " ".join(text.split())

prepro
merge = []
#reading file
for line in data1:
  line['text'] = prep(line['text'])
  #print(line['text'])
  temp={}
  temp["id"] = line["id"]
  temp["title"] = line["title"]
  temp["subtitle"] = line["subtitle"]
  temp["date"] = line["date"]
  temp["text"] = line["text"]
  temp["tag"] = line["tag"]
  temp["label"] = "Food"
  #make new dictionary
  merge.append(temp)
  #print(merge)

#reading file
for line in data2:
  line['text'] = prep(line['text'])
  # print(line['text'])
  temp={}
  temp["id"] = line["id"]
  temp["title"] = line["title"]
  temp["subtitle"] = line["subtitle"]
  temp["date"] = line["date"]
  temp["text"] = line["text"]
  temp["tag"] = line["tag"]
  temp["label"] = "Hotel"
  #make new dictionary
  merge.append(temp)

#reading file
for line in data3:
  line['text'] = prep(line['text'])
  # print(line['text'])
  temp={}
  temp["id"] = line["id"]
  temp["title"] = line["title"]
  temp["subtitle"] = line["subtitle"]
  temp["date"] = line["date"]
  temp["text"] = line["text"]
  temp["tag"] = line["tag"]
  temp["label"] = "Travel"
  #make new dictionary
  merge.append(temp)
  #sort the dictionary according to date
  sortdict = sorted(merge, key = lambda i: datetime.strptime(i["date"], "%d/%m/%Y, %H:%M WIB"))

#write new file for model
with open('test2.json', 'w') as fp:
    json.dump(sortdict, fp)

a.close()
b.close()
c.close()
fp.close()