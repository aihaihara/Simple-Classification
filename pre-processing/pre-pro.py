import os
import json
import codecs
from datetime import datetime
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

#open data
with open('JSON file', encoding='utf-8', errors='ignore') as a:
  data1 = json.load(a)

#create stopword remover
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

#create stemmer
stem = StemmerFactory()
stemmer = stem.create_stemmer()

#make a function that combines stopword remover and stemmer
def prep(text):
  text = stopword.remove(text)
  text = stemmer.stem(text)
  return " ".join(text.split())

#prepro
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

  #sort the dictionary according to date
  sortdict = sorted(merge, key = lambda i: datetime.strptime(i["date"], "%d/%m/%Y, %H:%M WIB"))

#write new file for model
with open('test.json', 'w') as fp:
    json.dump(sortdict, fp)

a.close()
fp.close()
