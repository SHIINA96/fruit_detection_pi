from random import randint, choice, random
from dotenv import load_dotenv
import pymongo, os, re, certifi

load_dotenv()

myclient = pymongo.MongoClient(os.getenv("DB_URL"), tlsCAFile=certifi.where())
mydb = myclient[os.getenv("DB")]
mycol = mydb[os.getenv("COLLECTION")]

# Set data and add into collection
fruit_list = ['apple','banana','orange']
units = ['kg','500g','each']
provider_list = ['park n save','new world','countdown','warehouse']

# check if unit is gram
pattern = r"^[^k]*g$" 

count_id = 1
mylist = []
for x in range(3):
    for y in range(5):
        weight = round(random()*10,2)
        fruit_name = fruit_list[x] + '-' + str(y)
        fruit_price = randint(200,500)/100.00
        unit = choice(units)
        if re.match(pattern, unit):
            fruit_price = fruit_price/weight*1000
            unit = 'kg'
        if unit == 'each':
            fruit_price = fruit_price/weight
            unit = 'kg'
        mydict = { "_id":count_id, "fruit": fruit_name, "price": round(fruit_price,2), "weight":weight, "unit": unit, "provider": choice(provider_list)}
        count_id += 1
        mylist.append(mydict)
        print(mydict)

x = mycol.insert_many(mylist)