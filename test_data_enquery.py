from dotenv import load_dotenv
import pymongo, os

load_dotenv()

myclient = pymongo.MongoClient(os.getenv("DB_URL"))
mydb = myclient[os.getenv("DB")]
mycol = mydb[os.getenv("COLLECTION")]

def find_the_fruit(fruit_name):
    myquery = { "fruit": { "$regex": "^"+fruit_name } }
    mydoc = mycol.find(myquery)
    result = []
    for x in mydoc:
        result.append(x)
    result.sort(key=lambda x: x['price'])   # sort by price
    price = result[0]['price']
    provider = result[0]['provider']

    return '{}/{} at {}'.format(price,'kg',provider)

print(find_the_fruit('apple'))