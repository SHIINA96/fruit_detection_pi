from dotenv import load_dotenv
import pymongo, os

load_dotenv()

myclient = pymongo.MongoClient(os.getenv("DB_URL"))
mydb = myclient[os.getenv("DB")]
mycol = mydb[os.getenv("COLLECTION")]

# delete all data in collection
mycol.drop()    

# another approach to delete all data in collection
# delete all data in collection
# x = mycol.delete_many({})
# print(x.deleted_count, " documents deleted.")