import asyncio

from pymongo import MongoClient
from twikit import Client

mongo_client = MongoClient("localhost", 27017)
db = mongo_client["tweet-o-mat"]
collection = db["tweets"]


async def login(auth_info_1: str, password: str) -> Client:
    """
    Log into the twitter API with the given auth_info_1 and password using the twikit library.
    """
    client = Client()
    await client.login(auth_info_1=auth_info_1, password=password)
    return client


async def fetch_and_store_tweets(
    username: str, client: Client, label: str, count: int
) -> None:
    """
    Fetch tweets from a user and store them in the database using the twikit library.
    """
    user = await client.get_user_by_screen_name(username)
    tweets = await user.get_tweets(tweet_type="Tweets", count=count)
    stored_tweets = 0
    while stored_tweets < count:
        for tweet in tweets:
            stored_tweets += 1
            collection.insert_one(
                {
                    "text": tweet.text,
                    "label": label,
                    "created_at": tweet.created_at,
                    "author": username,
                }
            )
        tweets = await tweets.next()


async def fetch_and_store_german_politics_tweets() -> None:
    """
    Fetch tweets from german politicians and store them in the database.
    """
    gruenen_handles = [
        "roberthabeck",
        "Die_Gruenen",
        "al_baerbock",
        "GoeringEckardt",
        "BriHasselmann",
    ]
    cdu_handles = [
        "_FriedrichMerz",
        "Markus_Soeder",
        "ArminLaschet",
        "CDU",
        "CSU",
    ]
    afd_handles = [
        "Alice_Weidel",
        "AFDimBundestag",
        "Bjoern_Hoecke",
        "Tino_Chrupalla",
        "DrBerndBaumann",
    ]
    spd_handles = [
        "OlafScholz",
        "spdde",
        "MartinSchulz",
        "KatjaMast",
        "spdbt",
    ]
    fdp_handles = [
        "MarcoBuschmann",
        "c_lindner",
        "fdp",
        "starkwatzinger",
        "KubickiWo",
    ]

    client = await login("<auth_info_1>", "<password>")

    for handle in gruenen_handles:
        await fetch_and_store_tweets(handle, client, "die grünen", 200)
        await asyncio.sleep(60 * 20)  # prevent rate limit

    for handle in cdu_handles:
        await fetch_and_store_tweets(handle, client, "cdu", 200)
        await asyncio.sleep(60 * 20)

    for handle in afd_handles:
        await fetch_and_store_tweets(handle, client, "afd", 200)
        await asyncio.sleep(60 * 20)

    for handle in spd_handles:
        await fetch_and_store_tweets(handle, client, "spd", 200)
        await asyncio.sleep(60 * 20)

    for handle in fdp_handles:
        await fetch_and_store_tweets(handle, client, "fdp", 200)
        await asyncio.sleep(60 * 20)
