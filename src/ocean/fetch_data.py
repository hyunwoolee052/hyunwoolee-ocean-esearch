import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

def fetch_data(id, key, sdate, edate):
    """
    Fetches data from the NIFS OpenAPI.
    
    Args:
        id (str): Identifier for the type of data to fetch.
        key (str): Your API key for the NIFS OpenAPI.
        sdate (str): Start date in 'YYYYMMDD' format.
        edate (str): End date in 'YYYYMMDD' format.
        
    Returns:
        str: The response from the API as a string.
    
    example:
    >>> fetch_data(id="sooList", key="qPwOeIrU-2506-KUVBOC-1211", sdate="20231201", edate="20231231")
    """
    data = {}
    data["id"] = id  # NIFS Serial Oceanographic observation data
    # data["id"] = "sooCode" # NIFS SOO Station information
    # NIFS OpenAPI key
    # Note: If the key is expired, update it
    #TODO: Create serviceKey.txt file with the key and expiration date
    data["key"] = key  # OpenAPI key
    data["sdate"] = sdate  # YYYYMMDD
    data["edate"] = edate  # YYYYMMDD
    url_values = urllib.parse.urlencode(data)
    url = f"https://www.nifs.go.kr/bweb/OpenAPI_json"
    full_url = f"{url}?{url_values}"
    with urllib.request.urlopen(full_url) as response:
        the_page = response.read().decode("cp949")
        # Convert the response to JSON
        json_data = json.loads(the_page)
        # Save the JSON data to a file
        with open("sooList.json", "w", encoding="utf-8") as f:
            json.dump(json_data["body"]["item"], f, ensure_ascii=False, indent=4, sort_keys=True)