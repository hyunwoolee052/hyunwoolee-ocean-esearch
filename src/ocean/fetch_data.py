#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import json
import urllib.parse
import urllib.request
from datetime import datetime


def get_service_key():
    """
    Retrieve the service key from a file.

    Returns
    -------
    str
        The service key if available and not expired.
    None
        Exits the program if the service key is expired.
    """
    service_key_file = Path.cwd() / "servicekey"
    with open(service_key_file, "r", encoding="utf-8") as f:
        expiration_date = f.readline().strip()
        service_key = f.readline().strip()
    if datetime.strptime(expiration_date, "%Y-%m-%d") >= datetime.now():
        return service_key
    else:
        print("Service key has expired. Please update the servicekey file.")
        return sys.exit(1)


def fetch_data(id, key, sdate, edate):
    """
    Fetch data from the NIFS OpenAPI and save it to a file.

    Parameters
    ----------
    id : str
        Identifier for the type of data to fetch.
    key : str
        API key for the NIFS OpenAPI.
    sdate : str
        Start date in 'YYYYMMDD' format.
    edate : str
        End date in 'YYYYMMDD' format.

    Returns
    -------
    None
        The function saves the fetched data to a file.
    """
    data = {
        "id": "sooList",  # or "sooCode" for station information
        "key": key,
        "sdate": sdate,
        "edate": edate,
    }

    url_values = urllib.parse.urlencode(data)
    url = "https://www.nifs.go.kr/bweb/OpenAPI_json"
    full_url = f"{url}?{url_values}"
    with urllib.request.urlopen(full_url) as response:
        the_page = response.read().decode("cp949")
        # Convert the response to JSON
        json_data = json.loads(the_page)

        # Save the JSON data to a file
        with open(f"sooList_{sdate[:6]}.json", "w", encoding="utf-8") as file:
            json.dump(
                json_data["body"]["item"],  # Save the 'item' part of the response
                file,
                ensure_ascii=False,
                indent=4,
            )
