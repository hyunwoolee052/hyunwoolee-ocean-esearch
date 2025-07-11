#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import json
import urllib.parse
import urllib.request
from datetime import date, datetime
import calendar


def get_service_key(service_key_file="servicekey.txt"):
    """
    Retrieve the service key from a file.

    Returns
    -------
    str
        The service key if available and not expired.
    None
        Exits the program if the service key is expired.
    """
    service_key_path = Path.cwd() / service_key_file
    with open(service_key_path, "r", encoding="utf-8") as f:
        expiration_date = f.readline().strip()
        service_key = f.readline().strip()
    if datetime.strptime(expiration_date, "%Y-%m-%d") >= datetime.now():
        return service_key
    else:
        print("Service key has expired. Please update the servicekey file.")
        sys.exit(1)


def fetch_data(id, key, sdate, edate, output_dir="data"):
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
    output_dir : str, optional
        Directory where the output file will be saved (default is "data").

    Returns
    -------
    bool
        True if data was fetched and saved, False otherwise.
    """
    data = {
        "id": id,
        "key": key,
        "sdate": sdate,
        "edate": edate,
    }

    url = "https://www.nifs.go.kr/bweb/OpenAPI_json"
    full_url = f"{url}?{urllib.parse.urlencode(data)}"
    try:
        with urllib.request.urlopen(full_url) as response:
            the_page = response.read().decode("cp949")
            json_data = json.loads(the_page)
            items = json_data.get("body", {}).get("item", [])
            if not items:
                print(f"No data found for {sdate} to {edate}.")
                return False
            # Ensure output directory exists
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            # Save the JSON data to a file in the output directory
            output_path = Path(output_dir) / f"{id}_{sdate[:6]}.json"
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(items, file, ensure_ascii=False, indent=4)
            print(f"Saved data to {output_path}")
            return True
    except Exception as e:
        print(f"Error fetching data for {sdate} to {edate}: {e}")
        return False


def fetch_range(years, months, id, key, output_dir="data"):
    """
    Fetch data for a range of years and months.

    Parameters
    ----------
    years : iterable
        Years to fetch.
    months : iterable
        Months to fetch.
    id : str
        Identifier for the type of data to fetch.
    key : str
        API key for the NIFS OpenAPI.
    output_dir : str, optional
        Directory where the output file will be saved (default is "data").
    """
    for year in years:
        for month in months:
            sdate = date(year, month, 1).strftime("%Y%m%d")
            eday = calendar.monthrange(year, month)[1]
            edate = date(year, month, eday).strftime("%Y%m%d")
            print(f"Fetching data for {sdate} to {edate}")
            fetch_data(id, key, sdate, edate, output_dir=output_dir)


def main():
    key = get_service_key()
    years = range(1968, 2025)
    months = range(1, 13)
    fetch_range(years, months, id="sooList", key=key, output_dir="data")


if __name__ == "__main__":
    main()
