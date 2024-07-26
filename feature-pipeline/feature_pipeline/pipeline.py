import datetime
import logging
from typing import Optional 
from pathlib import Path
import fire
import pandas as pd
from yarl import URL
import requests
logging.basicConfig(level=logging.INFO)

def run (
    # the date is set to current date for first export
    last_export_datetime: Optional[datetime.datetime] = None,
    days_delay: int = 15,
    days_export: int = 30,
    url: str = "https://drive.google.com/uc?export=download&id=1y48YeDymLurOTUO-GeFOUXVNc9MCApG5",
    feature_group_version: int = 1,
    datetime_format: str = "%Y-%m-%d %H:%M",
    cache_dir: Optional[Path] = None,
):
    # extraction
    logging.info(f"Extracting data from API")
    if last_export_datetime is None:
        last_export_datetime = datetime.datetime( 2023, 6, 30, 21, 0, 0) + datetime.timedelta(days=days_delay)
        last_export_datetime = last_export_datetime.replace(minute=0,second=0,microsecond=0)
    else:
        last_export_datetime = last_export_datetime.replace( minute=0, second=0, microsecond=0)
    expiring_dataset_datetime = datetime.datetime(2023, 6, 30, 21, 0, 0) + datetime.timedelta(days=days_delay)
    if last_export_datetime > expiring_dataset_datetime:
        last_export_datetime = expiring_dataset_datetime
        logging.warning("We clapped the last export datetime + datetime.timedelta(days=days_delay)")
        export_end = last_export_datetime - datetime.timedelta(days=days_delay)
        export_start = last_export_datetime - datetime.timedelta(days=days_delay + days_export)
        min_export_start = datetime.datetime(2020, 6, 30, 22, 0, 0)
        if export_start < min_export_start:
            export_start=min_export_start
            export_end=export_start+datetime.timedelta(days=days_export)
            logging.warning("We clapped 'export_start' to 'datetime(2020, 6, 30, 22, 0, 0)' and 'export_end' to 'export_start + datetime.timedelta(days=days_export)' as this is the latest window available in the dataset.")
    query_params = {
        "offset": 0,
        "sort": "HourUTC",
        "timezone": "utc",
        "start": export_start.strftime("%Y-%m-%dT%H:%M"),
        "end": export_end.strftime("%Y-%m-%dT%H:%M"),
    }
    url = URL(url) % query_params
    url = str(url)
    logging.info(f"Requesting data from API with : {url}")
    response = requests.get(url)
    logging.info("The response received with a status code of : {response.status_code}")
    try:
        response = response.json()
    except:
        logging.error(f"Response status = {response.status_code}. Could not decode response from API")

