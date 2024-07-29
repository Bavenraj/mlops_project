import datetime
import logging
from typing import Optional 
from pathlib import Path
import fire
import pandas as pd
from pandas.errors import EmptyDataError
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
    
    # _extract_records_from_file_url
    if cache_dir is None:
        cache_dir = "./output/data"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
    file_path = cache_dir / "ConsumptionDE35Hour.csv"
    if not file_path.exists():
        logging.info(f"Downloading data from: {url}")

        try:
            response = requests.get(url)
        except requests.exceptions.HTTPError as e:
            logging.error(
                f"Response status = {response.status_code}. Could not download the file due to: {e}"
            )

            return None
        
        if response.status_code != 200:
            raise ValueError(f"Response status = {response.status_code}. Could not download the file.")
    
        with file_path.open("w") as f:
            f.write(response.text)

        logging.info(f"Successfully downloaded data to: {file_path}")
    else:
        logging.info(f"Data already downloaded at: {file_path}")

    try:
        data = pd.read_csv(file_path, delimiter=";")
    except EmptyDataError:
        file_path.unlink(missing_ok=True)
        
        raise ValueError(f"Downloaded file at {file_path} is empty. Could not load it into a DataFrame.")

    records = data[(data["HourUTC"] >= export_start.strftime(datetime_format)) & (data["HourUTC"] < export_end.strftime(datetime_format))]

    # _extract_records_from_api_url
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
    
    records = response["records"]
    records = pd.DataFrame.from_records(records)


