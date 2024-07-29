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

    metadata = {
        "days_delay": days_delay,
        "days_export": days_export,
        "url": url,
        "export_datetime_utc_start": export_start.strftime(datetime_format),
        "export_datetime_utc_end": export_end.strftime(datetime_format),
        "datetime_format": datetime_format,
        "num_unique_samples_per_time_series": len(records["HourUTC"].unique()),
    }

    if metadata["num_unique_samples_per_time_series"] < days_export * 24:
        raise RuntimeError(
            f"Could not extract the expected number of samples from the api: {metadata['num_unique_samples_per_time_series']} < {days_export * 24}. \
            Check out the API at: https://www.energidataservice.dk/tso-electricity/ConsumptionDE35Hour ")
    
    logging.info("Successfully extracted data from File.")

    logging.info("Transforming Data")
    df = pd.DataFrame(records)
    data = df.copy()
    data.drop(columns=["HourDK"], inplace=True)
    data.rename(columns={ "HourUTC": "datetime_utc","PriceArea": "area","ConsumerType_DE35": "consumer_type", "TotalCon": "energy_consumption" }, inplace=True)
    data["datetime_utc"] = pd.to_datetime(data["datetime_utc"])
    data["area"] = data["area"].astype("string")
    data["consumer_type"] = data["consumer_type"].astype("int32")
    data["energy_consumption"] = data["energy_consumption"].astype("float64")

def extraction():
    """Will add here soon"""
