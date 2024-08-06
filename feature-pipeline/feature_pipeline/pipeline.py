import datetime
import logging
from typing import Optional 
from pathlib import Path
import fire
import pandas as pd
from pandas.errors import EmptyDataError
from yarl import URL
import requests
from great_expectations.core import ExpectationSuite, ExpectationConfigurationg
import hopsworks
from hsfs.feature_group import FeatureGroup
import os
from dotenv import load_dotenv, dotenv_values
import json
load_dotenv("../.env.default")
logging.basicConfig(level=logging.INFO)


def extraction(
    # the date is set to current date for first export
    last_export_datetime: Optional[datetime.datetime] = None,
    days_delay: int = 15,
    days_export: int = 30,
    url: str = "https://drive.google.com/uc?export=download&id=1y48YeDymLurOTUO-GeFOUXVNc9MCApG5",
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
        os.makedirs(cache_dir, exist_ok=True)
        
    file_path = f"{cache_dir}/ConsumptionDE35Hour.csv"
    if not os.path.exists(file_path):
        logging.info(f"Downloading data from: {url}")

        try:
            response = requests.get(url)
        except requests.exceptions.HTTPError as e:
            logging.error(
                f"Response status = {response.status_code}. Could not download the file due to: {e}"
            )
        
        if response.status_code != 200:
            raise ValueError(f"Response status = {response.status_code}. Could not download the file.")

        with open(file_path, "w") as f:
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

    return records, metadata

def transformation(records):
    logging.info("Transforming Data")
    df = pd.DataFrame(records)
    data = df.copy()
    data.drop(columns=["HourDK"], inplace=True)
    data.rename(columns={ "HourUTC": "datetime_utc","PriceArea": "area","ConsumerType_DE35": "consumer_type", "TotalCon": "energy_consumption" }, inplace=True)
    data["datetime_utc"] = pd.to_datetime(data["datetime_utc"])
    data["area"] = data["area"].astype("string")
    data["consumer_type"] = data["consumer_type"].astype("int32")
    data["energy_consumption"] = data["energy_consumption"].astype("float64")
    area_mappings = {"DK": 0, "DK1": 1, "DK2": 2}
    data["area"] = data["area"].map(area_mappings).astype("int8")
    logging.info("Successfully transformed Data")
    return data 

def validation():
    logging.info("Building validation expectation suite.")
    expectation_suite_energy_consumption = ExpectationSuite(
        expectation_suite_name="energy_consumption_suite"
    )

    #first expectation to see whether column is in correct list
    expectation_suite_energy_consumption.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_table_columns_to_match_ordered_list",
            kwargs={
                "column_list": [
                    "datetime_utc",
                    "area",
                    "consumer_type",
                    "energy_consumption",
                ]
            },
        )
    )

    #second expectation to see whether all 4 columns is available
    expectation_suite_energy_consumption.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_table_column_count_to_equal", kwargs={"value": 4}
        )
    )

    #third expectation to ensure date column is not null
    expectation_suite_energy_consumption.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "datetime_utc"},
        )
    )

    #fourth expectation to ensure that all three area values is available
    expectation_suite_energy_consumption.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_distinct_values_to_be_in_set",
            kwargs={"column": "area", "value_set": [0, 1, 2]},
        )
    )

    #fifth expectation to ensure that the area column is integer type
    expectation_suite_energy_consumption.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_of_type",
            kwargs={"column": "area", "type_": "int8"},
        )
    )

    #sixth expectation to ensure that all the area value to be in the dataset
    expectation_suite_energy_consumption.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_distinct_values_to_be_in_set",
            kwargs={
                "column": "consumer_type",
                "value_set": [
                    111,
                    112,
                    119,
                    121,
                    122,
                    123,
                    130,
                    211,
                    212,
                    215,
                    220,
                    310,
                    320,
                    330,
                    340,
                    350,
                    360,
                    370,
                    381,
                    382,
                    390,
                    410,
                    421,
                    422,
                    431,
                    432,
                    433,
                    441,
                    442,
                    443,
                    444,
                    445,
                    446,
                    447,
                    450,
                    461,
                    462,
                    999,
                ],
            },
        )
    )

    #seventh expectation to ensure that the consumer column is integer type
    expectation_suite_energy_consumption.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_of_type",
            kwargs={"column": "consumer_type", "type_": "int32"},
        )
    )

    #eighth expectation to ensure that the minimum consumption is 0
    expectation_suite_energy_consumption.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_min_to_be_between",
            kwargs={
                "column": "energy_consumption",
                "min_value": 0,
                "strict_min": False,
            },
        )
    )

    #ninth expectation to ensure that the consumption is in float type
    expectation_suite_energy_consumption.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_of_type",
            kwargs={"column": "energy_consumption", "type_": "float64"},
        )
    )

    #tenth expectation to ensure that the consumption is not null
    expectation_suite_energy_consumption.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "energy_consumption"},
        )
    )

    logging.info("Successfully built validation expectation suite.")

    return expectation_suite_energy_consumption

def loading(data, metadata, expectation_suite_energy_consumption, feature_group_version: int = 1):
    logging.info(f"Validating data and loading it to the feature store.")

    # Connect to feature store.
    project = hopsworks.login(
        api_key_value= os.getenv("FS_API_KEY"), project= os.getenv("FS_PROJECT_NAME")
    )
    feature_store = project.get_feature_store()

    # Create feature group.
    energy_feature_group = feature_store.get_or_create_feature_group(
    name="energy_consumption_denmark",
    version=feature_group_version,
    description="Denmark hourly energy consumption data. Data is uploaded with an 15 days delay.",
    primary_key=["area", "consumer_type"],
    event_time="datetime_utc",
    online_enabled=False,
    expectation_suite=expectation_suite_energy_consumption,
    )

    # Upload data.
    energy_feature_group.insert(
        features=data,
        overwrite=False,
        write_options={
            "wait_for_job": True,
        },
    )

    # Add feature descriptions.
    feature_descriptions = [
        {
            "name": "datetime_utc",
            "description": """
                            Datetime interval in UTC when the data was observed.
                            """,
            "validation_rules": "Always full hours, i.e. minutes are 00",
        },
        {
            "name": "area",
            "description": """
                            Denmark is divided in two price areas, divided by the Great Belt: DK1 and DK2.
                            If price area is “DK”, the data covers all Denmark.
                            """,
            "validation_rules": "0 (DK), 1 (DK1) or 2 (Dk2) (int)",
        },
        {
            "name": "consumer_type",
            "description": """
                            The consumer type is the Industry Code DE35 which is owned by Danish Energy. 
                            The code is used by Danish energy companies.
                            """,
            "validation_rules": ">0 (int)",
        },
        {
            "name": "energy_consumption",
            "description": "Total electricity consumption in kWh.",
            "validation_rules": ">=0 (float)",
        },
    ]

    for description in feature_descriptions:
        energy_feature_group.update_feature_description(
            description["name"], description["description"]
        )

    # Update statistics.
    energy_feature_group.statistics_config = {
        "enabled": True,
        "histograms": True,
        "correlations": True,
    }
    energy_feature_group.update_statistics_config()
    energy_feature_group.compute_statistics()
    metadata["feature_group_version"] = feature_group_version
    logging.info("Successfully validated data and loaded it to the feature store.")

    logging.info(f"Wrapping up the pipeline.")
    data_path = "./output/feature_pipeline_metadata.json"
    with open(data_path, "w") as f:
        json.dump(data, f)
    logging.info("Done!")


records, metadata = extraction()
data = transformation(records)
expectation_suite_energy_consumption = validation()
loading(data=data, metadata=metadata, expectation_suite_energy_consumption=expectation_suite_energy_consumption)
