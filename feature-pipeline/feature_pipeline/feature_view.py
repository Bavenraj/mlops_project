from datetime import datetime
from typing import Optional
import hopsworks
import hsfs
import logging
import json
import os
from dotenv import load_dotenv, dotenv_values

load_dotenv("../.env.default")

logging.basicConfig(level=logging.INFO)

def create(
    feature_group_version: Optional[int] = None,
    start_datetime: Optional[datetime] = None,
    end_datetime: Optional[datetime] = None,
) -> dict:

    if feature_group_version is None:
        data_path = "./output/feature_pipeline_metadata.json"
        if not data_path.exists():
            raise FileNotFoundError(f"Cached JSON from {data_path} does not exist.")

        with open(data_path, "r") as f:
            feature_pipeline_metadata = json.load(f)
        feature_group_version = feature_pipeline_metadata["feature_group_version"]

    if start_datetime is None or end_datetime is None:
        data_path = "./output/feature_pipeline_metadata.json"
        if not data_path.exists():
            raise FileNotFoundError(f"Cached JSON from {data_path} does not exist.")

        with open(data_path, "r") as f:
            feature_pipeline_metadata = json.load(f)

        start_datetime = datetime.strptime(
            feature_pipeline_metadata["export_datetime_utc_start"],
            feature_pipeline_metadata["datetime_format"],
        )
        end_datetime = datetime.strptime(
            feature_pipeline_metadata["export_datetime_utc_end"],
            feature_pipeline_metadata["datetime_format"],
        )

    project = hopsworks.login(
        api_key_value=os.getenv("FS_API_KEY"),
        project=os.getenv("FS_PROJECT_NAME"),
    )
    fs = project.get_feature_store()

    try:
        feature_views = fs.get_feature_views(name="energy_consumption_denmark_view")
        
    except hsfs.client.exceptions.RestAPIError:
        logging.info("No feature views found for energy_consumption_denmark_view.")

        feature_views = []

    for feature_view in feature_views:
        try:
            feature_view.delete_all_training_datasets()
        except hsfs.client.exceptions.RestAPIError:
            logger.error(
                f"Failed to delete training datasets for feature view {feature_view.name} with version {feature_view.version}."
            )

        try:
            feature_view.delete()
        except hsfs.client.exceptions.RestAPIError:
            logger.error(
                f"Failed to delete feature view {feature_view.name} with version {feature_view.version}."
            )

    # Create feature view in the given feature group version.
    energy_consumption_fg = fs.get_feature_group(
        "energy_consumption_denmark", version=feature_group_version
    )
    ds_query = energy_consumption_fg.select_all()
    feature_view = fs.create_feature_view(
        name="energy_consumption_denmark_view",
        description="Energy consumption for Denmark forecasting model.",
        query=ds_query,
        labels=[],
    )

    # Create training dataset.
    logger.info(
        f"Creating training dataset between {start_datetime} and {end_datetime}."
    )
    feature_view.create_training_data(
        description="Energy consumption training dataset",
        data_format="csv",
        start_time=start_datetime,
        end_time=end_datetime,
        write_options={"wait_for_job": True},
        coalesce=False,
    )

    # Save metadata.
    metadata = {
        "feature_view_version": feature_view.version,
        "training_dataset_version": 1,
    }
    utils.save_json(
        metadata,
        file_name="feature_view_metadata.json",
    )

    return metadata

