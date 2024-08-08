from typing import Tuple, Optional
import os
import hopsworks
import pandas as pd
import wandb
from sktime.forecasting.model_selection import temporal_train_test_split
from dotenv import load_dotenv, dotenv_values
import logging

load_dotenv("../.env.default")
logging.basicConfig(level=logging.INFO)

def init_wandb_run(
    name: str,
    group: Optional[str] = None,
    job_type: Optional[str] = None,
    add_timestamp_to_name: bool = False,
    run_id: Optional[str] = None,
    resume: Optional[str] = None,
    reinit: bool = False,
    project: str = os.getenv("WANDB_PROJECT"),
    entity: str = os.getenv("WANDB_ENTITY"),
):

    if add_timestamp_to_name:
        name = f"{name}_{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        group=group,
        job_type=job_type,
        id=run_id,
        reinit=reinit,
        resume=resume,
    )

    return run

def load_dataset_from_feature_store(
    feature_view_version: int, training_dataset_version: int, fh: int = 24
):
    project = hopsworks.login(
        api_key_value=os.getenv("FS_API_KEY"), project=os.getenv("FS_PROJECT_NAME")
    )
    fs = project.get_feature_store()

    with init_wandb_run(
        name="load_training_data", job_type="load_feature_view", group="dataset"
    ) as run:
        feature_view = fs.get_feature_view(
            name="energy_consumption_denmark_view", version=feature_view_version
        )
        data, _ = feature_view.get_training_data(
            training_dataset_version=training_dataset_version
        )

        fv_metadata = feature_view.to_dict()
        fv_metadata["query"] = fv_metadata["query"].to_string()
        fv_metadata["features"] = [f.name for f in fv_metadata["features"]]
        fv_metadata["link"] = feature_view._feature_view_engine._get_feature_view_url(
            feature_view
        )
        fv_metadata["feature_view_version"] = feature_view_version
        fv_metadata["training_dataset_version"] = training_dataset_version

        raw_data_at = wandb.Artifact(
            name="energy_consumption_denmark_feature_view",
            type="feature_view",
            metadata=fv_metadata,
        )
        run.log_artifact(raw_data_at)

        run.finish()

    with init_wandb_run(
        name="train_test_split", job_type="prepare_dataset", group="dataset"
    ) as run:
        run.use_artifact("energy_consumption_denmark_feature_view:latest")
        
        data["datetime_utc"] = pd.PeriodIndex(data["datetime_utc"], freq="H")
        data = data.set_index(["area", "consumer_type", "datetime_utc"]).sort_index()

        X = data.drop(columns=["energy_consumption"])
        # Prepare the time series to be forecasted.
        y = data[["energy_consumption"]]

        y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, test_size=fh)
        #y_train, y_test, X_train, X_test = prepare_data(data, fh=fh)

        for split in ["train", "test"]:
            split_X = locals()[f"X_{split}"]
            split_y = locals()[f"y_{split}"]

            split_metadata = {
                "timespan": [
                    split_X.index.get_level_values(-1).min(),
                    split_X.index.get_level_values(-1).max(),
                ],
                "dataset_size": len(split_X),
                "num_areas": len(split_X.index.get_level_values(0).unique()),
                "num_consumer_types": len(split_X.index.get_level_values(1).unique()),
                "y_features": split_y.columns.tolist(),
                "X_features": split_X.columns.tolist(),
            }
            artifact = wandb.Artifact(
                name=f"split_{split}",
                type="split",
                metadata=split_metadata,
            )
            run.log_artifact(artifact)

        run.finish()

    return y_train, y_test, X_train, X_test