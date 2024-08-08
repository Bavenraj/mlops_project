from functools import partial
from typing import Optional
import fire
import numpy as np
import pandas as pd
import wandb
import json
import hopsworks
import os
from dotenv import load_dotenv, dotenv_values
import logging
import lightgbm as lgb
from sktime.forecasting.compose import make_reduction, ForecastingPipeline
from sktime.forecasting.naive import NaiveForecaster
from sktime.transformations.series.date import DateTimeFeatures
from sktime.transformations.series.summarize import WindowSummarizer
load_dotenv("../.env.default")
logging.basicConfig(level=logging.INFO)
from matplotlib import pyplot as plt
from sktime.forecasting.model_evaluation import evaluate as cv_evaluate
from sktime.forecasting.model_selection import ExpandingWindowSplitter, temporal_train_test_split
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
from sktime.utils.plotting import plot_windows
from training_pipeline import transformers

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

def run(
    fh: int = 24,
    feature_view_version: Optional[int] = None,
    training_dataset_version: Optional[int] = None,
):
    
    data_path = "./output/feature_view_metadata.json"
    with open(data_path, "r") as f:
        feature_view_metadata = json.load(f)
        
    if feature_view_version is None:
        feature_view_version = feature_view_metadata["feature_view_version"]
    if training_dataset_version is None:
        training_dataset_version = feature_view_metadata["training_dataset_version"]

    y_train, _, X_train, _ = load_dataset_from_feature_store(
        feature_view_version=feature_view_version,
        training_dataset_version=training_dataset_version,
        fh=fh,
    )
    sweep_configs = {
    "method": "grid",
    "metric": {"name": "validation.MAPE", "goal": "minimize"},
    "parameters": {
        "forecaster__estimator__n_jobs": {"values": [-1]},
        "forecaster__estimator__n_estimators": {"values": [1000, 2000, 2500]},
        "forecaster__estimator__learning_rate": {"values": [0.1, 0.15]},
        "forecaster__estimator__max_depth": {"values": [-1, 5]},
        "forecaster__estimator__reg_lambda": {"values": [0, 0.01, 0.015]},
        "daily_season__manual_selection": {"values": [["day_of_week", "hour_of_day"]]},
        "forecaster_transformers__window_summarizer__lag_feature__lag": {
            "values": [list(range(1, 73))]
        },
        "forecaster_transformers__window_summarizer__lag_feature__mean": {
            "values": [[[1, 24], [1, 48], [1, 72]]]
        },
        "forecaster_transformers__window_summarizer__lag_feature__std": {
            "values": [[[1, 24], [1, 48]]]
        },
        "forecaster_transformers__window_summarizer__n_jobs": {"values": [1]},
    },
}
    sweep_id = wandb.sweep(
        sweep= sweep_configs, project=os.getenv("WANDB_PROJECT")
    )

    wandb.agent(
        project=os.getenv("WANDB_PROJECT"),
        sweep_id=sweep_id,
        function=partial(run_sweep, y_train=y_train, X_train=X_train, fh=fh),
    )

    metadata = {"sweep_id": sweep_id}
    data_path = "./output/last_sweep_metadata.json"
    with open(data_path, "w") as f:
        json.dump(metadata, f)
    return metadata


def run_sweep(y_train: pd.DataFrame, X_train: pd.DataFrame, fh: int):

    with init_wandb_run(
        name="experiment", job_type="hpo", group="train", add_timestamp_to_name=True
    ) as run:
        run.use_artifact("split_train:latest")

        config = wandb.config
        config = dict(config)
        lag = config.pop(
        "forecaster_transformers__window_summarizer__lag_feature__lag",
        list(range(1, 72 + 1)),
        )
        mean = config.pop(
            "forecaster_transformers__window_summarizer__lag_feature__mean",
            [[1, 24], [1, 48], [1, 72]],
        )
        std = config.pop(
            "forecaster_transformers__window_summarizer__lag_feature__std",
            [[1, 24], [1, 48], [1, 72]],
        )
        n_jobs = config.pop("forecaster_transformers__window_summarizer__n_jobs", 1)
        window_summarizer = WindowSummarizer(
            **{"lag_feature": {"lag": lag, "mean": mean, "std": std}},
            n_jobs=n_jobs,
        )

        regressor = lgb.LGBMRegressor()
        forecaster = make_reduction(
            regressor,
            transformers=[window_summarizer],
            strategy="recursive",
            pooling="global",
            window_length=None,
        )

        pipe = ForecastingPipeline(
            steps=[
                ("attach_area_and_consumer_type", transformers.AttachAreaConsumerType()),
                (
                    "daily_season",
                    DateTimeFeatures(
                        manual_selection=["day_of_week", "hour_of_day"],
                        keep_original_columns=True,
                    ),
                ),
                ("forecaster", forecaster),
            ]
        )
        model = pipe.set_params(**config)
        data_length = len(y_train.index.get_level_values(-1).unique())
        assert data_length >= fh * 10, "Not enough data to perform a 3 fold CV."

        cv_step_length = data_length // 3 # k =3
        initial_window = max(fh * 3, cv_step_length - fh)
        cv = ExpandingWindowSplitter(
            step_length=cv_step_length, fh=np.arange(fh) + 1, initial_window=initial_window
        )
        random_time_series = (
            y_train.groupby(level=[0, 1])
            .get_group((1, 111))
            .reset_index(level=[0, 1], drop=True)
        )
        plot_windows(cv, random_time_series)

        save_path = "./output/cv_scheme.png"
        plt.savefig(save_path)
        wandb.log({"cv_scheme": wandb.Image(save_path)})

        results = cv_evaluate(
            forecaster=model,
            y=y_train,
            X=X_train,
            cv=cv,
            strategy="refit",
            scoring=MeanAbsolutePercentageError(symmetric=False),
            error_score="raise",
            return_data=False,
        )

        results = results.rename(
            columns={
                "test_MeanAbsolutePercentageError": "MAPE",
                "fit_time": "fit_time",
                "pred_time": "prediction_time",
            }
        )
        mean_results = results[["MAPE", "fit_time", "prediction_time"]].mean(axis=0)
        mean_results = mean_results.to_dict()
        results = {"validation": mean_results}

        logging.info(f"Validation MAPE: {results['validation']['MAPE']:.2f}")
        logging.info(f"Mean fit time: {results['validation']['fit_time']:.2f} s")
        logging.info(f"Mean predict time: {results['validation']['prediction_time']:.2f} s")

        wandb.log(results)

        metadata = {
            "experiment": {"name": run.name, "fh": fh},
            "results": results,
            "config": config,
        }
        artifact = wandb.Artifact(
            name=f"config",
            type="model",
            metadata=metadata,
        )
        run.log_artifact(artifact)

        run.finish()

