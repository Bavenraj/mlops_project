import hopsworks
import os
from dotenv import load_dotenv, dotenv_values

load_dotenv("../.env.default")

def clean():
    project = hopsworks.login(
        api_key_value=os.getenv("FS_API_KEY"),
        project=os.getenv("FS_PROJECT_NAME"),
    )
    fs = project.get_feature_store()

    print("Deleting feature views and training datasets...")
    try:
        feature_views = fs.get_feature_views(name="energy_consumption_denmark_view")
        for feature_view in feature_views:
            try:
                feature_view.delete()
            except Exception as e:
                print(e)
    except Exception as e:
        print(e)

    print("Deleting feature groups...")
    try:
        feature_groups = fs.get_feature_groups(name="energy_consumption_denmark")
        for feature_group in feature_groups:
            try:
                feature_group.delete()
            except Exception as e:
                print(e)
    except Exception as e:
        print(e)