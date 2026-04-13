import pandas as pd
import logging
from typing import List
import pickle
from botocore.client import BaseClient
import io
import os
import pyarrow.fs as pafs
import boto3
from typing import cast, IO
from io import BytesIO
import matplotlib.pyplot as plt
from ml_and_backtester_app.data.data_manager import DataManager

logger = logging.getLogger(__name__)

class s3Utils:
    @staticmethod
    def get_pyarrow_s3_filesystem():
        return pafs.S3FileSystem(
            access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            session_token=os.getenv("AWS_SESSION_TOKEN"),
            region=os.getenv("AWS_DEFAULT_REGION"),
        )

    @staticmethod
    def push_object_to_s3_parquet(object_to_push:pd.DataFrame,
                                  path:str)->None:
        """
        Method to upload a dataframe to s3 in parquet format at the specified path
        :param object_to_push:
        :param path:
        :return: None
        """
        if not isinstance(object_to_push, pd.DataFrame):
            logger.error("object_to_push must be a pd.DataFrame")
            raise ValueError("object_to_push must be a pd.DataFrame")
        if not isinstance(path, str):
            logger.error("path must be a str")
            raise ValueError("path must be a str")

        fs = s3Utils.get_pyarrow_s3_filesystem()

        object_to_push.to_parquet(
            path=path,
            engine="pyarrow",
            filesystem=fs,
        )
        return

    @staticmethod
    def push_objects_to_s3_parquet(objects_dct:dict)->None:
        """
        Method to upload several dfs to s3 in parquet format at the given paths
        :param objects_dct: objects as values and paths as keys
        :return: None
        """
        if not isinstance(objects_dct, dict):
            logger.error("objects_dct must be a dict.")
            raise ValueError("objects_dct must be a dict.")
        for path, obj in objects_dct.items():
            s3Utils.push_object_to_s3_parquet(object_to_push=obj,
                                              path=path)
        return

    @staticmethod
    def pull_parquet_file_from_s3(path:str)->pd.DataFrame:
        if not isinstance(path, str):
            logger.error("path must be a str.")
            raise ValueError("path must be a str.")
        fs = s3Utils.get_pyarrow_s3_filesystem()
        res = pd.read_parquet(path=path,
                              engine="pyarrow",
                              filesystem=fs
                              )
        return res

    @staticmethod
    def pull_parquet_files_from_s3(paths:List[str])->dict:
        """
        Given a list of s3 files paths returns dfs in a dict
        :param paths: list of s3 paths
        :return: dict of dfs with keys as names of the s3 object and the dfs as values
        """
        if not isinstance(paths,list):
            logger.error("paths must be a list.")
            raise ValueError("paths must be a list.")
        for path in paths:
            if not isinstance(path, str):
                logger.error("all paths must be strings.")
                raise ValueError("all paths must be strings.")

        res_dct = {}
        for path in paths:
            name = path.split("/")
            name = name[-1]
            name = name.split(".")
            name = name[0]
            res = s3Utils.pull_parquet_file_from_s3(path=path)
            res_dct[name] = res

        return res_dct

    @staticmethod
    def replace_existing_files_in_s3(s3: BaseClient,
                                     bucket_name: str,
                                     files_dct: dict
                                     ) -> None:

        """
        Given a dict with the file names (as the ones in s3) as keys and the file content as values,
        replace the existing files in s3 with the new content.
        :return:
        """
        # Check data types of inputs
        # if not hasattr(s3, 'put_object'):
        #     logger.error("s3 must be a boto3 BaseClient instance.")
        #     raise ValueError("s3 must be a boto3 BaseClient instance.")
        if not isinstance(bucket_name, str):
            logger.error("bucket_name must be a string.")
            raise ValueError("bucket_name must be a string.")
        if not isinstance(files_dct, dict):
            logger.error("files_dct must be a dictionary.")
            raise ValueError("files_dct must be a dictionary with file names as keys and file content as values.")

        # Check that the file names (keys of the dict) are all present on s3
        for file_name in files_dct.keys():
            try:
                s3.head_object(Bucket=bucket_name, Key=file_name)
            except Exception as e:
                logger.error(f"File {file_name} does not exist in S3 bucket {bucket_name}: {e}")
                raise ValueError(f"File {file_name} does not exist in S3 bucket {bucket_name}.")

        # Upload new versions first
        for file_name, content in files_dct.items():
            ext = file_name.split('.')[-1]

            if ext == "parquet":
                buffer = io.BytesIO()
                content.to_parquet(buffer, index=True)
                buffer.seek(0)
                body = buffer

            elif ext == "pkl":
                body = pickle.dumps(content)

            else:
                raise ValueError(f"Unsupported extension: {file_name}")

            s3.put_object(
                Bucket=bucket_name,
                Key=file_name,
                Body=body
            )
            logger.info(f"Uploaded new version of {file_name} to S3 bucket {bucket_name}.")

        # Delete all non-latest versions + delete markers
        for file_name in files_dct.keys():
            paginator = s3.get_paginator("list_object_versions")

            to_delete = []

            for page in paginator.paginate(Bucket=bucket_name, Prefix=file_name):

                for v in page.get("Versions", []):
                    if not v["IsLatest"]:
                        to_delete.append({
                            "Key": file_name,
                            "VersionId": v["VersionId"]
                        })

                for d in page.get("DeleteMarkers", []):
                    if not d["IsLatest"]:
                        to_delete.append({
                            "Key": file_name,
                            "VersionId": d["VersionId"]
                        })

            if to_delete:
                s3.delete_objects(
                    Bucket=bucket_name,
                    Delete={"Objects": to_delete}
                )
            logger.info(f"Deleted previous versions of {file_name} in S3 bucket {bucket_name}.")

    @staticmethod
    def upload_df_with_index(df: pd.DataFrame, bucket:str, path:str)->None:
        """
        Upload a DataFrame to S3 in parquet format, including the index.
        :param df: DataFrame to upload.
        :param bucket: S3 bucket name.
        :param path: S3 path (including filename) where the DataFrame will be stored.
        :return: None
        """
        # Create an in-memory buffer
        buffer = BytesIO()

        # Write the DataFrame to the buffer in parquet format, including the index
        df.to_parquet(cast(IO[bytes], buffer), engine="pyarrow", index=True)

        # Reset the buffer's position to the beginning
        buffer.seek(0)

        # Initialize S3 client and upload the buffer content to S3
        s3_client = boto3.client('s3')
        s3_client.upload_fileobj(buffer, bucket, path)

        return


    @staticmethod
    def save_plot_to_s3(dm: DataManager, path_name: str, fig: plt.Figure) -> None:
        """
        Save a matplotlib figure to S3.

        Parameters
        ----------
        dm : DataManager instance with AWS client
        path_name : str
            S3 key (not full path)
        fig : plt.Figure
            Matplotlib figure to save
        """
        buffer = BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)

        dm.aws.s3.upload(src=buffer.getvalue(), key=path_name)