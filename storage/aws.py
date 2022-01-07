import boto3
import config

from .storage import Storage

class StorageAwsS3(Storage):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.bucket = boto3.resource("s3").Bucket(config.cloud_blob_root)

    def writeBytes(self, filename, obj, path_parts=[]):
        parts = [p for p in path_parts]
        parts.insert(0, self.datetimeDirectory)
        parts.append(filename)
        path = "/".join(parts)
        self.bucket.put_object(Key=path, Body=obj)
