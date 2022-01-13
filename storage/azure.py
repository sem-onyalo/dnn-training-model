from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient
import config

from .storage import Storage

class StorageAzure(Storage):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        credential = DefaultAzureCredential()
        account_url = f"https://{config.cloud_storage_account}.blob.core.windows.net"
        self.client = ContainerClient(account_url, config.cloud_storage_container, credential=credential)

    def writeBytes(self, filename, obj, path_parts=[]):
        parts = [p for p in path_parts]
        parts.insert(0, self.datetimeDirectory)
        parts.append(filename)
        path = "/".join(parts)
        self.client.upload_blob(path, obj)
