import os

from .storage import Storage

class StorageLocal(Storage):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def getRootDirectory(self):
        if self.rootDirectory != "" and not os.path.exists(self.rootDirectory):
            os.makedirs(self.rootDirectory)

        root = os.path.join(self.rootDirectory, self.datetimeDirectory)
        if not os.path.exists(root):
            os.makedirs(root)

        return root

    def writeBytes(self, filename, obj, path_parts=[]):
        root = self.getRootDirectory()

        if len(path_parts) > 0:
            root = os.path.join(root, *path_parts)
            os.makedirs(root, exist_ok=True)

        path = os.path.join(root, filename)
        with open(path, mode="wb") as fd:
            fd.write(obj)
