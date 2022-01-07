import datetime

class Storage:
    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("root", "")
        kwargs.setdefault("datetime", datetime.datetime.utcnow())

        self.rootDirectory = kwargs["root"]
        self.datetimeDirectory = kwargs["datetime"].strftime("%Y%m%dT%H%M%SZ")
