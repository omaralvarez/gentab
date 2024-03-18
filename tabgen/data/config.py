from tabgen.utils import console

import json


class Config:
    def __init__(self, path) -> None:
        with open(path, "r") as jsonfile:
            self._config = json.load(jsonfile)

        console.print("✅ Config {} loaded...".format(path))

    def __getitem__(self, key):
        return self._config[key]

    def __setitem__(self, key, data):
        self._config[key] = data

    def exists(self, key: str) -> bool:
        return True if key in self._config else False
