import json


class Config:
    url: str = None

    def __init__(self, test: bool = True):
        self.test = test
        with open("config.json") as f:
            self.data = json.load(f)
        if test == True:
            self.url = self.data["api-info"]["cert"]
            self.account_number = self.data["account-numbers"]["cert"]
            self.username = self.data["login"]["cert"]
            self.password = self.data["password"]["cert"]
        else:
            self.url = self.data["api-info"]["prod"]
            self.account_number = self.data["account-numbers"]["prod"]
            self.username = self.data["login"]["prod"]
            self.password = self.data["password"]["prod"]