import configparser
from pathlib import Path

config = configparser.ConfigParser()
config.read("default.ini")
if Path("live.ini").exists():
    config.read("live.ini")
