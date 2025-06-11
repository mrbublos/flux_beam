import datetime

class Logger:
    def __init__(self, name: str):
        self.name = name

    def info(self, message: str):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"INFO [{timestamp}] [{self.name}] {message}")

    def debug(self, message: str):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"DEBUG [{timestamp}] [{self.name}] {message}")

    def error(self, message: str):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"DEBUG [{timestamp}] [{self.name}] {message}")