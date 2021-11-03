
import os
import time
from enum import Enum


class WCode(Enum):
    exit = 1
    success = 2

class Wait:
    def __init__(self):
        pass
    
    @staticmethod
    def lock(lock_path, wait_time):
        end_time = time.time() + wait_time
        while os.path.isfile(lock_path):
            time.sleep(10)
            if time.time() > end_time:
                return WCode.exit
        return WCode.success