
from .my_time import Time

#format print
class Fprint:
    def __init__(self):
        pass

    @staticmethod
    def pt(s):
        """print with time"""
        print(f"[{Time.cur_time()}] {s}")