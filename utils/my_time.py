import time

class Time:
    def __init__(self):
        pass
    
    @staticmethod
    def date_to_timestamp(date):
        time_arr = time.strptime(date, "%Y-%m-%d %H:%M:%S")
        timestamp = int(time.mktime(time_arr))
        return timestamp
    
    @staticmethod
    def cur_time():
        """return cur date time string"""
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    
    @staticmethod
    def pre_timestamp(second):
        return int(time.time()) - int(second)
    
    @staticmethod
    def pre_day(days):
        return time.strftime('%Y%m%d', time.localtime(time.time() - days * 24 *3600))
    
    @staticmethod
    def today():
        return time.strftime('%Y%m%d', time.localtime(time.time()))
