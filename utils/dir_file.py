
import os
import shutil


#dir file operate
class DirFile:
    def __init__(self):
        pass
    
    @staticmethod
    def get_file_under_dir(dir_path):
        return [f"{dir_path}/{e}" for e in os.listdir(dir_path) 
                if os.path.isfile(f"{dir_path}/{e}")]
    
    @staticmethod
    def get_dir_under_dir(dir_path):
        return [f"{dir_path}/{e}" for e in os.listdir(dir_path) 
                if os.path.isdir(f"{dir_path}/{e}")]
    
    @staticmethod
    def remove_file(file_path):
        os.remove(file_path)
    
    @staticmethod
    def remove_dir(dir_path):
        if not os.path.isdir(dir_path):
            return
        shutil.rmtree(dir_path)
        
    @staticmethod
    def merge_files(file_list, merged_file):
        with open(merged_file, 'w') as fw:
            for f in file_list:
                with open(f) as fr:
                    fw.write(fr.read())
    
    @staticmethod
    def mkdir(path):
        if not os.path.isdir(path):
            os.mkdir(path)
            
    @staticmethod
    def file_bytes(path):
        return os.path.getsize(path)
    
    @staticmethod
    def file_ctime(path):
        return int(os.path.getctime(path))
    
    @staticmethod
    def file_mtime(path):
        return int(os.path.getmtime(path))