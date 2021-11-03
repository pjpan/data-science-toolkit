
import multiprocessing
from blessings import Terminal

class MultiProcess:
    def __init__(self, worker_num, need_print=False):
        self._worker_num = worker_num
        self._worker_list = []
        self._need_print = need_print
        if self._need_print:
            self._p_v_list = [multiprocessing.Array('u', 100) for i in range(self._worker_num)]
    
    @staticmethod
    def data_split(split_num, origin_file_list, hash_col):
        if not origin_file_list:
            return
        #mkdir in first origin file dir
        dir_path = f"{os.path.dirname(origin_file_list[0])}/tmp_split_data_dir"
        DirFile.remove_dir(dir_path)
        DirFile.mkdir(dir_path)
        out_path_list = [f"{dir_path}/part_{i}" for i in range(split_num)]
        out_file_list = [open(f, 'w') for f in out_path_list]
        for f in origin_file_list:
            with open(f) as fr:
                for line in fr:
                    l = line.strip().split('\t')
                    hash_code = hash(l[hash_col])
                    index = hash_code % split_num
                    out_file_list[index].write(line)
        for f in out_file_list:
            f.close()
        return out_path_list
    
    @staticmethod
    def data_clear(path_list, del_dir=False):
        if not path_list:
            return
        dir_path = os.path.dirname(path_list[0])
        for f in path_list:
            DirFile.remove_file(f)
        if del_dir:
            DirFile.remove_dir(dir_path)
            
    @staticmethod
    def list_split(split_num, origin_list):
        step = int(len(origin_list) / split_num)
        split_list = []
        for i in range(split_num):
            s_list = origin_list[i*step: (i+1)*step]
            if i == split_num - 1:
                s_list = origin_list[i*step: ]
            split_list.append(s_list)
        return split_list
    
    @staticmethod
    def dict_split(split_num, origin_dict):
        d_list = []
        key_list = list(origin_dict.keys())
        split_key_list = list_split(split_num, key_list)
        for i in range(split_num):
            d = {k:origin_dict[k] for k in split_key_list[i]}
            d_list.append(d)
        return d_list
    
    def update_p_v(self, i, s):
        if self._need_print:
            self._p_v_list[i][:] = list(s) + ['.' for i in range(100 - len(s))]
    
    
    def worker(self, func, args_list=None):
        if args_list is None:
            for i in range(self._worker_num):
                self._worker_list.append(
                    multiprocessing.Process(target=func, args=(self, i)))
        else:  
            for i in range(self._worker_num):
                self._worker_list.append(
                    multiprocessing.Process(target=func, args=(self, i, *args_list[i])))
    
    def _print(self):
        terminal = Terminal()
        Fprint.pt(f"worker start, worker number: {self._worker_num}")
        print('\n'.join([f"thread_{i}: " for i in range(self._worker_num)]))
        while True:
            with terminal.location(0, terminal.height - self._worker_num - 1):
                print('\n'.join([f"thread_{i}: {''.join(list(e))}" 
                                for i, e in enumerate(self._p_v_list)]),
                      flush=True)
            tmp_l = [''.join(list(e)).strip('.') == 'done' for e in self._p_v_list]
            if all(tmp_l):
                break
            time.sleep(3)
        Fprint.pt('all worker done')
    
    def start(self):
        for worker in self._worker_list:
            worker.start()
        if self._need_print:
            self._p_print = multiprocessing.Process(target=self._print, args=())
            self._p_print.start()
    
    def join(self):
        for worker in self._worker_list:
            worker.join()
        if self._need_print:
            self._p_print.join()