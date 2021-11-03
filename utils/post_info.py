

class Post:
    def __init__(self, f):
        self._f = f
        self._load()
        
    def _load(self):
        self._data = []
        self._index = {}
        with open(self._f) as fr:
            self._head = fr.readline().strip().split('\t')
            self._atrr_num = len(self._head)
            id_index = self._head.index('id')
            for line in fr:
                l = line.strip('\n').split('\t')
                if len(l) != self._atrr_num:
                    continue
                _id = l[id_index].strip()
                if _id == '':
                    continue 
                self._index[_id] = len(self._data)
                self._data.append([e.strip() for e in l])
    
    def __getitem__(self, item_attr):
        item, attr = item_attr
        row_index = self._index[item]
        col_index = self._head.index(attr)
        return self._data[row_index][col_index]
    
    def get_post_iter(self):
        return self._index.keys()
    
    def has(self, post_id):
        return post_id in self._index