
import subprocess
from .my_time import Time

class Cmd:
    def __init__(self, log=None):
        self._log_file = log
        if self._log_file is not None:
            self._log = open(log, 'w')
    
    def __del__(self):
        if self._log_file is not None:
            self._log.close()
    
    def do(self, cmd_str):
        status, output = subprocess.getstatusoutput(cmd_str)
        if self._log_file is not None:
            self._log.write(f"{output}\n")
            self._log.flush()
        return int(status)
        

class ShellParse:
    def __init__(self, shell_file):
        self._sh_f = shell_file
        self._replace_d = {
            'yesterday': Time.pre_day(1),
            'today': Time.today()
        }
        self._cmd_list = self._load(shell_file)
    
    def _load(self, f):
        l = []
        with open(f) as fr:
            for line in fr:
                line = line.strip()
                if line == '':
                    continue
                if line.startswith('#'):
                    continue
                line = self._replace(line)
                l.append(line)
        return l
    
    def _replace(self, line):
        if '{' not in line:
            return line
        return line.format(**self._replace_d)
    
    def cmd_list(self):
        return self._cmd_list
    
    