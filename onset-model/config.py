##############################################
# Original code by Bjöorn Lindqvist
# https://github.com/bjourne/onset-replication
##############################################

from os import name
from os.path import expanduser
from re import match
from socket import gethostname
import cnn_multiclass, cnn_multilabel_1, cnn_multilabel_2
from evaluation_multilabel_1 import evaluate as evaluate_multilabel1
from evaluation_multilabel_2 import evaluate as evaluate_multilabel2


CONFIGS = [
    (('nt', r'^XXXXX$'),
     {
         'data-dir' : r'XXXXX\data-pipeline\dataset',
         'cache-dir' : r'XXXXX\onset-model\tmp\cache',
         'model-dir' : r'XXXXX\onset-model\tmp\models',
         'multiclass' : {
             'module' : cnn_multiclass,
             'seed' : 3553105877,
             'digest' : 'XXXXX'
             
         },
         'multilabel1' : {
             'module' : cnn_multilabel_1,
             'seed' : 3553105877,
             'digest' : 'XXXXX',
             'eval' : evaluate_multilabel1 
             
         },
         'multilabel2' : {
             'module' : cnn_multilabel_2,
             'seed' : 3553105877,
             'digest' : 'XXXXX',
             'eval': evaluate_multilabel2
         }   

     })
]

def get_config():
    hostname = gethostname()
    for (sys_name, pat), config in CONFIGS:
        if sys_name == name and match(pat, hostname):
            return config
    raise Exception('No matching config for %s, %s!' % (name, hostname))
