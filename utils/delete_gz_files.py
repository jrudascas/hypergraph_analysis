import os
import gzip

path = '/home/jrudascas/Desktop/Tesis/data/datasets/test-retest/s3'
import os

for root, dirs, files in os.walk(path, topdown=False):
   for name in files:
      ext = name.split('.')[-1]
      new_name = name.split('.')[0] + '.' + name.split('.')[1]

      if ext == 'gz':
         os.remove(os.path.join(root, name))