import os
import gzip

path = '/home/jrudascas/Desktop/Tesis/data/datasets/test-retest/s3'

for root, dirs, files in os.walk(path, topdown=False):
   for name in files:
      ext = name.split('.')[-1]
      new_name = name.split('.')[0] + '.' + name.split('.')[1]

      if ext == 'gz':
         input = gzip.GzipFile(os.path.join(root, name), 'rb')
         s = input.read()
         input.close()


         output = open(os.path.join(root, new_name), 'wb')
         output.write(s)
         output.close()