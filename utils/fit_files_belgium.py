import os
import gzip

path = '/home/jrudascas/Desktop/Tesis/data/datasets/uws'
import os

for root, dirs, files in os.walk(path, topdown=False):
   for name in files:
      ext = name.split('.')[-1]
      new_name = name.split('.')[0] + '.' + name.split('.')[1]

      if ext != 'gz' and ext != 'nii':
         os.remove(os.path.join(root, name))

      if name == 'fmprage.nii.gz':
         input = gzip.GzipFile(os.path.join(root, name), 'rb')
         s = input.read()
         input.close()

         output = open(os.path.join(root, 't1.nii'), 'wb')
         output.write(s)
         output.close()

         os.remove(os.path.join(root, name))

      elif ext == 'gz':
         input = gzip.GzipFile(os.path.join(root, name), 'rb')
         s = input.read()
         input.close()

         output = open(os.path.join(root, 'fmri.nii'), 'wb')
         output.write(s)
         output.close()

         os.remove(os.path.join(root, name))