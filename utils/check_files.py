import os

path = '/home/jrudascas/Desktop/Tesis/data/hcp'

for root, dirs, files in os.walk(path, topdown=False):
   for name in files:
      if (os.stat(os.path.join(root, name)).st_size / 1024 / 1024) < 900:
         print('File: ' + os.path.join(root, name) + ' Size: ' + str(os.stat(os.path.join(root, name)).st_size/1024/1024))

   #for name in dirs:
   #   print('Folder: ' + os.path.join(root, name))