import subprocess
import os

dir_path = ""
outdir = ""
for filename in os.listdir(dir_path):
    if filename.endswith('.doc'):
        subprocess.call(
            ['soffice', '--headless', '--convert-to', 'txt', dir_path + "\\" + filename, '--outdir', outdir])
        print('processing.......'+ dir_path + "\\" + filename)
