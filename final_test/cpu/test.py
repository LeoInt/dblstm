import os
import subprocess

os.system("./app 100 4 320 > outfile")
a=""
with open("outfile") as f:
	a=f.read()

#a=subprocess.Popen(["./app","100","4","320"], stdout=subprocess.PIPE)
print a
