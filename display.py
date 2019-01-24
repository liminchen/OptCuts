import subprocess

from os import listdir
from os.path import isdir, join


resultsFolderPath = 'output/'
resultFolders = [f for f in listdir(resultsFolderPath) if isdir(join(resultsFolderPath, f))]

# list output folders for C++ to process
file = open("output/folderList.txt", "w+")

for resultFolderName in resultFolders:
	if resultFolderName != ".DS_Store":
		if ("OptCuts" in resultFolderName) or ("EBCuts" in resultFolderName) or ("DistMin" in resultFolderName):
			file.write("%s\n" % resultFolderName)

file.close()

# run OptCuts script to output expInfo for visualization
runCommand = "./build/OptCuts_bin 1 6 ./output/"
subprocess.call([runCommand], shell=True)

print("now you can open display/display.html to see the display")