import subprocess

from os import listdir
from os.path import isfile, join

meshFolderPath = 'input/'
onlyfiles = [f for f in listdir(meshFolderPath) if isfile(join(meshFolderPath, f))]

FracCutsPath = 'build/OptCuts_bin'

for inputModelNameI in onlyfiles:
	runCommand = FracCutsPath + ' 10 ' + meshFolderPath + inputModelNameI + ' 0.999 1 0 4.1 1 0'
	if subprocess.call([runCommand], shell=True):
		continue