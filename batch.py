import subprocess

from os import listdir
from os.path import isfile, join

meshFolderPath = '/Users/mincli/Downloads/meshes/'
onlyfiles = [f for f in listdir(meshFolderPath) if isfile(join(meshFolderPath, f))]

FracCutsPath = '/Users/mincli/Library/Developer/Xcode/DerivedData/FracCuts-agmhaiwbuwzkmvfhishexuvkyjdo/Build/Products/Release/FracCuts'

# for inputModelNameI in onlyfiles:
# 	# current best
# 	runCommand = FracCutsPath + ' 100 ' + inputModelNameI + ' 0.1 1 0'
# 	if subprocess.call([runCommand], shell=True):
# 		continue

# for inputModelNameI in onlyfiles:
# 	# current best
# 	runCommand = FracCutsPath + ' 100 ' + inputModelNameI + ' 0.05 1 0'
# 	if subprocess.call([runCommand], shell=True):
# 		continue

for inputModelNameI in onlyfiles:
	# current best
	runCommand = FracCutsPath + ' 100 ' + inputModelNameI + ' 0.025 1 0'
	if subprocess.call([runCommand], shell=True):
		continue

	# # standard alt
	# runCommand = FracCutsPath + ' 100 ' + inputModelNameI + ' 0.2 0 0'
	# subprocess.call([runCommand], shell=True)
	# runCommand = FracCutsPath + ' 100 ' + inputModelNameI + ' 0.1 0 0'
	# subprocess.call([runCommand], shell=True)
	# runCommand = FracCutsPath + ' 100 ' + inputModelNameI + ' 0.05 0 0'
	# subprocess.call([runCommand], shell=True)

	# # AutoCuts
	# runCommand = FracCutsPath + ' 100 ' + inputModelNameI + ' 0.2 4 1'
	# subprocess.call([runCommand], shell=True)
	# runCommand = FracCutsPath + ' 100 ' + inputModelNameI + ' 0.1 4 1'
	# subprocess.call([runCommand], shell=True)
	# runCommand = FracCutsPath + ' 100 ' + inputModelNameI + ' 0.05 4 1'
	# subprocess.call([runCommand], shell=True)