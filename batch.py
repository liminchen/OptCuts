import subprocess

from os import listdir
from os.path import isfile, join

# meshFolderPath = '/Users/mincli/Downloads/meshes/'
# meshFolderPath = '/Users/mincli/Downloads/meshes/needMoreTime/'
# meshFolderPath = '/Users/mincli/Downloads/meshes/closed/'
meshFolderPath = '/Users/mincli/Downloads/meshes/test_/'
# meshFolderPath = '/Users/mincli/Downloads/meshes/interiorSplitExp_/'
# meshFolderPath = '/Users/mincli/Downloads/meshes/bijectivityExp_/'
# meshFolderPath = '/Users/mincli/Downloads/meshes/small_/'
# meshFolderPath = '/Users/mincli/Downloads/meshes/fullBatch_/'
onlyfiles = [f for f in listdir(meshFolderPath) if isfile(join(meshFolderPath, f))]

priority = 'nice -n -10 '
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

# runCommand = priority + FracCutsPath + ' 1 2 /Users/mincli/Desktop/output_AutoCuts/'

for inputModelNameI in onlyfiles:
	# no prop, no filter
	runCommand = priority + FracCutsPath + ' 100 ' + meshFolderPath + inputModelNameI + ' 0.999 502 0'
	if subprocess.call([runCommand], shell=True):
		continue

	# # compute initial embedding and output for AutoCuts
	# runCommand = priority + FracCutsPath + ' 2 ' + meshFolderPath + inputModelNameI + ' 4'
	# if subprocess.call([runCommand], shell=True):
	# 	continue

	# # prop, no filter
	# runCommand = priority + FracCutsPath + ' 100 ' + meshFolderPath + inputModelNameI + ' 0.001 35 0'
	# if subprocess.call([runCommand], shell=True):
	# 	continue

	# # prop, no filter
	# runCommand = priority + FracCutsPath + ' 100 ' + meshFolderPath + inputModelNameI + ' 0.001 102 2'
	# if subprocess.call([runCommand], shell=True):
	# 	continue

	# # standard alt
	# runCommand = FracCutsPath + ' 100 ' + inputModelNameI + ' 0.2 0 0'
	# subprocess.call([runCommand], shell=True)
	# runCommand = FracCutsPath + ' 100 ' + inputModelNameI + ' 0.1 0 0'
	# subprocess.call([runCommand], shell=True)
	# runCommand = FracCutsPath + ' 100 ' + inputModelNameI + ' 0.05 0 0'
	# subprocess.call([runCommand], shell=True)

	# # AutoCuts
	# runCommand = FracCutsPath + ' 100 ' + meshFolderPath + inputModelNameI + ' 0.05 4 1'
	# if subprocess.call([runCommand], shell=True):
	# 	continue
	# runCommand = FracCutsPath + ' 100 ' + inputModelNameI + ' 0.1 4 1'
	# subprocess.call([runCommand], shell=True)
	# runCommand = FracCutsPath + ' 100 ' + inputModelNameI + ' 0.05 4 1'
	# subprocess.call([runCommand], shell=True)
