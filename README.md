# OptCuts
* FracCuts: source code
* FracCuts.xcodeproj: XCode project file
* paper: paper draft
* batch.py: a python script to automatically run a batch of examples

## Compile
IntelTBB, PARDISO, LibIGL(with Eigen(3.3+), Triangle, GLAD, GLFW, stb_image), and libgfortran are also needed.
If you are using PARDISO6, you will also need an openMP compatible compiler.

(Will make IntelTBB and PARDISO optional, and develop a no-visualization version.)

## Command Line Arguments
Format: progName mode inputMeshPath lambda delta methodType [anyStringYouLike]

Example: FracCuts 0 /Users/mincli/Downloads/meshes/closed/bunny.obj 0.025 6 0 test
* progName
  * FracCuts
* mode
  * 0: real-time optimization mode, UV coordinates change in each inner iteration will be visualized
  * 100: offline optimization mode, only UV cooridinates change after each alternating / homotopy iteration will be visualized
  * 1: diagnostic mode, under development
  * 2: mesh processing mode, under development
* inputMeshPath
  * can be either absolute path or relative path to global variable meshFolder in main.cpp
  * .obj and .off are supported
* lambda (parameter of seam energy)
  * 0: minimize Symmetric Dirishlet energy with initial cuts, methodType must also be set to 4
  * (0,1): joint optimization. OptCuts start with 0.999 and iteratively update it according to the distortions of intermediate UV maps; 0.025 works generally well for OptCuts with fixed lambda on all inputs, which is also a good starting point for trying our AutoCuts implementation
  * 1: not meaningful
* delta (initial homotopy parameter)
  * [4,16]: recommanded initial homotopy parameter for AutoCuts
  * For OptCuts, 0 is without fracture propagation which is usually used for debugging, > 0 is with fracture propagation,
  and in our visualization webpage, it also serves as a classification ID
* methodType
  * 0: OptCuts, must also have lambda > 0 and lambda < 1
  * 1: An AutoCuts implementation, must also have lambda > 0
  * 2: EBCuts, using the extremity-boundary cutting strategy from Geometry Images within OptCuts framework
  * 3: OptCuts with fixed lambda, must also have lambda > 0 and lambda < 1
  * 4: Ed minimization with the initial seams
* anyStringYouLike
  * optional, the appended string to the name of a folder to be created for holding all output files

## Output Files
Our program will automatically create a folder according to the input command line arguments under the path given by 
global variable outputFolderPath in main.cpp for holding all output files for the current input:
* 0.png: initial UV map, colored by distortion (red = distorted, green = isometric)
* anim.gif: UV map changes during optimization process, colored by distortion
* finalResult.png: output UV map, colored by distortion
* finalResult_mesh.obj: input model with output UV
* 3DView0_distortion.png: input model visualized with checkerboard texture and distortion color map
* 3DView0_seam.png: input model visualized with seams
* energyValPerIter.txt: energy value of E_w, E_SD, E_se of each inner iteration
* gradientPerIter.txt: energy gradient of E_w, E_SD, E_se of each inner iteration
* homotopyTransition.txt: inner iteration numbers when each descent step or homotopy iteration ends
* info.txt: parameterization results quality output for webpage visualization
* log.txt: for debugging

## Keyboard Events:
* '/': start/restart or pause the optimization, in offline optimization mode, optimization is started with the program, 
while in real-time optimization mode, optimization needs to be started by the user
* ' ': proceed optimization by 1 inner iteration, only supported when optimization is paused
* '0': view input model or UV
* '1': view current model or UV
* 'u': toggle viewing model or UV, default is UV
* 'd': toggle distortion visualization, default is on
* 'c': toggle checkerboard texture visualization, default is off
* 's': toggle seam visualization, default is on
* 'e': toggle boundary visualization, default is off
* 'b': toggle lighting, default is off
* 'p': toggle viewing fracture tails (drawn as blue dots)
* 'o': take a screenshot and save the model with current UV
