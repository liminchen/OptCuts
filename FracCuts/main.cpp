#include "IglUtils.hpp"
#include "Optimizer.hpp"
#include "SymStretchEnergy.hpp"
#include "ARAPEnergy.hpp"
#include "SeparationEnergy.hpp"
#include "CohesiveEnergy.hpp"

#include <igl/readOFF.h>
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/harmonic.h>
#include <igl/arap.h>
#include <igl/avg_edge_length.h>
#include <igl/viewer/Viewer.h>
#include <igl/png/writePNG.h>

#include <sys/stat.h> // for mkdir

#include <fstream>
#include <string>


// optimization
std::vector<const FracCuts::TriangleSoup*> triSoup;
FracCuts::Optimizer* optimizer;
std::vector<FracCuts::Energy*> energyTerms;
std::vector<double> energyParams;
bool optimization_on = false;
int iterNum = 0;
bool converged = false;
bool autoHomotopy = true;
std::ofstream homoTransFile;
bool fractureMode = false;

std::ofstream logFile;
std::string outputFolderPath = "/Users/mincli/Desktop/output_FracCuts/";
const std::string meshFolder = "/Users/mincli/Desktop/meshes/";

// visualization
igl::viewer::Viewer viewer;
const int channel_initial = 0;
const int channel_result = 1;
int viewChannel = channel_result;
bool viewUV = true; // view UV or 3D model
double texScale = 1.0;
bool showSeam = true;
bool showBoundary = false;
bool showDistortion = true;
bool showTexture = true; // show checkerboard
bool isLighting = false;


void proceedOptimization(void)
{
    std::cout << "Iteration" << iterNum << ":" << std::endl;
    converged = optimizer->solve(1);
    iterNum = optimizer->getIterNum();
}

void updateViewerData_seam(const Eigen::MatrixXd& V)
{
    if(showSeam) {
        const double seamDistThres = 1.0e-2;
        
        viewer.core.show_lines = false;
        Eigen::VectorXd seamScore;
        triSoup[viewChannel]->computeSeamScore(seamScore);
        Eigen::MatrixXd color;
        FracCuts::IglUtils::mapScalarToColor_bin(seamScore, color, seamDistThres);
        viewer.data.set_edges(Eigen::MatrixXd(0, 3), Eigen::MatrixXi(0, 2), Eigen::RowVector3d(0.0, 0.0, 0.0));
        for(int eI = 0; eI < triSoup[viewChannel]->cohE.rows(); eI++) {
            if(seamScore[eI] > seamDistThres) {
                viewer.data.add_edges(V.row(triSoup[viewChannel]->cohE.row(eI)[0]),
                    V.row(triSoup[viewChannel]->cohE.row(eI)[1]), color.row(eI));
                if(triSoup[viewChannel]->cohE.row(eI)[2] >= 0) {
                    viewer.data.add_edges(V.row(triSoup[viewChannel]->cohE.row(eI)[2]),
                        V.row(triSoup[viewChannel]->cohE.row(eI)[3]), color.row(eI));
                }
            }
            else if((seamScore[eI] < 0.0) && showBoundary) {
                viewer.data.add_edges(V.row(triSoup[viewChannel]->cohE.row(eI)[0]),
                    V.row(triSoup[viewChannel]->cohE.row(eI)[1]), color.row(eI));
            }
        }
    }
    else {
        viewer.core.show_lines = true;
        viewer.data.set_edges(Eigen::MatrixXd(0, 3), Eigen::MatrixXi(0, 2), Eigen::RowVector3d(0.0, 0.0, 0.0));
    }
}

void updateViewerData_distortion(void)
{
    if(showDistortion) {
        Eigen::VectorXd distortionPerElem;
        energyTerms[0]->getEnergyValPerElem(*triSoup[viewChannel], distortionPerElem, true);
        Eigen::MatrixXd color_distortionVis;
        FracCuts::IglUtils::mapScalarToColor(distortionPerElem, color_distortionVis, 4.0, 6.0);
        viewer.data.set_colors(color_distortionVis);
    }
    else {
        viewer.data.set_colors(Eigen::RowVector3d(1.0, 1.0, 0.0));
    }
}

void updateViewerData(void)
{
    const Eigen::MatrixXd UV_vis = triSoup[viewChannel]->V * texScale;
    if(viewUV) {
        if((UV_vis.rows() != viewer.data.V.rows()) ||
           (triSoup[viewChannel]->F.rows() != viewer.data.F.rows()))
        {
            viewer.data.clear();
        }
        viewer.data.set_mesh(UV_vis, triSoup[viewChannel]->F);
//        viewer.core.align_camera_center(UV[0], F[0]);
        viewer.core.align_camera_center(UV_vis, triSoup[viewChannel]->F);
        
        viewer.core.show_texture = false;
        viewer.core.lighting_factor = 0.0;

        updateViewerData_seam(UV_vis);
    }
    else {
        if((triSoup[viewChannel]->V_rest.rows() != viewer.data.V.rows()) ||
           (UV_vis.rows() != viewer.data.V_uv.rows()) ||
           (triSoup[viewChannel]->F.rows() != viewer.data.F.rows()))
        {
            viewer.data.clear();
        }
        viewer.data.set_mesh(triSoup[viewChannel]->V_rest, triSoup[viewChannel]->F);
//        viewer.core.align_camera_center(V[0], F[0]);
        viewer.core.align_camera_center(triSoup[viewChannel]->V_rest, triSoup[viewChannel]->F);
        
        if(showTexture) {
            viewer.data.set_uv(UV_vis);
            viewer.core.show_texture = true;
        }
        else {
            viewer.core.show_texture = false;
        }
        
        if(isLighting) {
            viewer.core.lighting_factor = 1.0;
        }
        else {
            viewer.core.lighting_factor = 0.0;
        }
        
        updateViewerData_seam(triSoup[viewChannel]->V_rest);
    }
    updateViewerData_distortion();
    
    viewer.data.compute_normals();
}

void saveScreenshot(const std::string& filePath, double scale = 1.0)
{
    int width = static_cast<int>(scale * (viewer.core.viewport[2] - viewer.core.viewport[0]));
    int height = static_cast<int>(scale * (viewer.core.viewport[3] - viewer.core.viewport[1]));
    
    // Allocate temporary buffers for image
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> R(width, height);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> G(width, height);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> B(width, height);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> A(width, height);
    
    // Draw the scene in the buffers
    viewer.core.draw_buffer(viewer.data, viewer.opengl, false, R, G, B, A);
    
    // Save it to a PNG
    igl::png::writePNG(R, G, B, A, filePath);
}

bool key_down(igl::viewer::Viewer& viewer, unsigned char key, int modifier)
{
    if((key >= '0') && (key <= '9')) {
        int changeToChannel = key - '0';
        if((changeToChannel < triSoup.size()) && (viewChannel != changeToChannel)) {
            viewChannel = changeToChannel;
        }
    }
    else {
        switch (key)
        {
            case ' ': {
                proceedOptimization();
                viewChannel = channel_result;
                break;
            }
                
            case '/': {
                optimization_on = !optimization_on;
                if(optimization_on) {
                    if(converged) {
                        optimization_on = false;
                        std::cout << "optimization converged." << std::endl;
                    }
                    else {
                        if(iterNum == 0) {
                            homoTransFile.open(outputFolderPath + "homotopyTransition.txt");
                            assert(homoTransFile.is_open());
                            saveScreenshot(outputFolderPath + std::to_string(iterNum) + ".png", 1.0);
                        }
                        std::cout << "start/resume optimization, press again to pause." << std::endl;
                        viewer.core.is_animating = true;
                    }
                }
                else {
                    std::cout << "pause optimization, press again to resume." << std::endl;
                    viewer.core.is_animating = false;
                }
                break;
            }
                
            case 'u':
            case 'U': {
                viewUV = !viewUV;
                break;
            }
                
            case 's':
            case 'S': {
                showSeam = !showSeam;
                break;
            }
                
            case 'e':
            case 'E': {
                showBoundary = !showBoundary;
                break;
            }
                
            case 'd':
            case 'D': {
                showDistortion = !showDistortion;
                break;
            }
                
            case 'c':
            case 'C': {
                showTexture = !showTexture;
                break;
            }
                
            case 'b':
            case 'B': {
                isLighting = !isLighting;
                break;
            }
                
            case 'h':
            case 'H': { // mannual homotopy optimization
                FracCuts::SeparationEnergy *sepE = NULL;
                for(const auto eTermI : energyTerms) {
                    sepE = dynamic_cast<FracCuts::SeparationEnergy*>(eTermI);
                    if(sepE != NULL) {
                        break;
                    }
                }
                
                if(sepE != NULL) {
                    saveScreenshot(outputFolderPath + "homotopyFS_" + std::to_string(sepE->getSigmaParam()) + ".png", 1.0);
                    triSoup[channel_result]->save(outputFolderPath + "homotopyFS_" + std::to_string(sepE->getSigmaParam()) + ".obj");
                    triSoup[channel_result]->saveAsMesh(outputFolderPath + "homotopyFS_" + std::to_string(sepE->getSigmaParam()) + "_mesh.obj");
                    
                    if(sepE->decreaseSigma())
                    {
                        homoTransFile << iterNum << std::endl;
                        optimizer->computeLastEnergyVal();
                        converged = false;
                        if(fractureMode) {
                            optimizer->createFracture();
                        }
                        else {
                            optimizer->updatePrecondMtrAndFactorize();
                        }
                    }
                    else {
                        triSoup[channel_result]->saveAsMesh(outputFolderPath + "result_mesh_01UV.obj", true);
                        
                        optimization_on = false;
                        viewer.core.is_animating = false;
                        std::cout << "optimization converged." << std::endl;
                        homoTransFile.close();
                    }
                }
                else {
                    std::cout << "No homotopy settings!" << std::endl;
                }
                break;
            }
                
            case 'o':
            case 'O': {
                saveScreenshot(outputFolderPath + std::to_string(iterNum) + ".png", 1.0);
                triSoup[channel_result]->save(outputFolderPath + std::to_string(iterNum) + ".obj");
                triSoup[channel_result]->saveAsMesh(outputFolderPath + std::to_string(iterNum) + "_mesh.obj");
                break;
            }
                
            default:
                break;
        }
    }
    
    updateViewerData();

    return false;
}

bool preDrawFunc(igl::viewer::Viewer& viewer)
{
    if(optimization_on)
    {
        proceedOptimization();
        viewChannel = channel_result;
        updateViewerData();
        
//        if((iterNum < 10) || (iterNum % 10 == 0)) {
//            saveScreenshot(outputFolderPath + std::to_string(iterNum) + ".png", 1.0);
//        }
        
        if(converged) {
            FracCuts::SeparationEnergy *sepE = NULL;
            for(const auto eTermI : energyTerms) {
                sepE = dynamic_cast<FracCuts::SeparationEnergy*>(eTermI);
                if(sepE != NULL) {
                    break;
                }
            }
            
            if(sepE != NULL) {
                saveScreenshot(outputFolderPath + "homotopy_" + std::to_string(
                    sepE->getSigmaParam()) + ".png", 1.0);
                triSoup[channel_result]->save(outputFolderPath + "homotopy_" + std::to_string(
                    sepE->getSigmaParam()) + ".obj");
                triSoup[channel_result]->saveAsMesh(outputFolderPath + "homotopy_" + std::to_string(
                    sepE->getSigmaParam()) + "_mesh.obj");
            }
            else {
                saveScreenshot(outputFolderPath + "result.png", 1.0);
                triSoup[channel_result]->save(outputFolderPath + "result_triSoup.obj");
                triSoup[channel_result]->saveAsMesh(outputFolderPath + "result_mesh.obj");
            }
            
            if(autoHomotopy && sepE && sepE->decreaseSigma())
            {
                homoTransFile << iterNum << std::endl;
                optimizer->computeLastEnergyVal();
                converged = false;
                if(fractureMode) {
                    optimizer->createFracture();
                }
                else {
                    optimizer->updatePrecondMtrAndFactorize();
                }
            }
            else {
                triSoup[channel_result]->saveAsMesh(outputFolderPath + "result_mesh_01UV.obj", true);
                
                optimization_on = false;
                viewer.core.is_animating = false;
                std::cout << "optimization converged." << std::endl;
                homoTransFile.close();
            }
        }
    }
    return false;
}

int main(int argc, char *argv[])
{
    int progMode = 0;
    if(argc > 1) {
        progMode = std::stoi(argv[1]);
    }
    switch(progMode) {
        case 0:
            // optimization mode
            std::cout << "Optimization mode" << std::endl;
            break;
            
        case 1: {
            // diagnostic mode
            if(argc > 2) {
                int diagMode = 0;
                diagMode = std::stoi(argv[2]);
                switch(diagMode) {
                    case 0: {
                        // compute SVD to a sparse matrix
                        if(argc > 3) {
                            Eigen::SparseMatrix<double> mtr;
                            FracCuts::IglUtils::loadSparseMatrixFromFile(argv[3], mtr);
                            Eigen::MatrixXd mtr_dense(mtr);
                            Eigen::BDCSVD<Eigen::MatrixXd> svd(mtr_dense);
                            std::cout << "singular values of mtr:" << std::endl << svd.singularValues() << std::endl;
                            std::cout << "det(mtr) = " << mtr_dense.determinant() << std::endl;
                        }
                        else {
                            std::cout << "Please enter matrix file path!" << std::endl;
                        }
                        break;
                    }
                        
                    default:
                        std::cout << "No diagMode " << diagMode << std::endl;
                        break;
                }
            }
            else {
                std::cout << "Please enter diagMode!" << std::endl;
            }
            return 0;
        }
            
        case 2: {
            // mesh processing mode
            if(argc > 2) {
                Eigen::MatrixXd V, UV, N;
                Eigen::MatrixXi F, FUV, FN;
                std::string meshPath = meshFolder + std::string(argv[2]);
                const std::string suffix = meshPath.substr(meshPath.find_last_of('.'));
                if(suffix == ".off") {
                    igl::readOFF(meshPath, V, F);
                }
                else if(suffix == ".obj") {
                    igl::readOBJ(meshPath, V, UV, N, F, FUV, FN);
                }
                else {
                    std::cout << "unkown mesh file format!" << std::endl;
                    return -1;
                }
                
                if(argc > 3) {
                    int procMode = 0;
                    procMode = std::stoi(argv[3]);
                    switch(procMode) {
                        case 0: {
                            // invert normal of a mesh
                            for(int triI = 0; triI < F.rows(); triI++) {
                                const Eigen::RowVector3i temp = F.row(triI);
                                F(triI, 1) = temp[2];
                                F(triI, 2) = temp[1];
                            }
                            igl::writeOBJ(meshFolder + "processedMesh.obj", V, F);
                            break;
                        }
                            
                        case 1: {
                            // turn a triangle soup into a mesh
                            if(V.rows() != F.rows() * 3) {
                                std::cout << "Input model is not a triangle soup!" << std::endl;
                                return -1;
                            }
                            
                            if(UV.rows() != V.rows()) {
                                std::cout << "UV coordinates not valid, will generate separate rigid mapping UV!" << std::endl;
                            }
                            
                            FracCuts::TriangleSoup inputTriSoup(V, F, UV, Eigen::MatrixXi(), false);
                            if(argc > 4) {
                                // input original model to get cohesive edge information
                                Eigen::MatrixXd V0;
                                Eigen::MatrixXi F0;
                                std::string meshPath = meshFolder + std::string(argv[4]);
                                const std::string suffix = meshPath.substr(meshPath.find_last_of('.'));
                                if(suffix == ".off") {
                                    igl::readOFF(meshPath, V0, F0);
                                }
                                else if(suffix == ".obj") {
                                    igl::readOBJ(meshPath, V0, F0);
                                }
                                else {
                                    std::cout << "unkown mesh file format!" << std::endl;
                                    return -1;
                                }

                                inputTriSoup.cohE = FracCuts::TriangleSoup(V0, F0, Eigen::MatrixXd()).cohE;
                                inputTriSoup.computeFeatures();
                            }
                            inputTriSoup.saveAsMesh(meshFolder + "processedMesh.obj", true);
                            break;
                        }
                            
                        default:
                            std::cout << "No procMode " << procMode << std::endl;
                            break;
                    }
                }
                else {
                    std::cout << "Please enter procMode!" << std::endl;
                }
            }
            else {
                std::cout << "Please enter mesh file path!" << std::endl;
            }
            
            return 0;
        }
            
        default: {
            std::cout<< "No progMode " << progMode << std::endl;
            return 0;
        }
    }
    
    // Optimization mode
    
    std::string meshFileName("cone2.0.obj");
    if(argc > 2) {
        meshFileName = std::string(argv[2]);
    }
    std::string meshFilePath = meshFolder + meshFileName;
    std::string meshName = meshFileName.substr(0, meshFileName.find_last_of('.'));
    // Load mesh
    Eigen::MatrixXd V, UV, N;
    Eigen::MatrixXi F, FUV, FN;
    const std::string suffix = meshFilePath.substr(meshFilePath.find_last_of('.'));
    bool loadSucceed = false;
    if(suffix == ".off") {
        loadSucceed = igl::readOFF(meshFilePath, V, F);
    }
    else if(suffix == ".obj") {
        loadSucceed = igl::readOBJ(meshFilePath, V, UV, N, F, FUV, FN);
    }
    else {
        std::cout << "unkown mesh file format!" << std::endl;
        return -1;
    }
    if(!loadSucceed) {
        std::cout << "failed to load mesh!" << std::endl;
        return -1;
    }
//    //DEBUG
//    FracCuts::TriangleSoup squareMesh(FracCuts::P_SQUARE, 1.0, 0.1, false);
//    V = squareMesh.V_rest;
//    F = squareMesh.F;
    
    // Set lambda
    double lambda = 0.5;
    if(argc > 3) {
        lambda = std::stod(argv[3]);
        if((lambda != lambda) || (lambda < 0.0) || (lambda > 1.0)) {
            std::cout << "Overwrite invalid lambda " << lambda << " to 0.5" << std::endl;
            lambda = 0.5;
        }
    }
    else {
        std::cout << "Use default lambda = " << lambda << std::endl;
    }
    
    // Set delta
    double delta = 16;
    if(argc > 4) {
        delta = std::stod(argv[4]);
        if((delta != delta) || (delta < 0.0)) {
            std::cout << "Overwrite invalid delta " << delta << " to 16" << std::endl;
            delta = 16;
        }
    }
    else {
        std::cout << "Use default delta = " << delta << std::endl;
    }
    
    bool startWithTriSoup = ((lambda == 0.0) ? false : true);
    if(argc > 5) {
        startWithTriSoup = !!std::stoi(argv[5]);
    }
    else {
        std::cout << "Use default, start from " << (startWithTriSoup ? "triangle soup": "full mesh") << std::endl;
    }
    if(startWithTriSoup) {
        assert((lambda > 0.0) && "must have edge energy to start from triangle soup!");
    }
    const std::string startDS = (startWithTriSoup ? "soup" : ((lambda == 0.0) ? "SD": "frac"));
    
    std::string folderTail = "";
    if(argc > 6) {
        if(argv[6][0] != '_') {
            folderTail += '_';
        }
        folderTail += argv[6];
    }
    
    if(UV.rows() != 0) {
        triSoup.emplace_back(new FracCuts::TriangleSoup(V, F, UV, FUV, startWithTriSoup));
//        FracCuts::TriangleSoup temp(V, F, UV, FUV, false);
//        Eigen::VectorXd tempVec = Eigen::VectorXd::Zero(F.rows());
//        temp.separateTriangle(tempVec, -1.0);
//        triSoup.emplace_back(new FracCuts::TriangleSoup(temp));
        outputFolderPath += meshName + "_input_" + FracCuts::IglUtils::rtos(lambda) + "_" + FracCuts::IglUtils::rtos(delta) +
            "_" +startDS + folderTail;
    }
    else {
        // * Harmonic map for initialization
        Eigen::VectorXi bnd;
        igl::boundary_loop(F, bnd); // Find the open boundary
        if(bnd.size()) {
            // Map the boundary to a circle, preserving edge proportions
            Eigen::MatrixXd bnd_uv;
            igl::map_vertices_to_circle(V, bnd, bnd_uv);
            
        //    // Harmonic parametrization
        //    UV.resize(UV.size() + 1);
        //    igl::harmonic(V[0], F[0], bnd, bnd_uv, 1, UV[0]);
            
            // Harmonic map with uniform weights
            Eigen::SparseMatrix<double> A, M;
            FracCuts::IglUtils::computeUniformLaplacian(F, A);
            Eigen::MatrixXd UV_Tutte;
            igl::harmonic(A, M, bnd, bnd_uv, 1, UV_Tutte);
            
            triSoup.emplace_back(new FracCuts::TriangleSoup(V, F, UV_Tutte, Eigen::MatrixXi(), startWithTriSoup));
            outputFolderPath += meshName + "_Tutte_" + FracCuts::IglUtils::rtos(lambda) + "_" + FracCuts::IglUtils::rtos(delta) +
                "_" + startDS + folderTail;
        }
        else {
            // rigid initialization for UV
            assert((lambda > 0.0) && startWithTriSoup);
            triSoup.emplace_back(new FracCuts::TriangleSoup(V, F, Eigen::MatrixXd()));
            outputFolderPath += meshName + "_rigid_" + FracCuts::IglUtils::rtos(lambda) + "_" + FracCuts::IglUtils::rtos(delta) +
                "_" + startDS + folderTail;
        }
    }
    
    mkdir(outputFolderPath.c_str(), 0777);
    outputFolderPath += '/';
    logFile.open(outputFolderPath + "log.txt");
    if(!logFile.is_open()) {
        std::cout << "failed to create log file, please ensure output directory is created successfully!" << std::endl;
        return -1;
    }
    
//    // * ARAP
//    igl::ARAPData arap_data;
//    arap_data.with_dynamics = false;
//    Eigen::VectorXi b  = Eigen::VectorXi::Zero(0);
//    Eigen::MatrixXd bc = Eigen::MatrixXd::Zero(0, 0);
//    
//    // Initialize ARAP
////    arap_data.max_iter = 300;
//    // 2 means that we're going to *solve* in 2d
//    arap_precomputation(V[0], F[0], 2, b, arap_data);
//    
//    // Solve arap using the harmonic map as initial guess
////    triSoup = FracCuts::TriangleSoup(V[0], F[0], UV[0]);
//    arap_solve(bc, arap_data, UV[0]);
    
    // * Our approach
    texScale = 10.0 / (triSoup[0]->bbox.row(1) - triSoup[0]->bbox.row(0)).maxCoeff();
    if(lambda != 1.0) {
        energyParams.emplace_back(1.0 - lambda);
    //    energyTerms.emplace_back(new FracCuts::ARAPEnergy());
        energyTerms.emplace_back(new FracCuts::SymStretchEnergy());
    }
    if(lambda != 0.0) {
        energyParams.emplace_back(lambda);
        energyTerms.emplace_back(new FracCuts::SeparationEnergy(triSoup[0]->avgEdgeLen * triSoup[0]->avgEdgeLen, delta));
    }
//    energyTerms.emplace_back(new FracCuts::CohesiveEnergy(triSoup[0]->avgEdgeLen, delta));
//    energyTerms.back()->checkEnergyVal(*triSoup[0]);
//    energyTerms.back()->checkGradient(*triSoup[0]);
//    energyTerms.back()->checkHessian(*triSoup[0]);
    optimizer = new FracCuts::Optimizer(*triSoup[0], energyTerms, energyParams);
    optimizer->precompute();
    triSoup.emplace_back(&optimizer->getResult());
    if((lambda > 0.0) && (!startWithTriSoup)) {
        // fracture mode
        fractureMode = true;
        optimizer->createFracture();
    }
    
    // Setup viewer and launch
    viewer.core.background_color << 1.0f, 1.0f, 1.0f, 0.0f;
    viewer.callback_key_down = &key_down;
    viewer.callback_pre_draw = &preDrawFunc;
    viewer.core.show_lines = true;
    viewer.core.orthographic = true;
    viewer.core.animation_max_fps = 60.0;
    updateViewerData();
    viewer.launch();
    
    
    // Before exit
    logFile.close();
    for(auto& eI : energyTerms) {
        delete eI;
    }
    delete optimizer;
    delete triSoup[0];
}
