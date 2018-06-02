#include "Types.hpp"
#include "IglUtils.hpp"
#include "Optimizer.hpp"
#include "SymStretchEnergy.hpp"
#include "ARAPEnergy.hpp"
#include "SeparationEnergy.hpp"
#include "CohesiveEnergy.hpp"
#include "GIF.hpp"
#include "Timer.hpp"

#include "Diagnostic.hpp"
#include "MeshProcessing.hpp"

#include <igl/readOFF.h>
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/harmonic.h>
#include <igl/arap.h>
#include <igl/avg_edge_length.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/png/writePNG.h>
#include <igl/euler_characteristic.h>
#include <igl/edge_lengths.h>

#include <sys/stat.h> // for mkdir

#include <fstream>
#include <string>
#include <ctime>


// optimization
FracCuts::MethodType methodType;
std::vector<const FracCuts::TriangleSoup*> triSoup;
int vertAmt_input;
FracCuts::TriangleSoup triSoup_backup;
FracCuts::Optimizer* optimizer;
std::vector<FracCuts::Energy*> energyTerms;
std::vector<double> energyParams;
bool bijectiveParam = false;
//bool bijectiveParam = true; //TODO: set as arguments!
bool rand1PInitCut = false;
//bool rand1PInitCut = true; //!!! for fast prototyping
double lambda_init;
bool optimization_on = false;
int iterNum = 0;
int iterNum_lastTopo = 0;
int converged = 0;
bool autoHomotopy = true;
std::ofstream homoTransFile;
bool fractureMode = false;
double fracThres = 0.0;
bool altBase = false;
bool outerLoopFinished = false;
const int boundMeasureType = 0; // 0: E_SD, 1: L2 Stretch
double upperBound = 4.0445;
const double convTol_upperBound = 1.0e-3; //TODO!!! related to avg edge len or upperBound?
std::vector<std::pair<double, double>> energyChanges_bSplit, energyChanges_iSplit, energyChanges_merge;
std::vector<std::vector<int>> paths_bSplit, paths_iSplit, paths_merge;
std::vector<Eigen::MatrixXd> newVertPoses_bSplit, newVertPoses_iSplit, newVertPoses_merge;
int opType_queried = -1;
std::vector<int> path_queried;
Eigen::MatrixXd newVertPos_queried;

std::ofstream logFile;
std::string outputFolderPath = "/Users/mincli/Desktop/output_FracCuts/";
const std::string meshFolder = "/Users/mincli/Desktop/meshes/";

// visualization
igl::opengl::glfw::Viewer viewer;
const int channel_initial = 0;
const int channel_result = 1;
const int channel_findExtrema = 2;
int viewChannel = channel_result;
bool viewUV = true; // view UV or 3D model
double texScale = 1.0;
bool showSeam = true;
Eigen::MatrixXd seamColor;
bool showBoundary = false;
int showDistortion = 1; // 0: don't show; 1: SD energy value; 2: L2 stretch value;
bool showTexture = true; // show checkerboard
bool isLighting = false;
bool showFracTail = true;
double secPast = 0.0;
time_t lastStart_world;
Timer timer, timer_step;
bool offlineMode = false;
bool saveInfo_postDraw = false;
std::string infoName = "";
bool isCapture3D = false;
int capture3DI = 0;
GifWriter GIFWriter;
const uint32_t GIFDelay = 10; //*10ms
double GIFScale = 0.5;


void proceedOptimization(int proceedNum = 1)
{
    for(int proceedI = 0; (proceedI < proceedNum) && (!converged); proceedI++) {
        std::cout << "Iteration" << iterNum << ":" << std::endl;
        converged = optimizer->solve(1);
        iterNum = optimizer->getIterNum();
    }
}

void updateViewerData_meshEdges(void)
{
    viewer.data().show_lines = !showSeam;
    
    viewer.data().set_edges(Eigen::MatrixXd(0, 3), Eigen::MatrixXi(0, 2), Eigen::RowVector3d(0.0, 0.0, 0.0));
    if(showSeam) {
        // only draw air mesh edges
        if(optimizer->isScaffolding() && viewUV && (viewChannel == channel_result)) {
            const Eigen::MatrixXd V_airMesh = optimizer->getAirMesh().V * texScale;
            for(int triI = 0; triI < optimizer->getAirMesh().F.rows(); triI++) {
                const Eigen::RowVector3i& triVInd = optimizer->getAirMesh().F.row(triI);
                for(int eI = 0; eI < 3; eI++) {
                    viewer.data().add_edges(V_airMesh.row(triVInd[eI]), V_airMesh.row(triVInd[(eI + 1) % 3]),
                                          Eigen::RowVector3d::Zero());
                }
            }
        }
    }
}

void updateViewerData_seam(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& UV)
{
    if(showSeam) {
        const double seamDistThres = 1.0e-2;
        Eigen::VectorXd seamScore;
        triSoup[viewChannel]->computeSeamScore(seamScore);
        
        const Eigen::VectorXd cohIndices = Eigen::VectorXd::LinSpaced(triSoup[viewChannel]->cohE.rows(),
                                                                0, triSoup[viewChannel]->cohE.rows() - 1);
        Eigen::MatrixXd color;
        FracCuts::IglUtils::mapScalarToColor(cohIndices, color, 0, cohIndices.rows() - 1, 1);
        
        //TODO: seamscore only for autocuts
        seamColor.resize(0, 3);
        for(int eI = 0; eI < triSoup[viewChannel]->cohE.rows(); eI++) {
            const Eigen::RowVector4i& cohE = triSoup[viewChannel]->cohE.row(eI);
            const auto finder = triSoup[viewChannel]->edge2Tri.find(std::pair<int, int>(cohE[0], cohE[1]));
            assert(finder != triSoup[viewChannel]->edge2Tri.end());
            const Eigen::RowVector3d& sn = triSoup[viewChannel]->triNormal.row(finder->second);
            if((seamScore[eI] > seamDistThres) || (methodType != FracCuts::MT_AUTOCUTS)) {
                // seam edge
                FracCuts::IglUtils::addThickEdge(V, F, UV, seamColor, color.row(eI), V.row(cohE[0]), V.row(cohE[1]),
                                                 triSoup[viewChannel]->virtualRadius * 0.005 * (viewUV ? texScale : 1.0),
                                                 texScale, !viewUV, sn);
                if(viewUV) {
                    FracCuts::IglUtils::addThickEdge(V, F, UV, seamColor, color.row(eI), V.row(cohE[2]), V.row(cohE[3]),
                                                     triSoup[viewChannel]->virtualRadius * 0.005 * (viewUV ? texScale : 1.0),
                                                     texScale, !viewUV, sn);
                }
            }
            else if((seamScore[eI] < 0.0) && showBoundary) {
                // boundary edge
                //TODO: debug!
                FracCuts::IglUtils::addThickEdge(V, F, UV, seamColor, color.row(eI), V.row(cohE[0]), V.row(cohE[1]),
                                                 triSoup[viewChannel]->virtualRadius * 0.005 * (viewUV ? texScale : 1.0),
                                                 texScale, !viewUV, sn);
            }
        }
    }
}

void updateViewerData_distortion(void)
{
    Eigen::MatrixXd color_distortionVis;
    
    switch(showDistortion) {
        case 1: { // show SD energy value
            Eigen::VectorXd distortionPerElem;
            energyTerms[0]->getEnergyValPerElem(*triSoup[viewChannel], distortionPerElem, true);
            FracCuts::IglUtils::mapScalarToColor(distortionPerElem, color_distortionVis, 4.0, 8.5);
            break;
        }
            
        case 2: { // show L2 stretch value
            Eigen::VectorXd l2StretchPerElem;
//            triSoup[viewChannel]->computeL2StretchPerElem(l2StretchPerElem);
//            dynamic_cast<FracCuts::SymStretchEnergy*>(energyTerms[0])->getDivGradPerElem(*triSoup[viewChannel], l2StretchPerElem);
//            std::cout << l2StretchPerElem << std::endl; //DEBUG
//            FracCuts::IglUtils::mapScalarToColor(l2StretchPerElem, color_distortionVis, 1.0, 2.0);
//            FracCuts::IglUtils::mapScalarToColor(l2StretchPerElem, color_distortionVis,
//                l2StretchPerElem.minCoeff(), l2StretchPerElem.maxCoeff());
            Eigen::VectorXd faceWeight;
            faceWeight.resize(triSoup[viewChannel]->F.rows());
            for(int fI = 0; fI < triSoup[viewChannel]->F.rows(); fI++) {
                const Eigen::RowVector3i& triVInd = triSoup[viewChannel]->F.row(fI);
                faceWeight[fI] = (triSoup[viewChannel]->vertWeight[triVInd[0]] +
                                  triSoup[viewChannel]->vertWeight[triVInd[1]] +
                                  triSoup[viewChannel]->vertWeight[triVInd[2]]) / 3.0;
            }
            FracCuts::IglUtils::mapScalarToColor(faceWeight, color_distortionVis,
                faceWeight.minCoeff(), faceWeight.maxCoeff());
            break;
        }
    
        case 0: {
            color_distortionVis = Eigen::MatrixXd::Ones(triSoup[viewChannel]->F.rows(), 3);
            color_distortionVis.col(2).setZero();
            break;
        }
            
        default:
            assert(0 && "unknown distortion visualization option!");
            break;
    }
    
    if(optimizer->isScaffolding() && viewUV && (viewChannel == channel_result)) {
        optimizer->getScaffold().augmentFColorwithAirMesh(color_distortionVis);
    }
    
    if(showSeam) {
        color_distortionVis.conservativeResize(color_distortionVis.rows() + seamColor.rows(), 3);
        color_distortionVis.bottomRows(seamColor.rows()) = seamColor;
    }
    
    viewer.data().set_colors(color_distortionVis);
}

void updateViewerData(void)
{
    Eigen::MatrixXd UV_vis = triSoup[viewChannel]->V * texScale;
    Eigen::MatrixXi F_vis = triSoup[viewChannel]->F;
    if(viewUV) {
        if(optimizer->isScaffolding() && (viewChannel == channel_result)) {
            optimizer->getScaffold().augmentUVwithAirMesh(UV_vis, texScale);
            optimizer->getScaffold().augmentFwithAirMesh(F_vis);
        }
        UV_vis.conservativeResize(UV_vis.rows(), 3);
        UV_vis.rightCols(1) = Eigen::VectorXd::Zero(UV_vis.rows());
        updateViewerData_seam(UV_vis, F_vis, UV_vis);
        
        if((UV_vis.rows() != viewer.data().V.rows()) ||
           (F_vis.rows() != viewer.data().F.rows()))
        {
            viewer.data().clear();
        }
        viewer.data().set_mesh(UV_vis, F_vis);
        viewer.core.align_camera_center(UV_vis, F_vis);
        
        viewer.data().show_texture = false;
        viewer.core.lighting_factor = 0.0;

        updateViewerData_meshEdges();
        
        viewer.data().set_points(Eigen::MatrixXd::Zero(0, 3), Eigen::RowVector3d(0.0, 0.0, 0.0));
        if(showFracTail) {
            for(const auto& tailVI : triSoup[viewChannel]->fracTail) {
                viewer.data().add_points(UV_vis.row(tailVI), Eigen::RowVector3d(0.0, 0.0, 0.0));
            }
        }
    }
    else {
        Eigen::MatrixXd V_vis = triSoup[viewChannel]->V_rest;
        updateViewerData_seam(V_vis, F_vis, UV_vis);
        
        if((V_vis.rows() != viewer.data().V.rows()) ||
           (UV_vis.rows() != viewer.data().V_uv.rows()) ||
           (F_vis.rows() != viewer.data().F.rows()))
        {
            viewer.data().clear();
        }
        viewer.data().set_mesh(V_vis, F_vis);
        viewer.core.align_camera_center(V_vis, F_vis);
        
        if(showTexture) {
            viewer.data().set_uv(UV_vis);
            viewer.data().show_texture = true;
        }
        else {
            viewer.data().show_texture = false;
        }
        
        if(isLighting) {
            viewer.core.lighting_factor = 1.0;
        }
        else {
            viewer.core.lighting_factor = 0.0;
        }
        
        updateViewerData_meshEdges();
        
        viewer.data().set_points(Eigen::MatrixXd::Zero(0, 3), Eigen::RowVector3d(0.0, 0.0, 0.0));
        if(showFracTail) {
            for(const auto& tailVI : triSoup[viewChannel]->fracTail) {
                viewer.data().add_points(V_vis.row(tailVI), Eigen::RowVector3d(0.0, 0.0, 0.0));
            }
        }
    }
    updateViewerData_distortion();
    
    viewer.data().compute_normals();
}

void saveScreenshot(const std::string& filePath, double scale = 1.0, bool writeGIF = false, bool writePNG = true)
{
    if(writeGIF) {
        scale = GIFScale;
    }
    
    int width = static_cast<int>(scale * (viewer.core.viewport[2] - viewer.core.viewport[0]));
    int height = static_cast<int>(scale * (viewer.core.viewport[3] - viewer.core.viewport[1]));
    
    // Allocate temporary buffers for image
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> R(width, height);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> G(width, height);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> B(width, height);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> A(width, height);
    
    // Draw the scene in the buffers
    viewer.core.draw_buffer(viewer.data(), false, R, G, B, A);
    
    if(writePNG) {
        // Save it to a PNG
        igl::png::writePNG(R, G, B, A, filePath);
    }
    
    if(writeGIF) {
        std::vector<uint8_t> img(width * height * 4);
        for(int rowI = 0; rowI < width; rowI++) {
            for(int colI = 0; colI < height; colI++) {
                int indStart = (rowI + (height - 1 - colI) * width) * 4;
                img[indStart] = R(rowI, colI);
                img[indStart + 1] = G(rowI, colI);
                img[indStart + 2] = B(rowI, colI);
                img[indStart + 3] = A(rowI, colI);
            }
        }
        GifWriteFrame(&GIFWriter, img.data(), width, height, GIFDelay);
    }
}

void saveInfo(bool writePNG = true, bool writeGIF = true, bool writeMesh = true)
{
    saveScreenshot(outputFolderPath + infoName + ".png", 1.0, writeGIF, writePNG);
    if(writeMesh) {
//        triSoup[channel_result]->save(outputFolderPath + infoName + "_triSoup.obj");
        triSoup[channel_result]->saveAsMesh(outputFolderPath + infoName + "_mesh.obj");
    }
}

void saveInfoForPresent(const std::string fileName = "info.txt")
{
    std::ofstream file;
    file.open(outputFolderPath + fileName);
    assert(file.is_open());
    
    file << vertAmt_input << " " <<
        triSoup[channel_initial]->F.rows() << std::endl;
    
    file << iterNum << " " << optimizer->getTopoIter() << " 0 0 " << lambda_init << " " << 1.0 - energyParams[0] << std::endl;
    
    file << "0.0 0.0 " << timer.timing_total() << " " << secPast <<
        " topo" << timer.timing(0) << " desc" << timer.timing(1) << " scaf" << timer.timing(2) << " enUp" << timer.timing(3) <<
        " mtrComp" << timer_step.timing(0) << " mtrAssem" << timer_step.timing(1) << " symFac" << timer_step.timing(2) <<
        " numFac" << timer_step.timing(3) << " backSolve" << timer_step.timing(4) << " lineSearch" << timer_step.timing(5) <<
        " bSplit" << timer_step.timing(6) << " iSplit" << timer_step.timing(7) << " cMerge" << timer_step.timing(8) << std::endl;
    
    double seamLen;
    if(energyParams[0] == 1.0) {
        // pure distortion minimization mode for models with initial cuts also reflected on the surface as boundary edges...
        triSoup[channel_result]->computeBoundaryLen(seamLen);
        seamLen /= 2.0;
    }
    else {
        triSoup[channel_result]->computeSeamSparsity(seamLen, !fractureMode);
//        // for models with initial cuts also reflected on the surface as boundary edges...
//        double boundaryLen;
//        triSoup[channel_result]->computeBoundaryLen(boundaryLen);
//        seamLen += boundaryLen;
    }
    double distortion;
    energyTerms[0]->computeEnergyVal(*triSoup[channel_result], distortion);
    file << distortion << " " <<
        seamLen / triSoup[channel_result]->virtualRadius << std::endl;
    
    triSoup[channel_result]->outputStandardStretch(file);
    
    file << "initialSeams " << triSoup[channel_result]->initSeams.rows() << std::endl;
    file << triSoup[channel_result]->initSeams << std::endl;
    
    file.close();
}

void toggleOptimization(void)
{
    optimization_on = !optimization_on;
    if(optimization_on) {
        if(converged) {
            optimization_on = false;
            std::cout << "optimization converged." << std::endl;
        }
        else {
            if(iterNum == 0) {
                GifBegin(&GIFWriter, (outputFolderPath + "anim.gif").c_str(),
                         GIFScale * (viewer.core.viewport[2] - viewer.core.viewport[0]),
                         GIFScale * (viewer.core.viewport[3] - viewer.core.viewport[1]), GIFDelay);
                
                homoTransFile.open(outputFolderPath + "homotopyTransition.txt");
                assert(homoTransFile.is_open());
                saveScreenshot(outputFolderPath + "0.png", 1.0, true);
            }
            std::cout << "start/resume optimization, press again to pause." << std::endl;
            viewer.core.is_animating = true;
            
            time(&lastStart_world);
        }
    }
    else {
        std::cout << "pause optimization, press again to resume." << std::endl;
        viewer.core.is_animating = false;
        std::cout << "World Time:\nTime past: " << secPast << "s." << std::endl;
        secPast += difftime(time(NULL), lastStart_world);
    }
}

bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
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
                toggleOptimization();
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
                showDistortion++;
                if(showDistortion > 2) {
                    showDistortion = 0;
                }
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
            case 'H': { //!!!needsUpdate mannual homotopy optimization
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
                        optimizer->updatePrecondMtrAndFactorize();
                        if(fractureMode) {
                            // won't be called now since we are using standard AutoCuts
                            assert(0);
                            optimizer->createFracture(fracThres, false, !altBase);
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
                infoName = std::to_string(iterNum);
                saveInfo(true, false, true);
                break;
            }
                
            case 'p':
            case 'P': {
                showFracTail = !showFracTail;
                break;
            }
                
            default:
                break;
        }
    }
    
    updateViewerData();

    return false;
}

bool postDrawFunc(igl::opengl::glfw::Viewer& viewer)
{
    if(offlineMode && (iterNum == 0)) {
        toggleOptimization();
    }
    
    if(saveInfo_postDraw) {
        saveInfo_postDraw = false;
        saveInfo(outerLoopFinished, true, outerLoopFinished);
        // Note that the content saved in the screenshots are depends on where updateViewerData() is called
        if(outerLoopFinished) {
//            triSoup[channel_result]->saveAsMesh(outputFolderPath + infoName + "_mesh_01UV.obj", true);
        }
    }
    
    if(outerLoopFinished) { //var name change!!!
        if(!isCapture3D) {
            viewer.core.is_animating = true;
            isCapture3D = true;
        }
        else {
            if(capture3DI < 2) {
                // take screenshot
                std::cout << "Taking screenshot for 3D View " << capture3DI / 2 << std::endl;
                std::string filePath = outputFolderPath + "3DView" + std::to_string(capture3DI / 2) +
                    ((capture3DI % 2 == 0) ? "_seam.png" : "_distortion.png");
                saveScreenshot(filePath, 1.0);
                capture3DI++;
            }
            else {
                GifEnd(&GIFWriter);
                saveInfoForPresent();
                if(offlineMode) {
                    exit(0);
                }
                else {
                    viewer.core.is_animating = false;
                    isCapture3D = false;
                    outerLoopFinished = false;
                }
            }
        }
    }
    
    return false;
}

int computeOptPicked(const std::vector<std::pair<double, double>>& energyChanges0,
                     const std::vector<std::pair<double, double>>& energyChanges1,
                     double lambda)
{
    assert(!energyChanges0.empty());
    assert(!energyChanges1.empty());
    assert((lambda >= 0.0) && (lambda <= 1.0));
    
    double minEChange0 = __DBL_MAX__;
    for(int ecI = 0; ecI < energyChanges0.size(); ecI++) {
        if((energyChanges0[ecI].first == __DBL_MAX__) || (energyChanges0[ecI].second == __DBL_MAX__)) {
            continue;
        }
        double EwChange = energyChanges0[ecI].first * (1.0 - lambda) + energyChanges0[ecI].second * lambda;
        if(EwChange < minEChange0) {
            minEChange0 = EwChange;
        }
    }
    
    double minEChange1 = __DBL_MAX__;
    for(int ecI = 0; ecI < energyChanges1.size(); ecI++) {
        if((energyChanges1[ecI].first == __DBL_MAX__) || (energyChanges1[ecI].second == __DBL_MAX__)) {
            continue;
        }
        double EwChange = energyChanges1[ecI].first * (1.0 - lambda) + energyChanges1[ecI].second * lambda;
        if(EwChange < minEChange1) {
            minEChange1 = EwChange;
        }
    }
    
    assert((minEChange0 != __DBL_MAX__) || (minEChange1 != __DBL_MAX__));
    return (minEChange0 > minEChange1);
}

int computeBestCand(const std::vector<std::pair<double, double>>& energyChanges, double lambda,
                    double& bestEChange)
{
    assert(!energyChanges.empty());
    assert((lambda >= 0.0) && (lambda <= 1.0));
    
    bestEChange = __DBL_MAX__;
    int id_minEChange = -1;
    for(int ecI = 0; ecI < energyChanges.size(); ecI++) {
        if((energyChanges[ecI].first == __DBL_MAX__) || (energyChanges[ecI].second == __DBL_MAX__)) {
            continue;
        }
        double EwChange = energyChanges[ecI].first * (1.0 - lambda) + energyChanges[ecI].second * lambda;
        if(EwChange < bestEChange) {
            bestEChange = EwChange;
            id_minEChange = ecI;
        }
    }
//    assert(id_minEChange >= 0);
    
    return id_minEChange;
}

double updateLambda(double measure_bound, double lambda_SD = energyParams[0], double kappa = 1.0, double kappa2 = 1.0)
{
    lambda_SD = std::max(0.0, kappa * (measure_bound - (upperBound - convTol_upperBound / 2.0)) + kappa2 * lambda_SD / (1.0 - lambda_SD));
    return lambda_SD / (1.0 + lambda_SD);
}

bool updateLambda_stationaryV(bool cancelMomentum = true, bool checkConvergence = false)
{
    Eigen::MatrixXd edgeLengths; igl::edge_lengths(triSoup[channel_result]->V_rest, triSoup[channel_result]->F, edgeLengths);
    const double eps_E_se = 1.0e-3 * edgeLengths.minCoeff() / triSoup[channel_result]->virtualRadius;
    
    // measurement and energy value computation
    const double E_SD = optimizer->getLastEnergyVal(true) / energyParams[0];
    double E_se; triSoup[channel_result]->computeSeamSparsity(E_se);
    E_se /= triSoup[channel_result]->virtualRadius;
    double stretch_l2, stretch_inf, stretch_shear, compress_inf;
    triSoup[channel_result]->computeStandardStretch(stretch_l2, stretch_inf, stretch_shear, compress_inf);
    double measure_bound;
    switch(boundMeasureType) {
        case 0:
            measure_bound = E_SD;
            break;
            
        case 1:
            measure_bound = stretch_l2;
            break;
            
        default:
            assert(0 && "invalid bound measure type");
            break;
    }
    const double eps_lambda = std::min(1.0e-3, std::abs(updateLambda(measure_bound) - energyParams[0]));
    
    //TODO?: stop when first violates bounds from feasible, don't go to best feasible. check after each merge whether distortion is violated
    // oscillation detection
    static int iterNum_bestFeasible = -1;
    static FracCuts::TriangleSoup triSoup_bestFeasible;
    static double E_se_bestFeasible = __DBL_MAX__;
    static int lastStationaryIterNum = 0; //!!! still necessary because boundary and interior query are with same iterNum
    static std::map<double, std::vector<std::pair<double, double>>> configs_stationaryV; //!!! better also include topology information
    if(iterNum != lastStationaryIterNum) {
        // not a roll back config
        const double lambda = 1.0 - energyParams[0];
        bool oscillate = false;
        const auto low = configs_stationaryV.lower_bound(E_se);
        if(low == configs_stationaryV.end()) {
            // all less than E_se
            if(!configs_stationaryV.empty()) {
                // use largest element
                if(std::abs(configs_stationaryV.rbegin()->first - E_se) < eps_E_se)
                {
                    for(const auto& lambdaI : configs_stationaryV.rbegin()->second) {
                        if((std::abs(lambdaI.first - lambda) < eps_lambda) &&
                           (std::abs(lambdaI.second - E_SD) < eps_E_se))
                        {
                            oscillate = true;
                            logFile << configs_stationaryV.rbegin()->first << ", " << lambdaI.second << std::endl;
                            logFile << E_se << ", " << lambda << ", " << E_SD << std::endl;
                            break;
                        }
                    }
                }
            }
        }
        else if(low == configs_stationaryV.begin()) {
            // all not less than E_se
            if(std::abs(low->first - E_se) < eps_E_se)
            {
                for(const auto& lambdaI : low->second) {
                    if((std::abs(lambdaI.first - lambda) < eps_lambda) &&
                       (std::abs(lambdaI.second - E_SD) < eps_E_se))
                    {
                        oscillate = true;
                        logFile << low->first << ", " << lambdaI.first << ", " << lambdaI.second << std::endl;
                        logFile << E_se << ", " << lambda << ", " << E_SD << std::endl;
                        break;
                    }
                }
            }
            
        }
        else {
            const auto prev = std::prev(low);
            if(std::abs(low->first - E_se) < eps_E_se) {
                for(const auto& lambdaI : low->second) {
                    if((std::abs(lambdaI.first - lambda) < eps_lambda) &&
                       (std::abs(lambdaI.second - E_SD) < eps_E_se))
                    {
                        oscillate = true;
                        logFile << low->first << ", " << lambdaI.first << ", " << lambdaI.second << std::endl;
                        logFile << E_se << ", " << lambda << ", " << E_SD << std::endl;
                        break;
                    }
                }
            }
            if((!oscillate) && (std::abs(prev->first - E_se) < eps_E_se)) {
                for(const auto& lambdaI : prev->second) {
                    if((std::abs(lambdaI.first - lambda) < eps_lambda) &&
                       (std::abs(lambdaI.second - E_SD) < eps_E_se))
                    {
                        oscillate = true;
                        logFile << prev->first << ", " << lambdaI.first << ", " << lambdaI.second << std::endl;
                        logFile << E_se << ", " << lambda << ", " << E_SD << std::endl;
                        break;
                    }
                }
            }
        }
        
        // record best feasible UV map
        if((measure_bound <= upperBound) && (E_se < E_se_bestFeasible)) {
            iterNum_bestFeasible = iterNum;
            triSoup_bestFeasible = *triSoup[channel_result];
            E_se_bestFeasible = E_se;
        }
        
        if(oscillate && (iterNum_bestFeasible >= 0)) {
            // arrive at the best feasible config again
            logFile << "oscillation detected at measure = " << measure_bound << ", b = " << upperBound <<
                "lambda = " << energyParams[0] << std::endl;
            logFile << lastStationaryIterNum << ", " << iterNum << std::endl;
            if(iterNum_bestFeasible != iterNum) {
                homoTransFile << iterNum_bestFeasible << std::endl;
                optimizer->setConfig(triSoup_bestFeasible, iterNum, optimizer->getTopoIter());
                logFile << "rolled back to best feasible in iter " << iterNum_bestFeasible << std::endl;
            }
            return false;
        }
        else {
            configs_stationaryV[E_se].emplace_back(std::pair<double, double>(lambda, E_SD));
        }
    }
    lastStationaryIterNum = iterNum;
    
    // convergence check
    if(checkConvergence) {
        if(measure_bound <= upperBound) {
            // save info at first feasible stationaryVT for comparison
            static bool saved = false;
            if(!saved) {
                logFile << "saving firstFeasibleS..." << std::endl;
                saveScreenshot(outputFolderPath + "firstFeasibleS.png", 0.5, false, true); //TODO: saved is before roll back...
                triSoup[channel_result]->saveAsMesh(outputFolderPath + "firstFeasibleS_mesh.obj");
                secPast += difftime(time(NULL), lastStart_world);
                saveInfoForPresent("info_firstFeasibleS.txt");
                time(&lastStart_world);
                saved = true;
                logFile << "firstFeasibleS saved" << std::endl;
            }
            
            if(measure_bound >= upperBound - convTol_upperBound) {
                logFile << "all converged at measure = " << measure_bound << ", b = " << upperBound <<
                    " lambda = " << energyParams[0] << std::endl;
                if(iterNum_bestFeasible != iterNum) {
                    assert(iterNum_bestFeasible >= 0);
                    homoTransFile << iterNum_bestFeasible << std::endl;
                    optimizer->setConfig(triSoup_bestFeasible, iterNum, optimizer->getTopoIter());
                    logFile << "rolled back to best feasible in iter " << iterNum_bestFeasible << std::endl;
                }
                return false;
            }
        }
    }
    
    // lambda update (dual update)
    energyParams[0] = updateLambda(measure_bound);
    //!!! needs to be careful on lambda update space
    
    // critical lambda scheme
    if(checkConvergence) {
        // update lambda until feasible update on T might be triggered
        if(measure_bound > upperBound) {
            // need to cut further, increase energyParams[0]
            logFile << "curUpdated = " << energyParams[0] << ", increase" << std::endl;
            
//            std::cout << "iSplit:" << std::endl;
//            for(const auto& i : energyChanges_iSplit) {
//                std::cout << i.first << "," << i.second << std::endl;
//            }
//            std::cout << "bSplit:" << std::endl;
//            for(const auto& i : energyChanges_bSplit) {
//                std::cout << i.first << "," << i.second << std::endl;
//            }
//            std::cout << "merge:" << std::endl;
//            for(const auto& i : energyChanges_merge) {
//                std::cout << i.first << "," << i.second << std::endl;
//            }
            //!!!DEBUG: with bijectivity, sometimes there might be no valid bSplit that could decrease E_SD
            if((!energyChanges_merge.empty()) &&
               (computeOptPicked(energyChanges_bSplit, energyChanges_merge, 1.0 - energyParams[0]) == 1)){// &&
//               (computeOptPicked(energyChanges_iSplit, energyChanges_merge, 1.0 - energyParams[0]) == 1)) {
                // still picking merge
                do {
                    energyParams[0] = updateLambda(measure_bound);
                } while((computeOptPicked(energyChanges_bSplit, energyChanges_merge, 1.0 - energyParams[0]) == 1));// &&
//                        (computeOptPicked(energyChanges_iSplit, energyChanges_merge, 1.0 - energyParams[0]) == 1));
                
                logFile << "iterativelyUpdated = " << energyParams[0] << ", increase for switch" << std::endl;
            }
            
            double eDec_b, eDec_i;
            int id_pickingBSplit = computeBestCand(energyChanges_bSplit, 1.0 - energyParams[0], eDec_b);
            int id_pickingISplit = computeBestCand(energyChanges_iSplit, 1.0 - energyParams[0], eDec_i);
            while((eDec_b > 0.0) && (eDec_i > 0.0)) {
                energyParams[0] = updateLambda(measure_bound);
                id_pickingBSplit = computeBestCand(energyChanges_bSplit, 1.0 - energyParams[0], eDec_b);
                id_pickingISplit = computeBestCand(energyChanges_iSplit, 1.0 - energyParams[0], eDec_i);
            }
            if(eDec_b <= 0.0) {
                opType_queried = 0;
                path_queried = paths_bSplit[id_pickingBSplit];
                newVertPos_queried = newVertPoses_bSplit[id_pickingBSplit];
            }
            else {
                opType_queried = 1;
                path_queried = paths_iSplit[id_pickingISplit];
                newVertPos_queried = newVertPoses_iSplit[id_pickingISplit];
            }
            
            logFile << "iterativelyUpdated = " << energyParams[0] << ", increased, current eDec = " <<
                eDec_b << ", " << eDec_i << "; id: " << id_pickingBSplit << ", " << id_pickingISplit << std::endl;
        }
        else {
            if(energyChanges_merge.empty()) {
                logFile << "No merge operation available, end process!" << std::endl;
                energyParams[0] = 1.0 - eps_lambda;
                optimizer->updateEnergyData(true, false, false);
                if(iterNum_bestFeasible != iterNum) {
                    homoTransFile << iterNum_bestFeasible << std::endl;
                    optimizer->setConfig(triSoup_bestFeasible, iterNum, optimizer->getTopoIter());
                }
                return false;
            }
            
            logFile << "curUpdated = " << energyParams[0] << ", decrease" << std::endl;
            
            //!!! also account for iSplit for this switch?
            if(computeOptPicked(energyChanges_bSplit, energyChanges_merge, 1.0 - energyParams[0]) == 0) {
                // still picking split
                do {
                    energyParams[0] = updateLambda(measure_bound);
                } while(computeOptPicked(energyChanges_bSplit, energyChanges_merge, 1.0 - energyParams[0]) == 0);
                
                logFile << "iterativelyUpdated = " << energyParams[0] << ", decrease for switch" << std::endl;
            }
            
            double eDec_m;
            int id_pickingMerge = computeBestCand(energyChanges_merge, 1.0 - energyParams[0], eDec_m);
            while(eDec_m > 0.0) {
                energyParams[0] = updateLambda(measure_bound);
                id_pickingMerge = computeBestCand(energyChanges_merge, 1.0 - energyParams[0], eDec_m);
            }
            opType_queried = 2;
            path_queried = paths_merge[id_pickingMerge];
            newVertPos_queried = newVertPoses_merge[id_pickingMerge];
            
            logFile << "iterativelyUpdated = " << energyParams[0] << ", decreased, current eDec = " << eDec_m << std::endl;
        }
    }
    
    // lambda value sanity check
    if(energyParams[0] > 1.0 - eps_lambda) {
        energyParams[0] = 1.0 - eps_lambda;
    }
    if(energyParams[0] < eps_lambda) {
        energyParams[0] = eps_lambda;
    }
    
    optimizer->updateEnergyData(true, false, false);
    
    logFile << "measure = " << measure_bound << ", b = " << upperBound << ", updated lambda = " << energyParams[0] << std::endl;
    return true;
}

void converge_preDrawFunc(igl::opengl::glfw::Viewer& viewer)
{
    infoName = "finalResult";
    
    if(!bijectiveParam) {
        // perform exact solve
//                            optimizer->setRelGL2Tol(1.0e-10);
        optimizer->setAllowEDecRelTol(false);
        //!! can recompute precondmtr if needed
        converged = false;
        optimizer->setPropagateFracture(false);
        while(!converged) {
            proceedOptimization(1000);
        }
    }
    
    secPast += difftime(time(NULL), lastStart_world);
    updateViewerData();
    
    optimizer->flushEnergyFileOutput();
    optimizer->flushGradFileOutput();
    
    optimization_on = false;
    viewer.core.is_animating = false;
    std::cout << "optimization converged, with " << secPast << "s." << std::endl;
    logFile << "optimization converged, with " << secPast << "s." << std::endl;
    homoTransFile.close();
    outerLoopFinished = true;
}

bool preDrawFunc(igl::opengl::glfw::Viewer& viewer)
{
    if(optimization_on)
    {
        if(offlineMode) {
            while(!converged) {
                proceedOptimization();
            }
        }
        else {
            proceedOptimization();
        }
        
//        viewChannel = channel_result;
        updateViewerData();
        
        FracCuts::SeparationEnergy *sepE = NULL;
        for(const auto eTermI : energyTerms) {
            sepE = dynamic_cast<FracCuts::SeparationEnergy*>(eTermI);
            if(sepE != NULL) {
                break;
            }
        }
        
        if(methodType == FracCuts::MT_AUTOCUTS) {
            assert(sepE != NULL);
            if((iterNum < 10) || (iterNum % 10 == 0)) {
//                saveScreenshot(outputFolderPath + std::to_string(iterNum) + ".png", 1.0);
                saveInfo_postDraw = true;
            }
        }
        
        if(converged) {
            saveInfo_postDraw = true;
            
            double stretch_l2, stretch_inf, stretch_shear, compress_inf;
            triSoup[channel_result]->computeStandardStretch(stretch_l2, stretch_inf, stretch_shear, compress_inf);
            double measure_bound;
            switch(boundMeasureType) {
                case 0:
                    measure_bound = optimizer->getLastEnergyVal(true) / energyParams[0];
                    break;
                    
                case 1:
                    measure_bound = stretch_l2;
                    break;
                    
                default:
                    assert(0 && "invalid bound measure type");
                    break;
            }
            
            switch(methodType) {
                case FracCuts::MT_AUTOCUTS: {
                    infoName = "homotopy_" + std::to_string(sepE->getSigmaParam());
                    if(autoHomotopy && sepE->decreaseSigma()) {
                        homoTransFile << iterNum << std::endl;
                        optimizer->computeLastEnergyVal();
                        converged = false;
                    }
                    else {
                        infoName = "finalResult";

                        // perform exact solve
//                        optimizer->setRelGL2Tol(1.0e-8);
                        optimizer->setAllowEDecRelTol(false);
                        converged = false;
                        while(!converged) {
                            proceedOptimization(1000);
                        }
                        secPast += difftime(time(NULL), lastStart_world);
                        updateViewerData();
                        
                        optimization_on = false;
                        viewer.core.is_animating = false;
                        std::cout << "optimization converged, with " << secPast << "s." << std::endl;
                        logFile << "optimization converged, with " << secPast << "s." << std::endl;
                        homoTransFile.close();
                        outerLoopFinished = true;
                    }
                    //!!!TODO: energy and grad file output!
                    // flush
                    // decrease delta will change E_s thus E_w
                    break;
                }
                    
                case FracCuts::MT_GEOMIMG: {
                    if(measure_bound <= upperBound) {
                        logFile << "measure reaches user specified upperbound " << upperBound << std::endl;
                        
                        infoName = "finalResult";
                        // perform exact solve
//                        optimizer->setRelGL2Tol(1.0e-10);
                        optimizer->setAllowEDecRelTol(false);
                        converged = false;
                        while(!converged) {
                            proceedOptimization(1000);
                        }
                        secPast += difftime(time(NULL), lastStart_world);
                        updateViewerData();
                        
                        optimization_on = false;
                        viewer.core.is_animating = false;
                        std::cout << "optimization converged, with " << secPast << "s." << std::endl;
                        logFile << "optimization converged, with " << secPast << "s." << std::endl;
                        homoTransFile.close();
                        outerLoopFinished = true;
                    }
                    else {
                        infoName = std::to_string(iterNum);
                        
                        // continue to make geometry image cuts
                        homoTransFile << iterNum << std::endl;
                        assert(optimizer->createFracture(fracThres, false, false));
                        converged = false;
                    }
                    optimizer->flushEnergyFileOutput();
                    optimizer->flushGradFileOutput();
                    break;
                }
                    
                case FracCuts::MT_OURS_FIXED:
                case FracCuts::MT_OURS: {
                    infoName = std::to_string(iterNum);
                    if(converged == 2) {
                        converged = 0;
                        return false;
                    }
                    
                    if((methodType == FracCuts::MT_OURS) && (measure_bound <= upperBound)) {
                        // save info once bound is reached for comparison
                        static bool saved = false;
                        if(!saved) {
                            saveScreenshot(outputFolderPath + "firstFeasible.png", 0.5, false, true);
                            //                            triSoup[channel_result]->save(outputFolderPath + infoName + "_triSoup.obj");
                            triSoup[channel_result]->saveAsMesh(outputFolderPath + "firstFeasible_mesh.obj");
                            secPast += difftime(time(NULL), lastStart_world);
                            saveInfoForPresent("info_firstFeasible.txt");
                            time(&lastStart_world);
                            saved = true;
                        }
                    }
                    
                    // if necessary, turn on scaffolding for random one point initial cut
                    if(!optimizer->isScaffolding() && bijectiveParam && rand1PInitCut) {
                        optimizer->setScaffolding(true);
                        //TODO: other mode?
                    }
                    
                    double E_se; triSoup[channel_result]->computeSeamSparsity(E_se);
                    E_se /= triSoup[channel_result]->virtualRadius;
                    const double E_SD = optimizer->getLastEnergyVal(true) / energyParams[0];
                    const double E_w = optimizer->getLastEnergyVal(true) +
                        (1.0 - energyParams[0]) * E_se;
                    std::cout << iterNum << ": " << E_SD << " " << E_se << " " << triSoup[channel_result]->V_rest.rows() << std::endl;
                    logFile << iterNum << ": " << E_SD << " " << E_se << " " << triSoup[channel_result]->V_rest.rows() << std::endl;
                    optimizer->flushEnergyFileOutput();
                    optimizer->flushGradFileOutput();
                    homoTransFile << iterNum_lastTopo << std::endl;
                    
                    // continue to split boundary
                    if((methodType == FracCuts::MT_OURS) &&
                       (!updateLambda_stationaryV()))
                    {
                        // oscillation detected
                        converge_preDrawFunc(viewer);
                    }
                    else {
                        logFile << "boundary op V " << triSoup[channel_result]->V_rest.rows() << std::endl;
                        if(optimizer->createFracture(fracThres, false, !altBase)) {
                            converged = false;
                        }
                        else {
                            // if no boundary op, try interior split if split is the current best boundary op
                            if((measure_bound > upperBound) &&
                               optimizer->createFracture(fracThres, false, !altBase, true))
                            {
                                logFile << "interior split " << triSoup[channel_result]->V_rest.rows() << std::endl;
                                converged = false;
                            }
                            else {
                                homoTransFile << iterNum << std::endl; // mark stationaryVT
                                if((methodType == FracCuts::MT_OURS_FIXED) ||
                                   (!updateLambda_stationaryV(false, true)))
                                {
                                    // all converged
                                    converge_preDrawFunc(viewer);
                                }
                                else {
                                    // split or merge after lambda update
                                    optimizer->createFracture(opType_queried, path_queried, newVertPos_queried, !altBase);
                                    opType_queried = -1;
                                    converged = false;
                                }
                            }
                        }
                    }
                    break;
                }
                    
                case FracCuts::MT_NOCUT: {
                    converge_preDrawFunc(viewer);
                    break;
                }
            }
        }
    }
    else {
        if(isCapture3D && (capture3DI < 2)) {
            // change view accordingly
            double rotDeg = ((capture3DI < 8) ? (M_PI_2 * (capture3DI / 2)) : M_PI_2);
            Eigen::Vector3f rotAxis = Eigen::Vector3f::UnitY();
            if((capture3DI / 2) == 4) {
                rotAxis = Eigen::Vector3f::UnitX();
            }
            else if((capture3DI / 2) == 5) {
                rotAxis = -Eigen::Vector3f::UnitX();
            }
            viewer.core.trackball_angle = Eigen::Quaternionf(Eigen::AngleAxisf(rotDeg, rotAxis));
            viewChannel = channel_result;
            viewUV = false;
            showSeam = true;
            showBoundary = false;
            isLighting = false;
            showTexture = showDistortion = (capture3DI % 2);
            updateViewerData();
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
            
        case 100: {
            // offline optimization mode
            offlineMode = true;
            std::cout << "Offline optimization mode" << std::endl;
            break;
        }
            
        case 1: {
            // diagnostic mode
            FracCuts::Diagnostic::run(argc, argv);
            return 0;
        }
            
        case 2: {
            // mesh processing mode
            FracCuts::MeshProcessing::run(argc, argv);
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
    std::string meshFilePath;
    if(meshFileName.at(0) == '/') {
        std::cout << "The input mesh file name is gloabl mesh file path." << std::endl;
        meshFilePath = meshFileName;
        meshFileName = meshFileName.substr(meshFileName.find_last_of('/') + 1);
    }
    else {
        meshFilePath = meshFolder + meshFileName;
    }
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
    vertAmt_input = V.rows();
//    //DEBUG
//    FracCuts::TriangleSoup squareMesh(FracCuts::P_SQUARE, 1.0, 0.1, false);
//    V = squareMesh.V_rest;
//    F = squareMesh.F;
    
//    //!!! for AutoCuts comparison
//    std::ifstream distFile(outputFolderPath + "distortion.txt");
//    assert(distFile.is_open());
//    std::string resultName; double resultDistortion;
//    bool distFound = false;
//    while(!distFile.eof()) {
//        distFile >> resultName >> resultDistortion;
//        if(resultName.find(meshName + "_Tutte_") != std::string::npos) {
//            distFound = true;
//            upperBound = resultDistortion;
//            break;
//        }
//    }
//    distFile.close();
//    if(distFound) {
//        std::cout << "AutoCuts comparison: reset distortion bound to " << upperBound << std::endl;
//    }
//    else {
//        exit(0);
//    }
    
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
    lambda_init = lambda;
    
    // Set delta
    double delta = 4.0;
    if(argc > 4) {
        delta = std::stod(argv[4]);
        if((delta != delta) || (delta < 0.0)) {
            std::cout << "Overwrite invalid delta " << delta << " to 4" << std::endl;
            delta = 4.0;
        }
    }
    else {
        std::cout << "Use default delta = " << delta << std::endl;
    }
    
    if(argc > 5) {
        methodType = FracCuts::MethodType(std::stoi(argv[5]));
    }
    else {
        std::cout << "Use default method: ours." << std::endl;
    }
    bool startWithTriSoup = (methodType == FracCuts::MT_AUTOCUTS);
    std::string startDS;
    switch (methodType) {
        case FracCuts::MT_OURS_FIXED:
            assert(lambda < 1.0);
            startDS = "OursFixed";
            break;
            
        case FracCuts::MT_OURS:
            assert(lambda < 1.0);
            startDS = "OursBounded";
            break;
            
        case FracCuts::MT_GEOMIMG:
            assert(lambda < 1.0);
            startDS = "GeomImg";
            bijectiveParam = false;
            break;
            
        case FracCuts::MT_AUTOCUTS:
            assert(lambda > 0.0);
            assert(delta > 0.0);
            startDS = "AutoCuts";
            bijectiveParam = false;
            break;
            
        case FracCuts::MT_NOCUT:
            lambda = 0.0;
            startDS = "NoCut";
            break;
            
        default:
            assert(0 && "method type not valid!");
            break;
    }
    
    std::string folderTail = "";
    if(argc > 6) {
        if(argv[6][0] != '_') {
            folderTail += '_';
        }
        folderTail += argv[6];
    }
    
    if(UV.rows() != 0) {
//        //DEBUG
//        // generate Tutte embedding using input seams
//        Eigen::VectorXi bnd;
//        igl::boundary_loop(FUV, bnd);
//        Eigen::MatrixXd bnd_uv;
//        FracCuts::IglUtils::map_vertices_to_circle(UV, bnd, bnd_uv);
//        Eigen::SparseMatrix<double> A, M;
//        FracCuts::IglUtils::computeUniformLaplacian(FUV, A);
//        igl::harmonic(A, M, bnd, bnd_uv, 1, UV);
        
        // with input UV
        FracCuts::TriangleSoup *temp = new FracCuts::TriangleSoup(V, F, UV, FUV, startWithTriSoup);
        outputFolderPath += meshName + "_input_" + FracCuts::IglUtils::rtos(lambda) + "_" +
            FracCuts::IglUtils::rtos(delta) + "_" +startDS + folderTail;
        
        std::vector<std::vector<int>> bnd_all;
        igl::boundary_loop(temp->F, bnd_all);
        int UVGridDim = std::ceil(std::sqrt(bnd_all.size()));
        if(!temp->checkInversion() || (bijectiveParam && (UVGridDim > 1))) {
            std::cout << "local injectivity violated in given input UV map, " <<
                "or multi-chart bijective UV map needs to be ensured, " <<
                "obtaining new initial UV map by applying Tutte's embedding..." << std::endl;
            
            Eigen::VectorXi bnd_stacked;
            Eigen::MatrixXd bnd_uv_stacked;
            int curBndVAmt = 0;
            for(int bndI = 0; bndI < bnd_all.size(); bndI++) {
                // map boundary to unit circle
                bnd_stacked.conservativeResize(curBndVAmt + bnd_all[bndI].size());
                bnd_stacked.tail(bnd_all[bndI].size()) = Eigen::VectorXi::Map(bnd_all[bndI].data(),
                                                                              bnd_all[bndI].size());

                Eigen::MatrixXd bnd_uv;
                igl::map_vertices_to_circle(temp->V_rest,
                                            bnd_stacked.tail(bnd_all[bndI].size()),
                                            bnd_uv);
                double xOffset = bndI % UVGridDim * 2.1, yOffset = bndI / UVGridDim * 2.1;
                for(int bnd_uvI = 0; bnd_uvI < bnd_uv.rows(); bnd_uvI++) {
                    bnd_uv(bnd_uvI, 0) += xOffset;
                    bnd_uv(bnd_uvI, 1) += yOffset;
                }
                bnd_uv_stacked.conservativeResize(curBndVAmt + bnd_uv.rows(), 2);
                bnd_uv_stacked.bottomRows(bnd_uv.rows()) = bnd_uv;

                curBndVAmt = bnd_stacked.size();
            }
            // Harmonic map with uniform weights
            Eigen::MatrixXd UV_Tutte;
            Eigen::SparseMatrix<double> A, M;
            FracCuts::IglUtils::computeUniformLaplacian(temp->F, A);
            igl::harmonic(A, M, bnd_stacked, bnd_uv_stacked, 1, temp->V);
            
            if(!temp->checkInversion()) {
                std::cout << "local injectivity still violated in the computed initial UV map, " <<
                    "please carefully check UV topology for e.g. non-manifold vertices. " <<
                    "Exit program..." << std::endl;
                exit(-1);
            }
        }
        
        triSoup.emplace_back(temp);
//        temp->saveAsMesh("/Users/mincli/Desktop/output_FracCuts/test.obj");//DEBUG
    }
    else {
        // no input UV
        // * Harmonic map for initialization
        Eigen::VectorXi bnd;
        igl::boundary_loop(F, bnd); // Find the open boundary
        if(bnd.size()) {
            // disk-topology
            
            //TODO: what if it has multiple boundaries? or multi-components?
            // Map the boundary to a circle, preserving edge proportions
            Eigen::MatrixXd bnd_uv;
//            igl::map_vertices_to_circle(V, bnd, bnd_uv);
            FracCuts::IglUtils::map_vertices_to_circle(V, bnd, bnd_uv);
            
            Eigen::MatrixXd UV_Tutte;
            
//            // Harmonic parametrization
//            igl::harmonic(V, F, bnd, bnd_uv, 1, UV_Tutte);
            
            // Harmonic map with uniform weights
            Eigen::SparseMatrix<double> A, M;
            FracCuts::IglUtils::computeUniformLaplacian(F, A);
            igl::harmonic(A, M, bnd, bnd_uv, 1, UV_Tutte);
//            FracCuts::IglUtils::computeMVCMtr(V, F, A);
//            FracCuts::IglUtils::fixedBoundaryParam_MVC(A, bnd, bnd_uv, UV_Tutte);
            
            triSoup.emplace_back(new FracCuts::TriangleSoup(V, F, UV_Tutte, Eigen::MatrixXi(), startWithTriSoup));
            outputFolderPath += meshName + "_Tutte_" + FracCuts::IglUtils::rtos(lambda) + "_" + FracCuts::IglUtils::rtos(delta) +
                "_" + startDS + folderTail;
        }
        else {
            // closed surface
            if(igl::euler_characteristic(V, F) != 2) {
                std::cout << "Input surface genus > 0 or has multiple connected components!" << std::endl;
                return -1;
            }
            
            if(startWithTriSoup) {
                // rigid initialization, the most stable initialization for AutoCuts...
                assert((lambda > 0.0) && startWithTriSoup);
                triSoup.emplace_back(new FracCuts::TriangleSoup(V, F, Eigen::MatrixXd()));
                outputFolderPath += meshName + "_rigid_" + FracCuts::IglUtils::rtos(lambda) + "_" + FracCuts::IglUtils::rtos(delta) +
                    "_" + startDS + folderTail;
            }
            else {
                FracCuts::TriangleSoup *temp = new FracCuts::TriangleSoup(V, F, Eigen::MatrixXd(), Eigen::MatrixXi(), false);
//                temp->farthestPointCut(); // open up a boundary for Tutte embedding
//                temp->highCurvOnePointCut();
                temp->onePointCut();
                rand1PInitCut = true;
                
                igl::boundary_loop(temp->F, bnd);
                assert(bnd.size());
                Eigen::MatrixXd bnd_uv;
                FracCuts::IglUtils::map_vertices_to_circle(temp->V_rest, bnd, bnd_uv);
                Eigen::SparseMatrix<double> A, M;
                FracCuts::IglUtils::computeUniformLaplacian(temp->F, A);
                Eigen::MatrixXd UV_Tutte;
                igl::harmonic(A, M, bnd, bnd_uv, 1, UV_Tutte);
                triSoup.emplace_back(new FracCuts::TriangleSoup(V, F, UV_Tutte, temp->F, startWithTriSoup, temp->initSeamLen));
                
                delete temp;
                outputFolderPath += meshName + "_Tutte_" + FracCuts::IglUtils::rtos(lambda) + "_" + FracCuts::IglUtils::rtos(delta) +
                                "_" + startDS + folderTail;
            }
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
    
    // setup timer
    timer.new_activity("topology");
    timer.new_activity("descent");
    timer.new_activity("scaffolding");
    timer.new_activity("energyUpdate");
    
    timer_step.new_activity("matrixComputation");
    timer_step.new_activity("matrixAssembly");
    timer_step.new_activity("symbolicFactorization");
    timer_step.new_activity("numericalFactorization");
    timer_step.new_activity("backSolve");
    timer_step.new_activity("lineSearch");
    timer_step.new_activity("boundarySplit");
    timer_step.new_activity("interiorSplit");
    timer_step.new_activity("cornerMerge");
    
    // * Our approach
    texScale = 10.0 / (triSoup[0]->bbox.row(1) - triSoup[0]->bbox.row(0)).maxCoeff();
    if(lambda != 1.0) {
        energyParams.emplace_back(1.0 - lambda);
//        energyTerms.emplace_back(new FracCuts::ARAPEnergy());
        energyTerms.emplace_back(new FracCuts::SymStretchEnergy());
//        energyTerms.back()->checkEnergyVal(*triSoup[0]);
//        energyTerms.back()->checkGradient(*triSoup[0]);
//        energyTerms.back()->checkHessian(*triSoup[0], true);
    }
    if((lambda != 0.0) && startWithTriSoup) {
        //DEBUG alternating framework
        energyParams.emplace_back(lambda);
        energyTerms.emplace_back(new FracCuts::SeparationEnergy(triSoup[0]->avgEdgeLen * triSoup[0]->avgEdgeLen, delta));
//        energyTerms.emplace_back(new FracCuts::CohesiveEnergy(triSoup[0]->avgEdgeLen, delta));
//        energyTerms.back()->checkEnergyVal(*triSoup[0]);
//        energyTerms.back()->checkGradient(*triSoup[0]);
//        energyTerms.back()->checkHessian(*triSoup[0]);
    }
    optimizer = new FracCuts::Optimizer(*triSoup[0], energyTerms, energyParams, 0, false, bijectiveParam && !rand1PInitCut); // for random one point initial cut, don't need air meshes in the beginning since it's impossible for a quad to intersect itself
    //TODO: bijectivity for other mode?
    optimizer->precompute();
    triSoup.emplace_back(&optimizer->getResult());
    triSoup_backup = optimizer->getResult();
    triSoup.emplace_back(&optimizer->getData_findExtrema()); // for visualizing UV map for finding extrema
    if((lambda > 0.0) && (!startWithTriSoup)) {
        //!!!TODO: put into switch(methodType)
        // fracture mode
        fractureMode = true;
        
        if(delta == 0.0) {
            altBase = true;
        }
    }
    
//    //TEST: regional seam placement
//    std::ifstream vWFile("/Users/mincli/Desktop/output_FracCuts/" + meshName + "_selected.txt");
//    if(vWFile.is_open()) {
//        while(!vWFile.eof()) {
//            int selected;
//            vWFile >> selected;
//            if(selected < optimizer->getResult().vertWeight.size()) {
//                optimizer->getResult().vertWeight[selected] = 100.0;
//            }
//        }
//        vWFile.close();
//    }
//    FracCuts::IglUtils::smoothVertField(optimizer->getResult(), optimizer->getResult().vertWeight);
    
    // Setup viewer and launch
    viewer.core.background_color << 1.0f, 1.0f, 1.0f, 0.0f;
    viewer.callback_key_down = &key_down;
    viewer.callback_pre_draw = &preDrawFunc;
    viewer.callback_post_draw = &postDrawFunc;
    viewer.data().show_lines = true;
    viewer.core.orthographic = true;
    viewer.core.camera_zoom *= 1.9;
    viewer.core.animation_max_fps = 60.0;
    viewer.data().point_size = 15.0f; //TODO: make it adaptive
    viewer.data().show_overlay = true;
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
