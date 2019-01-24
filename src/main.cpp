#include "Types.hpp"
#include "IglUtils.hpp"
#include "Optimizer.hpp"
#include "SymDirichletEnergy.hpp"
#include "GIF.hpp"
#include "Timer.hpp"

#include "Diagnostic.hpp"
#include "MeshProcessing.hpp"

#include "cut_to_disk.hpp" // hasn't been pulled into the older version of libigl we use
#include <igl/cut_mesh.h>
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


Eigen::MatrixXd V, UV, N;
Eigen::MatrixXi F, FUV, FN;

// optimization
OptCuts::MethodType methodType;
std::vector<const OptCuts::TriMesh*> triSoup;
int vertAmt_input;
OptCuts::TriMesh triSoup_backup;
OptCuts::Optimizer* optimizer;
std::vector<OptCuts::Energy*> energyTerms;
std::vector<double> energyParams;

bool bijectiveParam = true;
bool rand1PInitCut = false;
double lambda_init;
bool optimization_on = false;
int iterNum = 0;
int converged = 0;
bool fractureMode = false;
double fracThres = 0.0;
bool topoLineSearch = true;
int initCutOption = 0;
bool outerLoopFinished = false;
double upperBound = 4.1;
const double convTol_upperBound = 1.0e-3;

std::vector<std::pair<double, double>> energyChanges_bSplit, energyChanges_iSplit, energyChanges_merge;
std::vector<std::vector<int>> paths_bSplit, paths_iSplit, paths_merge;
std::vector<Eigen::MatrixXd> newVertPoses_bSplit, newVertPoses_iSplit, newVertPoses_merge;

int opType_queried = -1;
std::vector<int> path_queried;
Eigen::MatrixXd newVertPos_queried;
bool reQuery = false;
double filterExp_in = 0.6;
int inSplitTotalAmt;

std::ofstream logFile;
std::string outputFolderPath = "output/";

// visualization
bool headlessMode = false;
igl::opengl::glfw::Viewer viewer;
const int channel_initial = 0;
const int channel_result = 1;
const int channel_findExtrema = 2;
int viewChannel = channel_result;
bool viewUV = true; // view UV or 3D model
double texScale = 1.0;
bool showSeam = true;
Eigen::MatrixXd seamColor;
int showDistortion = 1; // 0: don't show; 1: SD energy value; 2: L2 stretch value;
bool showTexture = true; // show checkerboard
bool isLighting = false;
bool showFracTail = true;
float fracTailSize = 15.0f;
bool offlineMode = false;
bool saveInfo_postDraw = false;
std::string infoName = "";
bool isCapture3D = false;
int capture3DI = 0;

GifWriter GIFWriter;
const uint32_t GIFDelay = 10; //*10ms
double GIFScale = 0.25;

double secPast = 0.0;
time_t lastStart_world;
Timer timer, timer_step;


void saveInfo(bool writePNG = true, bool writeGIF = true, bool writeMesh = true);

void proceedOptimization(int proceedNum = 1)
{
    for(int proceedI = 0; (proceedI < proceedNum) && (!converged); proceedI++) {
//        infoName = std::to_string(iterNum);
//        saveInfo(true, false, true); //!!! output mesh for making video, PNG output only works under online rendering mode
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
        const Eigen::VectorXd cohIndices = Eigen::VectorXd::LinSpaced(triSoup[viewChannel]->cohE.rows(),
                                                                0, triSoup[viewChannel]->cohE.rows() - 1);
        Eigen::MatrixXd color;
        color.resize(cohIndices.size(), 3);
        color.rowwise() = Eigen::RowVector3d(1.0, 0.5, 0.0);
        
        seamColor.resize(0, 3);
        double seamThickness = (viewUV ? (triSoup[viewChannel]->virtualRadius * 0.0007 / viewer.core.model_zoom * texScale) :
                                (triSoup[viewChannel]->virtualRadius * 0.006));
        for(int eI = 0; eI < triSoup[viewChannel]->cohE.rows(); eI++) {
            const Eigen::RowVector4i& cohE = triSoup[viewChannel]->cohE.row(eI);
            const auto finder = triSoup[viewChannel]->edge2Tri.find(std::pair<int, int>(cohE[0], cohE[1]));
            assert(finder != triSoup[viewChannel]->edge2Tri.end());
            const Eigen::RowVector3d& sn = triSoup[viewChannel]->triNormal.row(finder->second);
            
            // seam edge
            OptCuts::IglUtils::addThickEdge(V, F, UV, seamColor, color.row(eI), V.row(cohE[0]), V.row(cohE[1]), seamThickness, texScale, !viewUV, sn);
            if(viewUV) {
                OptCuts::IglUtils::addThickEdge(V, F, UV, seamColor, color.row(eI), V.row(cohE[2]), V.row(cohE[3]), seamThickness, texScale, !viewUV, sn);
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
            OptCuts::IglUtils::mapScalarToColor(distortionPerElem, color_distortionVis, 4.0, 8.5);
            break;
        }
            
        case 2: { // show face weight
            Eigen::VectorXd faceWeight;
            faceWeight.resize(triSoup[viewChannel]->F.rows());
            for(int fI = 0; fI < triSoup[viewChannel]->F.rows(); fI++) {
                const Eigen::RowVector3i& triVInd = triSoup[viewChannel]->F.row(fI);
                faceWeight[fI] = (triSoup[viewChannel]->vertWeight[triVInd[0]] +
                                  triSoup[viewChannel]->vertWeight[triVInd[1]] +
                                  triSoup[viewChannel]->vertWeight[triVInd[2]]) / 3.0;
            }
//            OptCuts::IglUtils::mapScalarToColor(faceWeight, color_distortionVis,
//                faceWeight.minCoeff(), faceWeight.maxCoeff());
            igl::colormap(igl::COLOR_MAP_TYPE_VIRIDIS, faceWeight, true, color_distortionVis);
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
        viewer.core.align_camera_center(UV_vis, F_vis);
        updateViewerData_seam(UV_vis, F_vis, UV_vis);
        
        if((UV_vis.rows() != viewer.data().V.rows()) ||
           (F_vis.rows() != viewer.data().F.rows()))
        {
            viewer.data().clear();
        }
        viewer.data().set_mesh(UV_vis, F_vis);
        
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
        viewer.core.align_camera_center(V_vis, F_vis);
        updateViewerData_seam(V_vis, F_vis, UV_vis);
        
        if((V_vis.rows() != viewer.data().V.rows()) ||
           (UV_vis.rows() != viewer.data().V_uv.rows()) ||
           (F_vis.rows() != viewer.data().F.rows()))
        {
            viewer.data().clear();
        }
        viewer.data().set_mesh(V_vis, F_vis);
        
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
    if(headlessMode) {
        return;
    }
    
    if(writeGIF) {
        scale = GIFScale;
    }
    viewer.data().point_size = fracTailSize * scale;
    
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
    
    viewer.data().point_size = fracTailSize;
}

void saveInfo(bool writePNG, bool writeGIF, bool writeMesh)
{
    saveScreenshot(outputFolderPath + infoName + ".png", 0.5, writeGIF, writePNG);
    if(writeMesh) {
        triSoup[channel_result]->saveAsMesh(outputFolderPath + infoName + "_mesh.obj", F);
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
            if((!headlessMode) && (iterNum == 0)) {
                GifBegin(&GIFWriter, (outputFolderPath + "anim.gif").c_str(),
                         GIFScale * (viewer.core.viewport[2] - viewer.core.viewport[0]),
                         GIFScale * (viewer.core.viewport[3] - viewer.core.viewport[1]), GIFDelay);
                
                saveScreenshot(outputFolderPath + "0.png", 0.5, true);
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
//        saveInfo(true, false, false);
        // Note that the content saved in the screenshots are depends on where updateViewerData() is called
    }
    
    if(outerLoopFinished) {
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
                saveScreenshot(filePath, 0.5);
                capture3DI++;
            }
            else {
                if(!headlessMode) {
                    GifEnd(&GIFWriter);
                }
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
    
    return id_minEChange;
}

bool checkCand(const std::vector<std::pair<double, double>>& energyChanges)
{
    for(const auto& candI : energyChanges) {
        if((candI.first < 0.0) || (candI.second < 0.0)) {
            return true;
        }
    }
    
    double minEChange = __DBL_MAX__;
    for(const auto& candI : energyChanges) {
        if(candI.first < minEChange) {
            minEChange = candI.first;
        }
        if(candI.second < minEChange) {
            minEChange = candI.second;
        }
    }
    std::cout << "candidates not valid, minEChange: " << minEChange << std::endl;
    return false;
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
    double measure_bound = E_SD;
    const double eps_lambda = std::min(1.0e-3, std::abs(updateLambda(measure_bound) - energyParams[0]));
    
    //TODO?: stop when first violates bounds from feasible, don't go to best feasible. check after each merge whether distortion is violated
    // oscillation detection
    static int iterNum_bestFeasible = -1;
    static OptCuts::TriMesh triSoup_bestFeasible;
    static double E_se_bestFeasible = __DBL_MAX__;
    static int lastStationaryIterNum = 0; // still necessary because boundary and interior query are with same iterNum
    static std::map<double, std::vector<std::pair<double, double>>> configs_stationaryV;
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
//                logFile << "saving firstFeasibleS..." << std::endl;
//                saveScreenshot(outputFolderPath + "firstFeasibleS.png", 0.5, false, true); //TODO: saved is before roll back...
//                triSoup[channel_result]->saveAsMesh(outputFolderPath + "firstFeasibleS_mesh.obj", F);
                secPast += difftime(time(NULL), lastStart_world);
//                saveInfoForPresent("info_firstFeasibleS.txt");
                time(&lastStart_world);
                saved = true;
//                logFile << "firstFeasibleS saved" << std::endl;
            }
            
            if(measure_bound >= upperBound - convTol_upperBound) {
                logFile << "all converged at measure = " << measure_bound << ", b = " << upperBound <<
                    " lambda = " << energyParams[0] << std::endl;
                if(iterNum_bestFeasible != iterNum) {
                    assert(iterNum_bestFeasible >= 0);
                    optimizer->setConfig(triSoup_bestFeasible, iterNum, optimizer->getTopoIter());
                    logFile << "rolled back to best feasible in iter " << iterNum_bestFeasible << std::endl;
                }
                return false;
            }
        }
    }
    
    // lambda update (dual update)
    energyParams[0] = updateLambda(measure_bound);
    //TODO: needs to be careful on lambda update space
    
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
            if((!energyChanges_merge.empty()) &&
               (computeOptPicked(energyChanges_bSplit, energyChanges_merge, 1.0 - energyParams[0]) == 1))
            {
                // still picking merge
                do {
                    energyParams[0] = updateLambda(measure_bound);
                } while((computeOptPicked(energyChanges_bSplit, energyChanges_merge, 1.0 - energyParams[0]) == 1));
                
                logFile << "iterativelyUpdated = " << energyParams[0] << ", increase for switch" << std::endl;
            }
            
            if((!checkCand(energyChanges_iSplit)) && (!checkCand(energyChanges_bSplit))) {
                // if filtering too strong
                reQuery = true;
                logFile << "enlarge filtering!" << std::endl;
            }
            else {
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
        }
        else {
            bool noOp = true;
            for(const auto ecI : energyChanges_merge) {
                if(ecI.first != __DBL_MAX__) {
                    noOp = false;
                    break;
                }
            }
            if(noOp) {
                logFile << "No merge operation available, end process!" << std::endl;
                energyParams[0] = 1.0 - eps_lambda;
                optimizer->updateEnergyData(true, false, false);
                if(iterNum_bestFeasible != iterNum) {
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
        optimizer->setAllowEDecRelTol(false);
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
        
        updateViewerData();
        
        if(converged) {
            saveInfo_postDraw = true;
            
            double stretch_l2, stretch_inf, stretch_shear, compress_inf;
            triSoup[channel_result]->computeStandardStretch(stretch_l2, stretch_inf, stretch_shear, compress_inf);
            double measure_bound = optimizer->getLastEnergyVal(true) / energyParams[0];
            
            switch(methodType) {
                case OptCuts::MT_EBCUTS: {
                    if(measure_bound <= upperBound) {
                        logFile << "measure reaches user specified upperbound " << upperBound << std::endl;
                        
                        infoName = "finalResult";
                        // perform exact solve
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
                        outerLoopFinished = true;
                    }
                    else {
                        infoName = std::to_string(iterNum);
                        
                        // continue to make geometry image cuts
                        assert(optimizer->createFracture(fracThres, false, false));
                        converged = false;
                    }
                    optimizer->flushEnergyFileOutput();
                    optimizer->flushGradFileOutput();
                    break;
                }
                    
                case OptCuts::MT_OPTCUTS_NODUAL:
                case OptCuts::MT_OPTCUTS: {
                    infoName = std::to_string(iterNum);
                    if(converged == 2) {
                        converged = 0;
                        return false;
                    }
                    
                    if((methodType == OptCuts::MT_OPTCUTS) && (measure_bound <= upperBound)) {
                        // save info once bound is reached for comparison
                        static bool saved = false;
                        if(!saved) {
//                            saveScreenshot(outputFolderPath + "firstFeasible.png", 0.5, false, true);
                            //                            triSoup[channel_result]->save(outputFolderPath + infoName + "_triSoup.obj");
//                            triSoup[channel_result]->saveAsMesh(outputFolderPath + "firstFeasible_mesh.obj", F);
                            secPast += difftime(time(NULL), lastStart_world);
//                            saveInfoForPresent("info_firstFeasible.txt");
                            time(&lastStart_world);
                            saved = true;
                        }
                    }
                    
                    // if necessary, turn on scaffolding for random one point initial cut
                    if(!optimizer->isScaffolding() && bijectiveParam && rand1PInitCut) {
                        optimizer->setScaffolding(true);
                    }
                    
                    double E_se; triSoup[channel_result]->computeSeamSparsity(E_se);
                    E_se /= triSoup[channel_result]->virtualRadius;
                    const double E_SD = optimizer->getLastEnergyVal(true) / energyParams[0];
                    
                    std::cout << iterNum << ": " << E_SD << " " << E_se << " " << triSoup[channel_result]->V_rest.rows() << std::endl;
                    logFile << iterNum << ": " << E_SD << " " << E_se << " " << triSoup[channel_result]->V_rest.rows() << std::endl;
                    optimizer->flushEnergyFileOutput();
                    optimizer->flushGradFileOutput();
                    
                    // continue to split boundary
                    if((methodType == OptCuts::MT_OPTCUTS) &&
                       (!updateLambda_stationaryV()))
                    {
                        // oscillation detected
                        converge_preDrawFunc(viewer);
                    }
                    else {
                        logFile << "boundary op V " << triSoup[channel_result]->V_rest.rows() << std::endl;
                        if(optimizer->createFracture(fracThres, false, topoLineSearch)) {
                            converged = false;
                        }
                        else {
                            // if no boundary op, try interior split if split is the current best boundary op
                            if((measure_bound > upperBound) &&
                               optimizer->createFracture(fracThres, false, topoLineSearch, true))
                            {
                                logFile << "interior split " << triSoup[channel_result]->V_rest.rows() << std::endl;
                                converged = false;
                            }
                            else {
                                if((methodType == OptCuts::MT_OPTCUTS_NODUAL) ||
                                   (!updateLambda_stationaryV(false, true)))
                                {
                                    // all converged
                                    converge_preDrawFunc(viewer);
                                }
                                else {
                                    // split or merge after lambda update
                                    if(reQuery) {
                                        filterExp_in += std::log(2.0) / std::log(inSplitTotalAmt);
                                        filterExp_in = std::min(1.0, filterExp_in);
                                        while(!optimizer->createFracture(fracThres, false, topoLineSearch, true))
                                        {
                                            filterExp_in += std::log(2.0) / std::log(inSplitTotalAmt);
                                            filterExp_in = std::min(1.0, filterExp_in);
                                        }
                                        reQuery = false;
                                        //TODO: set filtering param back?
                                    }
                                    else {
                                        optimizer->createFracture(opType_queried, path_queried, newVertPos_queried, topoLineSearch);
                                    }
                                    opType_queried = -1;
                                    converged = false;
                                }
                            }
                        }
                    }
                    break;
                }
                    
                case OptCuts::MT_DISTMIN: {
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
            isLighting = false;
            showTexture = capture3DI % 2;
            showDistortion = 2 - capture3DI % 2;
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
            
        case 10: {
            // offline optimization mode
            offlineMode = true;
            std::cout << "Offline optimization mode" << std::endl;
            break;
        }
        
        case 100: {
            // headless mode
            offlineMode = true;
            headlessMode = true;
            std::cout << "Headless mode" << std::endl;
            break;
        }
            
        case 1: {
            // diagnostic mode
            OptCuts::Diagnostic::run(argc, argv);
            return 0;
        }
            
        case 2: {
            // mesh processing mode
            OptCuts::MeshProcessing::run(argc, argv);
            return 0;
        }
            
        default: {
            std::cout<< "No progMode " << progMode << std::endl;
            return 0;
        }
    }
    
    // Optimization mode
    mkdir(outputFolderPath.c_str(), 0777);
    
    std::string meshFileName("cone2.0.obj");
    if(argc > 2) {
        meshFileName = std::string(argv[2]);
    }
    std::string meshFilePath = meshFileName;
    meshFileName = meshFileName.substr(meshFileName.find_last_of('/') + 1);
    
    std::string meshFolderPath = meshFilePath.substr(0, meshFilePath.find_last_of('/'));
    std::string meshName = meshFileName.substr(0, meshFileName.find_last_of('.'));
    
    // Load mesh
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
//    OptCuts::TriMesh squareMesh(OptCuts::P_SQUARE, 1.0, 0.1, false);
//    V = squareMesh.V_rest;
//    F = squareMesh.F;
    
    // Set lambda
    lambda_init = 0.999;
    if(argc > 3) {
        lambda_init = std::stod(argv[3]);
        if((lambda_init != lambda_init) || (lambda_init < 0.0) || (lambda_init >= 1.0)) {
            std::cout << "Overwrite invalid lambda " << lambda_init << " to 0.999" << std::endl;
            lambda_init = 0.999;
        }
    }
    else {
        std::cout << "Use default lambda = " << lambda_init << std::endl;
    }
    
    // Set testID
    double testID = 1.0; // test id for naming result folder
    if(argc > 4) {
        testID = std::stod(argv[4]);
        if((testID != testID) || (testID < 0.0)) {
            std::cout << "Overwrite invalid testID " << testID << " to 1" << std::endl;
            testID = 1.0;
        }
    }
    else {
        std::cout << "Use default testID = " << testID << std::endl;
    }
    
    if(argc > 5) {
        methodType = OptCuts::MethodType(std::stoi(argv[5]));
    }
    else {
        std::cout << "Use default method: OptCuts." << std::endl;
    }
    
    std::string startDS;
    switch (methodType) {
        case OptCuts::MT_OPTCUTS_NODUAL:
            startDS = "OptCuts_noDual";
            break;
            
        case OptCuts::MT_OPTCUTS:
            startDS = "OptCuts";
            break;
            
        case OptCuts::MT_EBCUTS:
            startDS = "EBCuts";
            bijectiveParam = false;
            break;
            
        case OptCuts::MT_DISTMIN:
            lambda_init = 0.0;
            startDS = "DistMin";
            break;
            
        default:
            assert(0 && "method type not valid!");
            break;
    }
    
    if(argc > 6) {
        upperBound = std::stod(argv[6]);
        if(upperBound == 0.0) {
            // read in b_d for comparing to other methods
            bool useScriptedBound = false;
            std::ifstream distFile(outputFolderPath + "distortion.txt");
            assert(distFile.is_open());
            
            std::string resultName; double resultDistortion;
            while(!distFile.eof()) {
                distFile >> resultName >> resultDistortion;
                if((resultName.find(meshName + "_Tutte_") != std::string::npos) ||
                   (resultName.find(meshName + "_input_") != std::string::npos) ||
                   (resultName.find(meshName + "_HighGenus_") != std::string::npos) ||
                   (resultName.find(meshName + "_rigid_") != std::string::npos) ||
                   (resultName.find(meshName + "_zbrush_") != std::string::npos) ||
                   (resultName.find(meshName + "_unwrella_") != std::string::npos))
                {
                    useScriptedBound = true;
                    upperBound = resultDistortion;
                    assert(upperBound > 4.0);
                    break;
                }
            }
            distFile.close();
            
            assert(useScriptedBound);
            std::cout << "Use scripted b_d = " << upperBound << std::endl;
        }
        else {
            if(upperBound <= 4.0) {
                std::cout << "input b_d <= 4.0! use 4.1 instead." << std::endl;
                upperBound = 4.1;
            }
            else {
                std::cout << "use b_d = " << upperBound << std::endl;
            }
        }
    }
    
    if(argc > 7) {
        bijectiveParam = std::stoi(argv[7]);
        std::cout << "bijectivity " << (bijectiveParam ? "ON" : "OFF") << std::endl;
    }
    
    if(argc > 8) {
        initCutOption = std::stoi(argv[8]);
    }
    switch(initCutOption) {
        case 0:
            std::cout << "random 2-edge initial cut for closed surface" << std::endl;
            break;
            
        case 1:
            std::cout << "farthest 2-point initial cut for closed surface" << std::endl;
            break;
            
        default:
            std::cout << "input initial cut option invalid, use default" << std::endl;
            std::cout << "random 2-edge initial cut for closed surface" << std::endl;
            initCutOption = 0;
            break;
    }
    
    std::string folderTail = "";
    if(argc > 9) {
        if(argv[9][0] != '_') {
            folderTail += '_';
        }
        folderTail += argv[9];
    }
    
    if(UV.rows() != 0)
    {
        // with input UV
        OptCuts::TriMesh *temp = new OptCuts::TriMesh(V, F, UV, FUV, false);
        outputFolderPath += meshName + "_input_" + OptCuts::IglUtils::rtos(lambda_init) + "_" +
            OptCuts::IglUtils::rtos(testID) + "_" +startDS + folderTail;
        
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
            OptCuts::IglUtils::computeUniformLaplacian(temp->F, A);
            igl::harmonic(A, M, bnd_stacked, bnd_uv_stacked, 1, temp->V);

            if(!temp->checkInversion()) {
                std::cout << "local injectivity still violated in the computed initial UV map, " <<
                    "please carefully check UV topology for e.g. non-manifold vertices. " <<
                    "Exit program..." << std::endl;
                exit(-1);
            }
        }
        
        triSoup.emplace_back(temp);
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
            OptCuts::IglUtils::map_vertices_to_circle(V, bnd, bnd_uv);
            
            Eigen::MatrixXd UV_Tutte;
            
            // Harmonic map with uniform weights
            Eigen::SparseMatrix<double> A, M;
            OptCuts::IglUtils::computeUniformLaplacian(F, A);
            igl::harmonic(A, M, bnd, bnd_uv, 1, UV_Tutte);
//            OptCuts::IglUtils::computeMVCMtr(V, F, A);
//            OptCuts::IglUtils::fixedBoundaryParam_MVC(A, bnd, bnd_uv, UV_Tutte);
            
            triSoup.emplace_back(new OptCuts::TriMesh(V, F, UV_Tutte, Eigen::MatrixXi(), false));
            outputFolderPath += meshName + "_Tutte_" + OptCuts::IglUtils::rtos(lambda_init) + "_" + OptCuts::IglUtils::rtos(testID) +
                "_" + startDS + folderTail;
        }
        else {
            // closed surface
            int genus = 1 - igl::euler_characteristic(V, F) / 2;
            if(genus != 0) {
                std::cout << "Input surface genus = " + std::to_string(genus) + " or has multiple connected components!" << std::endl;
                
                std::vector<std::vector<int>> cuts;
                igl::cut_to_disk(F, cuts);
                
                // record cohesive edge information,
                // transfer information format for cut_mesh
                OptCuts::TriMesh temp(V, F, Eigen::MatrixXd(), Eigen::MatrixXi(), false);
                Eigen::MatrixXi cutFlags(F.rows(), 3);
                Eigen::MatrixXi cohEdgeRecord;
                cutFlags.setZero();
                for(const auto& seamI : cuts) {
                    for(int segI = 0; segI + 1 < seamI.size(); segI++) {
                        std::pair<int, int> edge(seamI[segI], seamI[segI + 1]);
                        auto finder = temp.edge2Tri.find(edge);
                        assert(finder != temp.edge2Tri.end());
                        int i = 0;
                        for(; i < 3; i++) {
                            if(temp.F(finder->second, i) == edge.first) {
                                cutFlags(finder->second, i) = 1;
                                break;
                            }
                        }
                        
                        int cohERI = cohEdgeRecord.rows();
                        cohEdgeRecord.conservativeResize(cohERI + 1, 4);
                        cohEdgeRecord(cohERI, 0) = finder->second;
                        cohEdgeRecord(cohERI, 1) = i;
                        
                        edge.second = seamI[segI];
                        edge.first = seamI[segI + 1];
                        finder = temp.edge2Tri.find(edge);
                        assert(finder != temp.edge2Tri.end());
                        for(i = 0; i < 3; i++) {
                            if(temp.F(finder->second, i) == edge.first) {
                                cutFlags(finder->second, i) = 1;
                                break;
                            }
                        }
                        
                        cohEdgeRecord(cohERI, 2) = finder->second;
                        cohEdgeRecord(cohERI, 3) = i;
                    }
                }
                
                Eigen::MatrixXd Vcut;
                Eigen::MatrixXi Fcut;
                igl::cut_mesh(temp.V_rest, temp.F, cutFlags, Vcut, Fcut);
                igl::writeOBJ(outputFolderPath + meshName + "_disk.obj", Vcut, Fcut);
                V = Vcut;
                F = Fcut;
                
                igl::boundary_loop(F, bnd); // Find the open boundary
                assert(bnd.size());
                
                Eigen::MatrixXd bnd_uv;
                OptCuts::IglUtils::map_vertices_to_circle(V, bnd, bnd_uv);
                
                Eigen::MatrixXd UV_Tutte;
                
                // Harmonic map with uniform weights
                Eigen::SparseMatrix<double> A, M;
                OptCuts::IglUtils::computeUniformLaplacian(F, A);
                igl::harmonic(A, M, bnd, bnd_uv, 1, UV_Tutte);
                //            OptCuts::IglUtils::computeMVCMtr(V, F, A);
                //            OptCuts::IglUtils::fixedBoundaryParam_MVC(A, bnd, bnd_uv, UV_Tutte);
                
                OptCuts::TriMesh* ptr = new OptCuts::TriMesh(V, F, UV_Tutte, Eigen::MatrixXi(), false);
                ptr->buildCohEfromRecord(cohEdgeRecord);
                triSoup.emplace_back(ptr);
                outputFolderPath += meshName + "_HighGenus_" + OptCuts::IglUtils::rtos(lambda_init) + "_" + OptCuts::IglUtils::rtos(testID) + "_" + startDS + folderTail;
            }
            else {
                OptCuts::TriMesh *temp = new OptCuts::TriMesh(V, F, Eigen::MatrixXd(), Eigen::MatrixXi(), false);
                
                switch (initCutOption) {
                    case 0:
                        temp->onePointCut();
                        rand1PInitCut = true;
                        break;
                        
                    case 1:
                        temp->farthestPointCut();
                        break;
                        
                    default:
                        assert(0);
                        break;
                }
                
                igl::boundary_loop(temp->F, bnd);
                assert(bnd.size());
                
                Eigen::MatrixXd bnd_uv;
                OptCuts::IglUtils::map_vertices_to_circle(temp->V_rest, bnd, bnd_uv);
                
                Eigen::SparseMatrix<double> A, M;
                OptCuts::IglUtils::computeUniformLaplacian(temp->F, A);
                
                Eigen::MatrixXd UV_Tutte;
                igl::harmonic(A, M, bnd, bnd_uv, 1, UV_Tutte);
                
                triSoup.emplace_back(new OptCuts::TriMesh(V, F, UV_Tutte, temp->F, false, temp->initSeamLen));
                
                // try initialize one-point cut with different vertices
                // until no inversion is detected
                int splitVI = 0;
                while(!triSoup.back()->checkInversion(true)) {
                    std::cout << "element inversion detected during UV initialization " <<
                        "due to rounding errors, trying another vertex..." << std::endl;
                    
                    delete temp;
                    temp = new OptCuts::TriMesh(V, F, Eigen::MatrixXd(), Eigen::MatrixXi(), false);
                    temp->onePointCut(++splitVI);
                    
                    igl::boundary_loop(temp->F, bnd);
                    assert(bnd.size());
                    
                    OptCuts::IglUtils::map_vertices_to_circle(temp->V_rest, bnd, bnd_uv);
                    
                    OptCuts::IglUtils::computeUniformLaplacian(temp->F, A);
                    
                    igl::harmonic(A, M, bnd, bnd_uv, 1, UV_Tutte);
                    
                    delete triSoup.back();
                    triSoup.back() = new OptCuts::TriMesh(V, F, UV_Tutte, temp->F, false, temp->initSeamLen);
                }
                
                delete temp;
                outputFolderPath += meshName + "_Tutte_" + OptCuts::IglUtils::rtos(lambda_init) + "_" + OptCuts::IglUtils::rtos(testID) +
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
    energyParams.emplace_back(1.0 - lambda_init);
    energyTerms.emplace_back(new OptCuts::SymDirichletEnergy());
    
    optimizer = new OptCuts::Optimizer(*triSoup[0], energyTerms, energyParams, 0, false, bijectiveParam && !rand1PInitCut); // for random one point initial cut, don't need air meshes in the beginning since it's impossible for a quad to intersect itself
    
    optimizer->precompute();
    triSoup.emplace_back(&optimizer->getResult());
    triSoup_backup = optimizer->getResult();
    triSoup.emplace_back(&optimizer->getData_findExtrema()); // for visualizing UV map for finding extrema
    if(lambda_init > 0.0) {
        // fracture mode
        fractureMode = true;
    }
    
    /////////////////////////////////////////////////////////////////////////////
    // regional seam placement
    std::ifstream vWFile(meshFolderPath + "/" + meshName + "_selected.txt");
    if(vWFile.is_open()) {
        while(!vWFile.eof()) {
            int selected;
            vWFile >> selected;
            if(selected < optimizer->getResult().vertWeight.size()) {
                optimizer->getResult().vertWeight[selected] = 100.0;
            }
        }
        vWFile.close();
        
        OptCuts::IglUtils::smoothVertField(optimizer->getResult(),
                                           optimizer->getResult().vertWeight);
        
        std::cout << "OptCuts with regional seam placement" << std::endl;
    }
    //////////////////////////////////////////////////////////////////////////////
    
    if(headlessMode) {
        while(true) {
            preDrawFunc(viewer);
            postDrawFunc(viewer);
        }
    }
    else {
        // Setup viewer and launch
        viewer.core.background_color << 1.0f, 1.0f, 1.0f, 0.0f;
        viewer.callback_key_down = &key_down;
        viewer.callback_pre_draw = &preDrawFunc;
        viewer.callback_post_draw = &postDrawFunc;
        viewer.data().show_lines = true;
        viewer.core.orthographic = true;
        viewer.core.camera_zoom *= 1.9;
        viewer.core.animation_max_fps = 60.0;
        viewer.data().point_size = fracTailSize;
        viewer.data().show_overlay = true;
        updateViewerData();
        viewer.launch();
    }
    
    // Before exit
    logFile.close();
    for(auto& eI : energyTerms) {
        delete eI;
    }
    delete optimizer;
    delete triSoup[0];
}
