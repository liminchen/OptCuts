#include "Types.hpp"
#include "IglUtils.hpp"
#include "Optimizer.hpp"
#include "SymStretchEnergy.hpp"
#include "ARAPEnergy.hpp"
#include "SeparationEnergy.hpp"
#include "CohesiveEnergy.hpp"
#include "GIF.hpp"

#include "Diagnostic.hpp"
#include "MeshProcessing.hpp"

#include <igl/readOFF.h>
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/harmonic.h>
#include <igl/arap.h>
#include <igl/avg_edge_length.h>
#include <igl/viewer/Viewer.h>
#include <igl/png/writePNG.h>
#include <igl/euler_characteristic.h>

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
bool optimization_on = false;
int iterNum = 0;
int iterNum_lastTopo = 0;
int iterAmt_rollBack = 0;
int iterAmt_rollBack_topo = 0;
int iterNum_backUp = 0;
bool converged = false;
bool autoHomotopy = true;
std::ofstream homoTransFile;
bool fractureMode = false;
//double fracThres = 0.0; //stop according to local estimation TODO: make as prog args
double fracThres = -__DBL_MAX__; //stop according to descent step TODO: make as prog args
bool lastFractureIn = false;
bool altBase = false;
bool outerLoopFinished = false;
double lastE_w = 0.0;
double lastE_SD = 0.0;
double lastE_se = 0.0;
const double upperBound_E_SD = 4.05;
const double convTol_upperBound_E_SD = 1.0e-3; //TODO!!! related to avg edge len?

std::ofstream logFile;
std::string outputFolderPath = "/Users/mincli/Desktop/output_FracCuts/";
const std::string meshFolder = "/Users/mincli/Desktop/meshes/";

// visualization
igl::viewer::Viewer viewer;
const int channel_initial = 0;
const int channel_result = 1;
const int channel_findExtrema = 2;
int viewChannel = channel_result;
bool viewUV = true; // view UV or 3D model
double texScale = 1.0;
bool showSeam = true;
bool showBoundary = false;
int showDistortion = 1; // 0: don't show; 1: SD energy value; 2: L2 stretch value;
bool showTexture = true; // show checkerboard
bool isLighting = false;
bool showFracTail = true;
clock_t ticksPast = 0, ticksPast_frac = 0, lastStart;
double secPast = 0.0;
time_t lastStart_world;
bool offlineMode = false;
bool saveInfo_postDraw = false;
std::string infoName = "";
bool isCapture3D = false;
int capture3DI = 0;
GifWriter GIFWriter;
const uint32_t GIFDelay = 16; //*10ms
double GIFScale = 0.5;


void proceedOptimization(int proceedNum = 1)
{
    for(int proceedI = 0; (proceedI < proceedNum) && (!converged); proceedI++) {
        std::cout << "Iteration" << iterNum << ":" << std::endl;
        lastStart = clock();
        converged = optimizer->solve(1);
        ticksPast += clock() - lastStart;
        iterNum = optimizer->getIterNum();
    }
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
    switch(showDistortion) {
        case 1: { // show SD energy value
            Eigen::VectorXd distortionPerElem;
            energyTerms[0]->getEnergyValPerElem(*triSoup[viewChannel], distortionPerElem, true);
    //        dynamic_cast<FracCuts::SymStretchEnergy*>(energyTerms[0])->getDivGradPerElem(*triSoup[viewChannel], distortionPerElem);
            Eigen::MatrixXd color_distortionVis;
            FracCuts::IglUtils::mapScalarToColor(distortionPerElem, color_distortionVis, 4.0, 6.25);
    //        FracCuts::IglUtils::mapScalarToColor(distortionPerElem, color_distortionVis,
    //            distortionPerElem.minCoeff(), distortionPerElem.maxCoeff());
            viewer.data.set_colors(color_distortionVis);
            break;
        }
            
        case 2: { // show L2 stretch value
            Eigen::VectorXd l2StretchPerElem;
            triSoup[viewChannel]->computeL2StretchPerElem(l2StretchPerElem);
            Eigen::MatrixXd color_distortionVis;
            FracCuts::IglUtils::mapScalarToColor(l2StretchPerElem, color_distortionVis, 1.0, 2.0);
            viewer.data.set_colors(color_distortionVis);
            break;
        }
    
        case 0: {
            viewer.data.set_colors(Eigen::RowVector3d(1.0, 1.0, 0.0));
            break;
        }
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
        viewer.core.align_camera_center(UV_vis, triSoup[viewChannel]->F);
        
        viewer.core.show_texture = false;
        viewer.core.lighting_factor = 0.0;

        updateViewerData_seam(UV_vis);
        
        viewer.data.set_points(Eigen::MatrixXd::Zero(0, 3), Eigen::RowVector3d(0.0, 0.0, 1.0));
        if(showFracTail) {
            for(const auto& tailVI : triSoup[viewChannel]->fracTail) {
                viewer.data.add_points(UV_vis.row(tailVI), Eigen::RowVector3d(0.0, 0.0, 1.0));
            }
        }
    }
    else {
        if((triSoup[viewChannel]->V_rest.rows() != viewer.data.V.rows()) ||
           (UV_vis.rows() != viewer.data.V_uv.rows()) ||
           (triSoup[viewChannel]->F.rows() != viewer.data.F.rows()))
        {
            viewer.data.clear();
        }
        viewer.data.set_mesh(triSoup[viewChannel]->V_rest, triSoup[viewChannel]->F);
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
        
        viewer.data.set_points(Eigen::MatrixXd::Zero(0, 3), Eigen::RowVector3d(0.0, 0.0, 1.0));
        if(showFracTail) {
            for(const auto& tailVI : triSoup[viewChannel]->fracTail) {
                viewer.data.add_points(triSoup[viewChannel]->V_rest.row(tailVI), Eigen::RowVector3d(0.0, 0.0, 1.0));
            }
        }
    }
    updateViewerData_distortion();
    
    viewer.data.compute_normals();
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
    viewer.core.draw_buffer(viewer.data, viewer.opengl, false, R, G, B, A);
    
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
    
    file << iterNum << " " << optimizer->getTopoIter() << " "
        << iterAmt_rollBack << " " << iterAmt_rollBack_topo << std::endl;
    
    file << static_cast<double>(ticksPast) / CLOCKS_PER_SEC << " " <<
        static_cast<double>(ticksPast_frac) / CLOCKS_PER_SEC << " " <<
        secPast << std::endl;
    
    double seamLen;
    if(energyParams[0] == 1.0) {
        // pure distortion minimization mode for models with initial cuts also reflected on the surface as boundary edges...
        triSoup[channel_result]->computeBoundaryLen(seamLen);
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
        std::cout << "Time past: " << static_cast<double>(ticksPast) / CLOCKS_PER_SEC << "s." << std::endl;
        std::cout << "Time for fracture: " << static_cast<double>(ticksPast_frac) / CLOCKS_PER_SEC << "s." << std::endl;
        std::cout << "World Time:\nTime past: " << secPast << "s." << std::endl;
        secPast += difftime(time(NULL), lastStart_world);
    }
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
                saveInfo();
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

bool postDrawFunc(igl::viewer::Viewer& viewer)
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

//void computeCriticalLambda_merge(double& CL_merge)
//{
//    double E_se_old; triSoup[channel_result]->computeSeamSparsity(E_se_old);
//    E_se_old /= triSoup[channel_result]->virtualRadius;
//    const double E_SD_old = optimizer->getLastEnergyVal() / energyParams[0];
//    FracCuts::TriangleSoup triSoup_old = optimizer->getResult();
//    optimizer->createFracture(fracThres, 2, false, false);
//    
//    bool converged_old = converged;
//    converged = false;
//    while(!converged) {
//        proceedOptimization(100);
//    }
//    converged = converged_old;
//    //!!! iter number backup?
//    
//    double E_se; triSoup[channel_result]->computeSeamSparsity(E_se);
//    E_se /= triSoup[channel_result]->virtualRadius;
//    const double E_SD = optimizer->getLastEnergyVal() / energyParams[0];
//    CL_merge = (E_SD - E_SD_old) / (E_se_old - E_se + E_SD - E_SD_old);
//    
//    optimizer->setConfig(triSoup_old);
//}

//void computeNextLargerE_SD(double& nextLargerE_SD)
//{
//    //!!! count time!
//    const double E_SD_old = optimizer->getLastEnergyVal() / energyParams[0];
//    FracCuts::TriangleSoup triSoup_old = optimizer->getResult();
//    
//    optimizer->getResult().mergeEdge(0.0, fracThres);
//    optimizer->updateEnergyData();
//    
//    bool converged_old = converged;
//    converged = false;
//    while(!converged) {
//        proceedOptimization(100);
//    }
//    converged = converged_old;
//    //!!! iter number backup?
//    
//    nextLargerE_SD = optimizer->getLastEnergyVal() / energyParams[0];
//    
//    optimizer->setConfig(triSoup_old);
//}

bool updateLambda_stationaryV(bool cancelMomentum = true, bool checkConvergence = false)
{
    static const double minLambda = 1.0e-3;
    
    const double E_SD = optimizer->getLastEnergyVal() / energyParams[0];
    if(checkConvergence) {
        if(std::abs(upperBound_E_SD - E_SD) < convTol_upperBound_E_SD) {
            logFile << "converged at E_SD = " << E_SD << ", b = " << upperBound_E_SD << std::endl;
            return false;
        }
    }
    if(E_SD < upperBound_E_SD) {
//        if(checkConvergence) {
//            if(upperBound_E_SD - E_SD < 5.0e-3) {
//                double nextLargerE_SD = 0.0;
//                computeNextLargerE_SD(nextLargerE_SD);
//                if(nextLargerE_SD > upperBound_E_SD) {
//                    logFile << "converged at E_SD = " << E_SD << ", b = " << upperBound_E_SD << ", nextLarge = " << nextLargerE_SD << std::endl;
//                    return false;
//                }
//            }
//        }
        
        static bool firstReach = true;
        if(firstReach) {
            firstReach = false;
            if(cancelMomentum) {
                double E_se; triSoup[channel_result]->computeSeamSparsity(E_se);
                E_se /= triSoup[channel_result]->virtualRadius;
                energyParams[0] = (E_se - lastE_se) / (lastE_SD - lastE_se - E_SD + E_se);
                assert(energyParams[0] < 1.0);
                if(energyParams[0] < minLambda) {
                    energyParams[0] = minLambda;
                }
                optimizer->updateEnergyData();

                logFile << "E_SD = " << E_SD << ", b = " << upperBound_E_SD << ", new lambda = " << energyParams[0] << std::endl;
                return true;
            }
        }
    }
    
//        double E_se; triSoup[channel_result]->computeSeamSparsity(E_se);
//    E_se /= triSoup[channel_result]->virtualRadius;
//        const double E_w = optimizer->getLastEnergyVal() +
//            (1.0 - energyParams[0]) * E_se;
    const double kai = 100.0;
    const double kai2 = 1.0;
//        const double kai = 0.5 / (lastE_w - E_w);
    energyParams[0] = kai * (E_SD - upperBound_E_SD) + kai2 * energyParams[0] / (1.0 - energyParams[0]);
    energyParams[0] /= (1.0 + energyParams[0]);
    assert(energyParams[0] < 1.0);
    if(energyParams[0] < minLambda) {
        energyParams[0] = minLambda;
    }
    optimizer->updateEnergyData();
    
    logFile << "E_SD = " << E_SD << ", b = " << upperBound_E_SD << ", new lambda = " << energyParams[0] << std::endl;
    return true;
}

bool preDrawFunc(igl::viewer::Viewer& viewer)
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
                        const double timeUsed = static_cast<double>(ticksPast) / CLOCKS_PER_SEC;
                        std::cout << "optimization converged, with " << timeUsed << "s." << std::endl;
                        logFile << "optimization converged, with " << timeUsed << "s." << std::endl;
                        homoTransFile.close();
                        outerLoopFinished = true;
                    }
                    //!!!TODO: energy and grad file output!
                    // flush
                    // decrease delta will change E_s thus E_w
                    break;
                }
                    
                case FracCuts::MT_GEOMIMG: {
                    if(optimizer->getLastEnergyVal() / energyParams[0] - upperBound_E_SD
                       < convTol_upperBound_E_SD)
                    {
                        logFile << "E_SD reaches user specified upperbound " << upperBound_E_SD << std::endl;
                        
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
                        const double timeUsed = static_cast<double>(ticksPast) / CLOCKS_PER_SEC;
                        const double timeUsed_frac = static_cast<double>(ticksPast_frac) / CLOCKS_PER_SEC;
                        std::cout << "optimization converged, with " << timeUsed << "s, where " <<
                        timeUsed_frac << "s is for fracture computation." << std::endl;
                        logFile << "optimization converged, with " << timeUsed << "s, where " <<
                        timeUsed_frac << "s is for fracture computation." << std::endl;
                        homoTransFile.close();
                        outerLoopFinished = true;
                    }
                    else {
                        infoName = std::to_string(iterNum);
                        
                        // continue to make geometry image cuts
                        homoTransFile << iterNum << std::endl;
                        lastStart = clock();
                        assert(optimizer->createFracture(fracThres, false, false));
                        ticksPast += clock() - lastStart;
                        converged = false;
                    }
                    optimizer->flushEnergyFileOutput();
                    optimizer->flushGradFileOutput();
                    break;
                }
                    
                case FracCuts::MT_OURS: {
                    infoName = std::to_string(iterNum);
                    if(optimizer->getLastEnergyVal() / energyParams[0] - upperBound_E_SD
                       < convTol_upperBound_E_SD)
                    {
                        // save info once bound is reached for comparison
                        static bool saved = false;
                        if(!saved) {
                            saveScreenshot(outputFolderPath + "firstFeasible.png", 0.5, false, true);
                            //                            triSoup[channel_result]->save(outputFolderPath + infoName + "_triSoup.obj");
                            triSoup[channel_result]->saveAsMesh(outputFolderPath + "firstFeasible_mesh.obj");
                            saveInfoForPresent("info_firstFeasible.txt");
                            saved = true;
                        }
                    }
                    
                    double E_se; triSoup[channel_result]->computeSeamSparsity(E_se);
                    E_se /= triSoup[channel_result]->virtualRadius;
                    const double E_w = optimizer->getLastEnergyVal() +
                        (1.0 - energyParams[0]) * E_se;
                    std::cout << "E_w from " << lastE_w << " to " << E_w << std::endl;
                    
                    if(E_w > lastE_w) {
                        assert(fracThres < 0.0);
                        
                        // roll back
                        optimizer->clearEnergyFileOutputBuffer();
                        optimizer->clearGradFileOutputBuffer();
                        optimizer->setConfig(triSoup_backup, iterNum_backUp, optimizer->getTopoIter() - 1);
                        iterAmt_rollBack += iterNum - iterNum_backUp;
                        iterNum = iterNum_backUp;
                        iterAmt_rollBack_topo++;
                        
                        if(lastFractureIn) {
                            if(!updateLambda_stationaryV(false, true)) {
                                // if the last topology operation is interior split
                                logFile << "E_w is not decreased, end process." << std::endl;
                                
                                infoName = "finalResult";
                                // perform exact solve
                                //                            optimizer->setRelGL2Tol(1.0e-10);
                                optimizer->setAllowEDecRelTol(false);
                                //!! can recompute precondmtr if needed
                                converged = false;
                                while(!converged) {
                                    proceedOptimization(1000);
                                }
                                secPast += difftime(time(NULL), lastStart_world);
                                updateViewerData();
                                
                                optimizer->flushEnergyFileOutput();
                                optimizer->flushGradFileOutput();
                                
                                optimization_on = false;
                                viewer.core.is_animating = false;
                                const double timeUsed = static_cast<double>(ticksPast) / CLOCKS_PER_SEC;
                                const double timeUsed_frac = static_cast<double>(ticksPast_frac) / CLOCKS_PER_SEC;
                                std::cout << "optimization converged, with " << timeUsed << "s, where " <<
                                timeUsed_frac << "s is for fracture computation." << std::endl;
                                logFile << "optimization converged, with " << timeUsed << "s, where " <<
                                timeUsed_frac << "s is for fracture computation." << std::endl;
                                homoTransFile.close();
                                outerLoopFinished = true;
                            }
                            else {
                                // continue to split boundary after lambda update
                                saveInfo_postDraw = false;
                                
                                triSoup[channel_result]->computeSeamSparsity(lastE_se);
                                lastE_se /= triSoup[channel_result]->virtualRadius;
                                lastE_SD = optimizer->getLastEnergyVal() / energyParams[0];
                                lastE_w = optimizer->getLastEnergyVal() +
                                    (1.0 - energyParams[0]) * lastE_se;
                                
                                iterNum_lastTopo = iterNum;
                                lastStart = clock();
                                if(optimizer->createFracture(fracThres, false, !altBase)) {
                                    lastFractureIn = false;
                                    ticksPast += clock() - lastStart;
                                    converged = false;
                                }
                                else {
                                    // won't happen now since we are splitting anyway
                                    assert(0);
                                }
                            }
                        }
                        else {
                            // last topology operation is boundary split,
                            // roll back and try interior split
                            saveInfo_postDraw = false;
                            iterNum_lastTopo = iterNum;
                            lastStart = clock();
                            optimizer->createFracture(fracThres, false, !altBase, true);
                            lastFractureIn = true;
                            ticksPast += clock() - lastStart;
                            converged = false;
                        }
                    }
                    else {
                        optimizer->flushEnergyFileOutput();
                        optimizer->flushGradFileOutput();
                        homoTransFile << iterNum_lastTopo << std::endl;
                        
                        // continue to split boundary
                        updateLambda_stationaryV();
                        
                        triSoup_backup = optimizer->getResult();
                        iterNum_backUp = iterNum;
                        triSoup[channel_result]->computeSeamSparsity(lastE_se);
                        lastE_se /= triSoup[channel_result]->virtualRadius;
                        lastE_SD = optimizer->getLastEnergyVal() / energyParams[0];
                        lastE_w = optimizer->getLastEnergyVal() +
                            (1.0 - energyParams[0]) * lastE_se;
                        
                        iterNum_lastTopo = iterNum;
                        lastStart = clock();
                        if(optimizer->createFracture(fracThres, false, !altBase)) {
                            lastFractureIn = false;
                            ticksPast += clock() - lastStart;
                            converged = false;
                        }
                        else {
                            // won't happen now since we are splitting anyway
                            assert(0);
                        }
                    }
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
        case FracCuts::MT_OURS:
            assert(lambda < 1.0);
            startDS = "Ours";
            break;
            
        case FracCuts::MT_GEOMIMG:
            assert(lambda < 1.0);
            startDS = "GeomImg";
            break;
            
        case FracCuts::MT_AUTOCUTS:
            assert(lambda > 0.0);
            assert(delta > 0.0);
            startDS = "AutoCuts";
            break;
            
        default:
            //!!!TODO: pure SD
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
        // with input UV
        triSoup.emplace_back(new FracCuts::TriangleSoup(V, F, UV, FUV, startWithTriSoup));
        outputFolderPath += meshName + "_input_" + FracCuts::IglUtils::rtos(lambda) + "_" + FracCuts::IglUtils::rtos(delta) +
            "_" +startDS + folderTail;
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
    //            temp->farthestPointCut(); // open up a boundary for Tutte embedding
//                temp->highCurvOnePointCut();
                temp->onePointCut();
                
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
    
    // * Our approach
    texScale = 10.0 / (triSoup[0]->bbox.row(1) - triSoup[0]->bbox.row(0)).maxCoeff();
    if(lambda != 1.0) {
        energyParams.emplace_back(1.0 - lambda);
    //    energyTerms.emplace_back(new FracCuts::ARAPEnergy());
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
    optimizer = new FracCuts::Optimizer(*triSoup[0], energyTerms, energyParams, false);
    lastStart = clock();
    optimizer->precompute();
    ticksPast += clock() - lastStart;
    triSoup.emplace_back(&optimizer->getResult());
    triSoup_backup = optimizer->getResult();
    triSoup.emplace_back(&optimizer->getData_findExtrema()); // for visualizing UV map for finding extrema
    if((lambda > 0.0) && (!startWithTriSoup)) {
        //!!!TODO: put into switch(methodType)
        // fracture mode
        fractureMode = true;
        
        triSoup[channel_result]->computeSeamSparsity(lastE_se);
        lastE_se /= triSoup[channel_result]->virtualRadius;
        lastE_SD = optimizer->getLastEnergyVal() / energyParams[0];
        lastE_w = optimizer->getLastEnergyVal() + (1.0 - energyParams[0]) * lastE_se;
        
        if(delta == 0.0) {
            altBase = true;
        }
    }
    
    // Setup viewer and launch
    viewer.core.background_color << 1.0f, 1.0f, 1.0f, 0.0f;
    viewer.callback_key_down = &key_down;
    viewer.callback_pre_draw = &preDrawFunc;
    viewer.callback_post_draw = &postDrawFunc;
    viewer.core.show_lines = true;
    viewer.core.orthographic = true;
    viewer.core.camera_zoom *= 1.9;
    viewer.core.animation_max_fps = 60.0;
    viewer.core.point_size = 16.0f;
    viewer.core.show_overlay = true;
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
