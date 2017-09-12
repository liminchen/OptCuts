#include "IglUtils.hpp"
#include "Optimizer.hpp"
#include "SymStretchEnergy.hpp"
#include "ARAPEnergy.hpp"
#include "SeparationEnergy.hpp"

#include <igl/readOFF.h>
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/harmonic.h>
#include <igl/arap.h>
#include <igl/avg_edge_length.h>
#include <igl/viewer/Viewer.h>
#include <igl/png/writePNG.h>

#include <fstream>
#include <string>

#ifndef TUTORIAL_SHARED_PATH
#define TUTORIAL_SHARED_PATH "/Users/mincli/Documents/libigl/tutorial/shared"
#endif


// mesh
std::vector<Eigen::MatrixXd> V;
std::vector<Eigen::MatrixXi> F;
std::vector<Eigen::MatrixXd> UV;
std::vector<Eigen::MatrixXi> E; //!! change to triSoup

// optimization
FracCuts::TriangleSoup triSoup;
FracCuts::Optimizer* optimizer;
bool optimization_on = false;
int iterNum = 0;
std::vector<FracCuts::Energy*> energyTerms;
std::vector<double> energyParams;
bool converged = false;

std::ofstream logFile;
std::string outputFolderPath = "/Users/mincli/Desktop/output_FracCuts/";

// visualization
igl::viewer::Viewer viewer;
int viewChannel = 0;
const int channel_initial = 0;
const int channel_result = 1;
bool viewUV = false;
const double texScale = 20.0;
bool showSeam = false;


void proceedOptimization(void)
{
    std::cout << "Iteration" << iterNum << ":" << std::endl;
    converged = optimizer->solve(1);
    UV[channel_result] = optimizer->getResult().V * texScale;
    iterNum++;
}

void updateViewerData(void)
{
    if(viewUV) {
        if((UV[viewChannel].rows() != viewer.data.V.rows()) ||
           (F[viewChannel].rows() != viewer.data.F.rows()))
        {
            viewer.data.clear();
        }
        viewer.data.set_mesh(UV[viewChannel], F[viewChannel]);
//        viewer.core.align_camera_center(UV[0], F[0]);
        viewer.core.align_camera_center(UV[viewChannel], F[viewChannel]);
        viewer.core.show_texture = false;
        viewer.core.lighting_factor = 0.0;
        
        if(showSeam) {
            viewer.core.show_lines = false;
            Eigen::MatrixXd UV_3D(UV[viewChannel].rows(), 3);
            UV_3D << UV[viewChannel], Eigen::VectorXd::Zero(UV[viewChannel].rows());
            Eigen::VectorXd seamScore;
            optimizer->getResult().computeSeamScore(seamScore);
            Eigen::MatrixXd color;
            FracCuts::IglUtils::mapScalarToColor(seamScore, color, 1.0);
            viewer.data.set_edges(Eigen::MatrixXd(0, 3), Eigen::MatrixXi(0, 2), Eigen::RowVector3d(0.0, 0.0, 0.0));
            for(int eI = 0; eI < E[viewChannel].rows(); eI++) {
                if(seamScore[eI] > 1e-1) {
                    viewer.data.add_edges(UV_3D.row(E[viewChannel].row(eI)[0]),
                        UV_3D.row(E[viewChannel].row(eI)[1]), color.row(eI));
                    if(E[viewChannel].row(eI)[2] >= 0) {
                        viewer.data.add_edges(UV_3D.row(E[viewChannel].row(eI)[2]),
                            UV_3D.row(E[viewChannel].row(eI)[3]), color.row(eI));
                    }
                }
            }
        }
        else {
            viewer.core.show_lines = true;
            viewer.data.set_edges(Eigen::MatrixXd(0, 3), Eigen::MatrixXi(0, 2), Eigen::RowVector3d(0.0, 0.0, 0.0));
        }
    }
    else {
        if((V[viewChannel].rows() != viewer.data.V.rows()) ||
           (UV[viewChannel].rows() != viewer.data.V_uv.rows()) ||
           (F[viewChannel].rows() != viewer.data.F.rows()))
        {
            viewer.data.clear();
        }
        viewer.data.set_mesh(V[viewChannel], F[viewChannel]);
        viewer.data.set_uv(UV[viewChannel]);
//        viewer.core.align_camera_center(V[0], F[0]);
        viewer.core.align_camera_center(V[viewChannel], F[viewChannel]);
        viewer.core.show_texture = true;
        viewer.core.lighting_factor = 1.0;
        
        if(showSeam) {
            viewer.core.show_lines = false;
            Eigen::VectorXd seamScore;
            optimizer->getResult().computeSeamScore(seamScore);
            Eigen::MatrixXd color;
            FracCuts::IglUtils::mapScalarToColor(seamScore, color, 1.0);
            viewer.data.set_edges(Eigen::MatrixXd(0, 3), Eigen::MatrixXi(0, 2), Eigen::RowVector3d(0.0, 0.0, 0.0));
            for(int eI = 0; eI < E[viewChannel].rows(); eI++) {
                if(seamScore[eI] > 1e-1) {
                    viewer.data.add_edges(V[viewChannel].row(E[viewChannel].row(eI)[0]),
                        V[viewChannel].row(E[viewChannel].row(eI)[1]), color.row(eI));
                    if(E[viewChannel].row(eI)[2] >= 0) {
                        viewer.data.add_edges(V[viewChannel].row(E[viewChannel].row(eI)[2]),
                            V[viewChannel].row(E[viewChannel].row(eI)[3]), color.row(eI));
                    }
                }
            }
        }
        else {
            viewer.core.show_lines = true;
            viewer.data.set_edges(Eigen::MatrixXd(0, 3), Eigen::MatrixXi(0, 2), Eigen::RowVector3d(0.0, 0.0, 0.0));
        }
    }
    
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
        if((changeToChannel < V.size()) && (viewChannel != changeToChannel)) {
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
                
            case 'h':
            case 'H': { // mannual homotopy optimization
                dynamic_cast<FracCuts::SeparationEnergy*>(energyTerms.back())->decreaseSigma();
                optimizer->computeLastEnergyVal();
                converged = false;
                break;
            }
                
            case 'o':
            case 'O': {
                saveScreenshot(outputFolderPath + std::to_string(iterNum) + ".png", 0.5);
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
//        saveScreenshot(outputFolderPath + std::to_string(iterNum) + ".png", 0.5);
        if(converged) {
            optimization_on = false;
            viewer.core.is_animating = false;
            std::cout << "optimization converged." << std::endl;
        }
    }
    return false;
}

int main(int argc, char *argv[])
{
    if(argc > 1) {
        outputFolderPath += argv[1];
        if(outputFolderPath.back() != '/') {
            outputFolderPath += '/';
        }
    }
    logFile.open(outputFolderPath + "log.txt");
    
    // Load a mesh in OFF format
    V.resize(V.size() + 1);
    F.resize(F.size() + 1);
    E.resize(E.size() + 1);
    igl::readOFF(TUTORIAL_SHARED_PATH "/camelhead_f1000.off", V[0], F[0]);
//    FracCuts::TriangleSoup squareMesh(FracCuts::Primitive::P_SQUARE, 1.0, 0.1, false);
//    V[0] = squareMesh.V_rest;
//    F[0] = squareMesh.F;
    const double edgeLen = igl::avg_edge_length(V[0], F[0]);
    
    // Harmonic map
    // Find the open boundary
    Eigen::VectorXi bnd;
    igl::boundary_loop(F[0], bnd);
    
    // Map the boundary to a circle, preserving edge proportions
    Eigen::MatrixXd bnd_uv;
    igl::map_vertices_to_circle(V[0], bnd, bnd_uv);
    
//    // * Harmonic parametrization for the internal vertices
//    UV.resize(UV.size() + 1);
//    igl::harmonic(V[0], F[0], bnd, bnd_uv, 1, UV[0]);
    
    // * Harmonic map with uniform weights
    Eigen::SparseMatrix<double> A, M;
    FracCuts::IglUtils::computeUniformLaplacian(F[0], A);
    UV.resize(UV.size() + 1);
    igl::harmonic(A, M, bnd, bnd_uv, 1, UV[0]);
    
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
    triSoup = FracCuts::TriangleSoup(V[0], F[0], UV[0]);
    energyParams.emplace_back(1.0e0);
//    energyTerms.emplace_back(new FracCuts::ARAPEnergy());
    energyTerms.emplace_back(new FracCuts::SymStretchEnergy());
    energyParams.emplace_back(2.5e-1);
    energyTerms.emplace_back(new FracCuts::SeparationEnergy(edgeLen * edgeLen, 8.0));
//    energyTerms.back()->checkEnergyVal(triSoup);
//    energyTerms.back()->checkGradient(triSoup);
//    energyTerms.back()->checkHessian(triSoup);
    optimizer = new FracCuts::Optimizer(triSoup, energyTerms, energyParams);
    optimizer->precompute();
    
    
    // Scale UV to make the texture more clear
    UV[0] *= texScale;
    V.emplace_back(triSoup.V_rest);
    F.emplace_back(triSoup.F);
    UV.emplace_back(triSoup.V * texScale);
    E.emplace_back(triSoup.cohE);
    
    // setup viewer
    viewer.core.background_color << 1.0f, 1.0f, 1.0f, 0.0f;
    viewer.callback_key_down = &key_down;
    viewer.callback_pre_draw = &preDrawFunc;
    viewer.core.show_lines = true;
    viewer.core.orthographic = true;
    updateViewerData();
    
//    Eigen::VectorXd distortionPerElem;
//    energyTerms[0]->getEnergyValPerElem(triSoup, distortionPerElem);
//    FracCuts::IglUtils::mapScalarToColor(distortionPerElem, color_distortionVis);
//    viewer.data.set_face_based(true);
//    viewer.data.set_colors(color_distortionVis);
//    viewer.core.lighting_factor = 0.0;
    
    // Launch the viewer
    viewer.launch();
    
    
    for(auto& eI : energyTerms) {
        delete eI;
    }
    delete optimizer;
    logFile.close();
}
