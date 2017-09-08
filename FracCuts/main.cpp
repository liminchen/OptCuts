#include "IglUtils.hpp"
#include "Optimizer.hpp"
#include "SymStretchEnergy.hpp"
#include "ARAPEnergy.hpp"

#include <igl/readOFF.h>
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/harmonic.h>
#include <igl/arap.h>
#include <igl/viewer/Viewer.h>

#include <fstream>
#include <string>

#ifndef TUTORIAL_SHARED_PATH
#define TUTORIAL_SHARED_PATH "/Users/mincli/Documents/libigl/tutorial/shared"
#endif


// mesh
std::vector<Eigen::MatrixXd> V;
std::vector<Eigen::MatrixXi> F;
std::vector<Eigen::MatrixXd> UV;

// optimization
FracCuts::TriangleSoup triSoup;
FracCuts::Optimizer* optimizer;
bool optimization_on = false;
int iterNum = 0;

std::ofstream logFile;
std::string outputFolderPath = "/Users/mincli/Desktop/";

// visualization
igl::viewer::Viewer viewer;
int viewChannel = 0;
const int channel_initial = 0;
const int channel_result = 1;
bool viewUV = false;
const double texScale = 20.0;


void proceedOptimization(void)
{
    std::cout << "Iteration" << iterNum << ":" << std::endl;
    UV[channel_result] = optimizer->solve(1).V * texScale;
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
        viewer.core.align_camera_center(UV[0], F[0]);
        viewer.core.show_texture = false;
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
        viewer.core.align_camera_center(V[0], F[0]);
        viewer.core.show_texture = true;
    }
    
    viewer.data.compute_normals();
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
                    std::cout << "start/resume optimization, press again to pause." << std::endl;
                    viewer.core.is_animating = true;
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
    }
    return false;
}

int main(int argc, char *argv[])
{
    logFile.open(outputFolderPath + "log.txt");
    
    // Load a mesh in OFF format
    V.resize(V.size() + 1);
    F.resize(F.size() + 1);
    igl::readOFF(TUTORIAL_SHARED_PATH "/camelhead.off", V[0], F[0]);
//    FracCuts::TriangleSoup squareMesh(FracCuts::Primitive::P_SQUARE);
//    V[0] = squareMesh.V_rest;
//    F[0] = squareMesh.F;
    
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
    std::vector<FracCuts::Energy*> energyTerms;
    std::vector<double> energyParams;
    energyParams.emplace_back(1.0);
//    energyTerms.emplace_back(new FracCuts::ARAPEnergy());
    energyTerms.emplace_back(new FracCuts::SymStretchEnergy());
//    energyTerms.back()->checkEnergyVal(triSoup);
//    energyTerms.back()->checkGradient(triSoup);
    optimizer = new FracCuts::Optimizer(triSoup, energyTerms, energyParams);
    optimizer->precompute();
    
    
    // Scale UV to make the texture more clear
    UV[0] *= texScale;
    V.emplace_back(V[0]);
    F.emplace_back(F[0]);
    UV.emplace_back(UV[0]);
    
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
