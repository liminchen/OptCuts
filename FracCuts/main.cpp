#include "IglUtils.hpp"
#include "TriangleSoup.hpp"
#include "Optimizer.hpp"

#include <igl/boundary_loop.h>
#include <igl/harmonic.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/readOFF.h>
#include <igl/viewer/Viewer.h>

#ifndef TUTORIAL_SHARED_PATH
#define TUTORIAL_SHARED_PATH "/Users/mincli/Documents/libigl/tutorial/shared"
#endif

Eigen::MatrixXd V;
Eigen::MatrixXi F;
Eigen::MatrixXd V_uv_cot;
Eigen::MatrixXd V_uv_uniform;

bool key_down(igl::viewer::Viewer& viewer, unsigned char key, int modifier)
{
    switch (key)
    {
        case '1':
        {
            // Plot the 3D mesh
            viewer.data.set_mesh(V,F);
            viewer.data.set_uv(V_uv_cot);
            viewer.core.align_camera_center(V,F);
            
            // Draw checkerboard texture
            viewer.core.show_texture = true;
            
            break;
        }
            
        case '2':
        {
            // Plot the mesh in 2D using the UV coordinates as vertex coordinates
            viewer.data.set_mesh(V_uv_cot,F);
            viewer.core.align_camera_center(V_uv_cot,F);
            
            viewer.core.show_texture = false;
            
            break;
        }
            
        case '3':
        {
            // Plot the mesh in 2D using the UV coordinates as vertex coordinates
            viewer.data.set_mesh(V, F);
            viewer.data.set_uv(V_uv_uniform);
            viewer.core.align_camera_center(V, F);
            
            viewer.core.show_texture = true;
            
            break;
        }
            
        case '4':
        {
            // Plot the mesh in 2D using the UV coordinates as vertex coordinates
            viewer.data.set_mesh(V_uv_uniform, F);
            viewer.core.align_camera_center(V_uv_cot, F);
            
            viewer.core.show_texture = false;
            
            break;
        }
            
        default:
            break;
    }
    
    viewer.data.compute_normals();
    
    return false;
}

int main(int argc, char *argv[])
{
    // Load a mesh in OFF format
    igl::readOFF(TUTORIAL_SHARED_PATH "/camelhead.off", V, F);
    
    // Find the open boundary
    Eigen::VectorXi bnd;
    igl::boundary_loop(F,bnd);
    
    // Map the boundary to a circle, preserving edge proportions
    Eigen::MatrixXd bnd_uv;
    igl::map_vertices_to_circle(V,bnd,bnd_uv);
    
    // Harmonic parametrization for the internal vertices
    igl::harmonic(V,F,bnd,bnd_uv,1,V_uv_cot);
    
    // Compute graph laplacian
    Eigen::SparseMatrix<double> A, M;
    FracCuts::IglUtils::computeUniformLaplacian(F, A);
    igl::harmonic(A,M,bnd,bnd_uv,1,V_uv_uniform);
    
    // Scale UV to make the texture more clear
    V_uv_cot *= 5;
    V_uv_uniform *= 5;
    
    // Plot the mesh
    igl::viewer::Viewer viewer;
//    FracCuts::TriangleSoup triSoup(V, F);
    viewer.data.set_mesh(V, F);
//    viewer.data.set_mesh(triSoup.V, triSoup.F);
    viewer.data.set_uv(V_uv_cot);
    viewer.callback_key_down = &key_down;
    
    // Enable wireframe
    viewer.core.show_lines = true;
    
    // Draw checkerboard texture
    viewer.core.show_texture = true;
    
    // Launch the viewer
    viewer.launch();
}
