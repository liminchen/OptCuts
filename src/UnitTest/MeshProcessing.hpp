//
//  MeshProcessing.hpp
//  OptCuts
//
//  Created by Minchen Li on 1/31/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#ifndef MeshProcessing_hpp
#define MeshProcessing_hpp

#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/harmonic.h>
#include <igl/euler_characteristic.h>
#include <igl/per_vertex_normals.h>

#include <cstdio>

extern std::string outputFolderPath;

namespace OptCuts {
    class MeshProcessing
    {
    public:
        static void run(int argc, char *argv[])
        {
            if(argc > 2) {
                Eigen::MatrixXd V, UV, N;
                Eigen::MatrixXi F, FUV, FN;
                std::string meshPath = std::string(argv[2]);
                std::string meshFileName = meshPath.substr(meshPath.find_last_of('/') + 1);
                std::string meshName = meshFileName.substr(0, meshFileName.find_last_of('.'));
                const std::string suffix = meshFileName.substr(meshFileName.find_last_of('.'));
                if(suffix == ".off") {
                    igl::readOFF(meshPath, V, F);
                }
                else if(suffix == ".obj") {
                    igl::readOBJ(meshPath, V, UV, N, F, FUV, FN);
                }
                else {
                    std::cout << "unkown mesh file format!" << std::endl;
                    return;
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
                            igl::writeOBJ(outputFolderPath + meshName + "_processed.obj", V, F);
                            break;
                        }
                            
                        case 1: {
                            // turn a triangle soup into a mesh
                            if(V.rows() != F.rows() * 3) {
                                std::cout << "Input model is not a triangle soup!" << std::endl;
                                return;
                            }
                            
                            if(UV.rows() != V.rows()) {
                                std::cout << "UV coordinates not valid, will generate separate rigid mapping UV!" << std::endl;
                            }
                            
                            OptCuts::TriMesh inputTriSoup(V, F, UV, Eigen::MatrixXi(), false);
                            if(argc > 4) {
                                // input original model to get cohesive edge information
                                Eigen::MatrixXd V0;
                                Eigen::MatrixXi F0;
                                std::string meshPath = std::string(argv[4]);
                                const std::string suffix = meshPath.substr(meshPath.find_last_of('.'));
                                if(suffix == ".off") {
                                    igl::readOFF(meshPath, V0, F0);
                                }
                                else if(suffix == ".obj") {
                                    igl::readOBJ(meshPath, V0, F0);
                                }
                                else {
                                    std::cout << "unkown mesh file format!" << std::endl;
                                    return;
                                }
                                
                                inputTriSoup.cohE = OptCuts::TriMesh(V0, F0, Eigen::MatrixXd()).cohE;
                                inputTriSoup.computeFeatures();
                            }
                            inputTriSoup.saveAsMesh(outputFolderPath + meshName + "_processed.obj", true);
                            break;
                        }
                            
                        case 2: {
                            // save texture as mesh
                            if(UV.rows() == 0) {
                                // no input UV
                                std::cout << "compute harmonic UV map" << std::endl;
                                Eigen::VectorXi bnd;
                                igl::boundary_loop(F, bnd); // Find the open boundary
                                if(bnd.size()) {
                                    std::cout << "disk-topology surface" << std::endl;
                                    FUV.resize(0, 3);
                                    
                                    //TODO: what if it has multiple boundaries? or multi-components?
                                    // Map the boundary to a circle, preserving edge proportions
                                    Eigen::MatrixXd bnd_uv;
                                    //            igl::map_vertices_to_circle(V, bnd, bnd_uv);
                                    OptCuts::IglUtils::map_vertices_to_circle(V, bnd, bnd_uv);
                                    
                                    //            // Harmonic parametrization
                                    //            igl::harmonic(V, F, bnd, bnd_uv, 1, UV);
                                    
                                    // Harmonic map with uniform weights
                                    Eigen::SparseMatrix<double> A, M;
                                    OptCuts::IglUtils::computeUniformLaplacian(F, A);
                                    igl::harmonic(A, M, bnd, bnd_uv, 1, UV);
                                    //            OptCuts::IglUtils::computeMVCMtr(V, F, A);
                                    //            OptCuts::IglUtils::fixedBoundaryParam_MVC(A, bnd, bnd_uv, UV);
                                }
                                else {
                                    // closed surface
                                    std::cout << "closed surface" << std::endl;
                                    if(igl::euler_characteristic(V, F) != 2) {
                                        std::cout << "Input surface genus > 0 or has multiple connected components!" << std::endl;
                                        exit(-1);
                                    }
                                    
                                    OptCuts::TriMesh *temp = new OptCuts::TriMesh(V, F, Eigen::MatrixXd(), Eigen::MatrixXi(), false);
                                    //            temp->farthestPointCut(); // open up a boundary for Tutte embedding
                                    //                temp->highCurvOnePointCut();
                                    temp->onePointCut();
                                    FUV = temp->F;
                                    
                                    igl::boundary_loop(temp->F, bnd);
                                    assert(bnd.size());
                                    Eigen::MatrixXd bnd_uv;
                                    OptCuts::IglUtils::map_vertices_to_circle(temp->V_rest, bnd, bnd_uv);
                                    Eigen::SparseMatrix<double> A, M;
                                    OptCuts::IglUtils::computeUniformLaplacian(temp->F, A);
                                    igl::harmonic(A, M, bnd, bnd_uv, 1, UV);
                                    
                                    delete temp;
                                }
                            }
                            else {
                                std::cout << "use input UV" << std::endl;
                            }
                            
                            Eigen::MatrixXd V_uv;
                            V_uv.resize(UV.rows(), 3);
                            V_uv << UV, Eigen::VectorXd::Zero(UV.rows(), 1);
                            if(FUV.rows() == 0) {
                                assert(F.rows() > 0);
                                std::cout << "output with F" << std::endl;
                                igl::writeOBJ(outputFolderPath + meshName + "_UV.obj", V_uv, F);
                            }
                            else {
                                std::cout << "output with FUV" << std::endl;
                                igl::writeOBJ(outputFolderPath + meshName + "_UV.obj", V_uv, FUV);
                            }
                            
                            std::cout << "texture saved as mesh into " << outputFolderPath << meshName << "_UV.obj" << std::endl;
                            
                            break;
                        }
                            
                        case 3: {
                            // map texture to 0-1 and output UV boundary path in both 2D and 3D
                            
                            if(UV.rows() == 0) {
                                std::cout << "no input UV" << std::endl;
                                break;
                            }
                            
                            double minUV_x = UV.col(0).minCoeff(), minUV_y = UV.col(1).minCoeff();
                            double maxUV_x = UV.col(0).maxCoeff(), maxUV_y = UV.col(1).maxCoeff();
                            double divider = 0.0;
                            for(int triI = 0; triI < F.rows(); triI++) {
                                const Eigen::Vector3i& triVInd = F.row(triI);
                                const Eigen::Vector3d e01 = V.row(triVInd[1]) - V.row(triVInd[0]);
                                const Eigen::Vector3d e02 = V.row(triVInd[2]) - V.row(triVInd[0]);
                                divider += 0.5 * e01.cross(e02).norm();
                            }
                            divider = std::sqrt(divider);
                            for(int uvI = 0; uvI < UV.rows(); uvI++) {
                                UV(uvI, 0) = (UV(uvI, 0) - minUV_x) / divider;
                                UV(uvI, 1) = (UV(uvI, 1) - minUV_y) / divider;
                            }
                            
                            if(N.rows() == 0) {
                                igl::per_vertex_normals(V, F, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA, N);
                                FN = F;
                            }
                            
                            igl::writeOBJ(outputFolderPath + meshName + "_with01UV.obj", V, F, N, FN, UV, FUV);
                            std::cout << "texture mapped to [0,1]^2 and saved into " << outputFolderPath << meshName << "_01UV.obj" << std::endl;
                            
                            Eigen::MatrixXd V_UV;
                            if(meshPath.find("AutoCuts") != std::string::npos) {
                                V_UV = UV;
                                UV.conservativeResize(UV.rows(), 2);
                            }
                            else {
                                V_UV.resize(UV.rows(), 3);
                                V_UV << UV, Eigen::VectorXd::Zero(UV.rows());
                            }
                            igl::writeOBJ(outputFolderPath + meshName + "_01UV.obj", V_UV, FUV,
                                          Eigen::MatrixXd(), Eigen::MatrixXi(), UV, FUV);
                            
                            
                            OptCuts::TriMesh temp(V, F, UV, FUV, false);
                            
                            std::vector<std::vector<int>> bnd_all;
                            igl::boundary_loop(temp.F, bnd_all);
                            
                            FILE *out = fopen((outputFolderPath + meshName + "_with01UV.sp").c_str(), "w");
                            assert(out);
                            FILE *out_UV = fopen((outputFolderPath + meshName + "_01UV.sp").c_str(), "w");
                            assert(out_UV);
                            
                            fprintf(out, "%lu\n", bnd_all.size());
                            fprintf(out_UV, "%lu\n", bnd_all.size());
                            for(const auto& bndI : bnd_all) {
                                fprintf(out, "%lu\n", bndI.size());
                                fprintf(out_UV, "%lu\n", bndI.size());
                                for(const auto& i : bndI) {
                                    const Eigen::RowVector3d& v = temp.V_rest.row(i);
                                    fprintf(out, "%le %le %le\n", v[0], v[1], v[2]);
                                    const Eigen::RowVector2d& uv = temp.V.row(i);
                                    fprintf(out_UV, "%le %le 0.0\n", uv[0], uv[1]);
                                }
                            }
                            
                            fclose(out);
                            fclose(out_UV);
                            
                            break;
                        }
                            
                        case 4: {
                            // output as seamster format
                            assert(F.rows() > 0);
                            IglUtils::saveMesh_Seamster(outputFolderPath + meshName + ".seamster",
                                                        V, F);
                            
                            std::cout << "mesh saved into " << outputFolderPath << meshName << ".seamster" << std::endl;
                            break;
                        }
                            
                        case 5: {
                            // merge closed surface mesh file and UV file
                            Eigen::MatrixXd V_UV;
                            Eigen::MatrixXi F_UV;
                            igl::readOBJ(meshPath.substr(0, meshPath.find("_closed.obj")) + ".obj",
                                                         V_UV, F_UV);
                            Eigen::VectorXi bnd;
                            igl::boundary_loop(F_UV, bnd); // Find the open boundary
                            assert(bnd.size());
                            Eigen::MatrixXd bnd_uv;
                            //            igl::map_vertices_to_circle(V, bnd, bnd_uv);
                            OptCuts::IglUtils::map_vertices_to_circle(V_UV, bnd, bnd_uv);
                            
                            //            // Harmonic parametrization
                            //            igl::harmonic(V, F, bnd, bnd_uv, 1, UV);
                            
                            // Harmonic map with uniform weights
                            Eigen::SparseMatrix<double> A, M;
                            OptCuts::IglUtils::computeUniformLaplacian(F_UV, A);
                            igl::harmonic(A, M, bnd, bnd_uv, 1, V_UV);
                            //            OptCuts::IglUtils::computeMVCMtr(V, F, A);
                            //            OptCuts::IglUtils::fixedBoundaryParam_MVC(A, bnd, bnd_uv, UV);
                            
                            igl::writeOBJ(meshPath.substr(0, meshPath.find("_closed.obj")) +
                                          "_withSUV.obj", V, F, N, FN, V_UV, F_UV);
                            std::cout << "mesh saved in " << meshPath.substr(0, meshPath.find("_closed.obj")) + "_withSUV.obj" << std::endl;
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
        }
    };
}

#endif /* MeshProcessing_hpp */
