//
//  MeshProcessing.hpp
//  FracCuts
//
//  Created by Minchen Li on 1/31/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#ifndef MeshProcessing_hpp
#define MeshProcessing_hpp

#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>

#include <cstdio>

extern const std::string meshFolder;

namespace FracCuts {
    class MeshProcessing
    {
    public:
        static void run(int argc, char *argv[])
        {
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
                            igl::writeOBJ(meshFolder + "processedMesh.obj", V, F);
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
                                    return;
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
        }
    };
}

#endif /* MeshProcessing_hpp */
