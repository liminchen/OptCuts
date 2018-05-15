//
//  Diagnostic.hpp
//  FracCuts
//
//  Created by Minchen Li on 1/31/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#ifndef Diagnostic_hpp
#define Diagnostic_hpp

#include "TriangleSoup.hpp"

#include <igl/readOBJ.h>
#include <igl/opengl/glfw/Viewer.h>

#include <cstdio>

extern igl::opengl::glfw::Viewer viewer;
extern bool viewUV;
extern bool showTexture;
extern int showDistortion;
extern double texScale;
extern std::vector<const FracCuts::TriangleSoup*> triSoup;
extern std::vector<FracCuts::Energy*> energyTerms;
extern void updateViewerData(void);
extern bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier);
extern bool preDrawFunc(igl::opengl::glfw::Viewer& viewer);
extern bool postDrawFunc(igl::opengl::glfw::Viewer& viewer);
extern void saveScreenshot(const std::string& filePath, double scale, bool writeGIF, bool writePNG);

namespace FracCuts{
    class Diagnostic
    {
    public:
        static void run(int argc, char *argv[])
        {
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
                        
                    case 1: {
                        // compute and output metric for joint optimization of distortion and seams
                        const std::string resultsFolderPath(argv[3]);
                        FILE *dirList = fopen((resultsFolderPath + "/folderList.txt").c_str(), "r");
                        assert(dirList);
                        FILE *out = fopen((resultsFolderPath + "/stats.txt").c_str(), "w");
                        assert(out);
                                          
                        char buf[BUFSIZ];
                        while((!feof(dirList)) && fscanf(dirList, "%s", buf)) {
                            FILE *in = fopen((resultsFolderPath + '/' + std::string(buf) + "/info.txt").c_str(), "r");
                            assert(in);
                            char line[BUFSIZ];
                            for(int i = 0; i < 3; i++) {
                                fgets(line, BUFSIZ, in);
                            }
                            double bypass, time;
                            sscanf(line, "%le %le %le", &bypass, &bypass, &time);
                            fgets(line, BUFSIZ, in);
                            double seamLen, l2Stretch, E_SD;
                            sscanf(line, "%le %le", &E_SD, &seamLen);
                            fgets(line, BUFSIZ, in);
                            sscanf(line, "%le", &l2Stretch);
                            fclose(in);
                            
                            std::string meshPath(resultsFolderPath + '/' + std::string(buf) + "/finalResult_mesh.obj");
                            Eigen::MatrixXd V, UV, N;
                            Eigen::MatrixXi F, FUV, FN;
                            igl::readOBJ(meshPath, V, UV, N, F, FUV, FN);
                            TriangleSoup resultMesh(V, F, UV, FUV, false, 0.0);
                            TriangleSoup originalMesh(V, F, Eigen::MatrixXd(), Eigen::MatrixXi(), false, 0.0);
                            double absGaussianCurve, absGaussianCurve_original;
                            resultMesh.computeAbsGaussianCurv(absGaussianCurve);
                            originalMesh.computeAbsGaussianCurv(absGaussianCurve_original);
                            
                            fprintf(out, "%6lf %6lf %6lf %.6lf %.0lf\n", l2Stretch, seamLen, E_SD,
                                    (absGaussianCurve_original - absGaussianCurve) / seamLen, time);
                            
                            std::cout << buf << " processed" << std::endl;
                        }
                        
                        fclose(out);
                        fclose(dirList);
                        
                        std::cout << "stats.txt output in " << resultsFolderPath << std::endl;
                        break;
                    }
                        
                    case 2: {
                        // compute and output our metric for AutoCuts output
                        const std::string resultsFolderPath(argv[3]);
                        FILE *dirList = fopen((resultsFolderPath + "/folderList.txt").c_str(), "r");
                        assert(dirList);
                        
                        // for rendering:
                        energyTerms.emplace_back(new FracCuts::SymStretchEnergy());
                        triSoup.resize(2);
                        viewer.core.background_color << 1.0f, 1.0f, 1.0f, 0.0f;
                        viewer.callback_key_down = &key_down;
                        viewer.callback_pre_draw = &preDrawFunc;
                        viewer.callback_post_draw = &postDrawFunc;
                        viewer.data().show_lines = true;
                        viewer.core.orthographic = true;
                        viewer.core.camera_zoom *= 1.9;
                        viewer.core.animation_max_fps = 60.0;
                        viewer.data().point_size = 16.0f;
                        viewer.data().show_overlay = true;
                        viewer.core.is_animating = true;
                        viewer.launch_init(true, false);
                        
                        char buf[BUFSIZ];
                        while((!feof(dirList)) && fscanf(dirList, "%s", buf)) {
                            double lambda;
                            int iterNum, worldTime;
                            FILE *in = fopen((resultsFolderPath + '/' + std::string(buf) + "/info_pre.txt").c_str(), "r");
                            assert(in);
                            fscanf(in, "%lf%d%d", &lambda, &iterNum, &worldTime);
                            fclose(in);
                            
                            std::string meshPath(resultsFolderPath + '/' + std::string(buf) + "/finalResult_mesh.obj");
                            Eigen::MatrixXd V, UV, N;
                            Eigen::MatrixXi F, FUV, FN;
                            igl::readOBJ(meshPath, V, UV, N, F, FUV, FN);
                            UV.conservativeResize(UV.rows(), 2);
                            TriangleSoup resultMesh(V, F, UV, FUV, false, 0.0);
                            
                            std::ofstream file;
                            file.open(resultsFolderPath + '/' + std::string(buf) + "/info.txt");
                            assert(file.is_open());
                            
                            file << V.rows() << " " << F.rows() << std::endl;
                            
                            file << iterNum << " " << 0 << " "
                                << 0 << " " << 0 << " "
                                << lambda << " " << lambda << std::endl;
                            
                            file << 0 << " " << 0 << " " << worldTime << std::endl;
                            
                            double seamLen;
                            resultMesh.computeSeamSparsity(seamLen, false);
                            double distortion;
                            SymStretchEnergy SD;
                            SD.computeEnergyVal(resultMesh, distortion);
                            file << distortion << " " << seamLen / resultMesh.virtualRadius << std::endl;
                            
                            resultMesh.outputStandardStretch(file);

                            file.close();
                            
                            triSoup[0] = triSoup[1] = &resultMesh;
                            texScale = 10.0 / (triSoup[0]->bbox.row(1) - triSoup[0]->bbox.row(0)).maxCoeff();
                            viewUV = true;
                            showTexture = false;
                            showDistortion = true;
                            updateViewerData();
                            viewer.launch_rendering(false);
                            saveScreenshot(resultsFolderPath + '/' + std::string(buf) + "/finalResult.png",
                                           1.0, false, true);
                            
                            for(int capture3DI = 0; capture3DI < 2; capture3DI++) {
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
                                viewUV = false;
                                showTexture = showDistortion = (capture3DI % 2);
                                updateViewerData();
                                std::string filePath = resultsFolderPath + '/' + std::string(buf) + "/3DView" + std::to_string(capture3DI / 2) + ((capture3DI % 2 == 0) ? "_seam.png" : "_distortion.png");
                                viewer.launch_rendering(false);
                                saveScreenshot(filePath, 1.0, false, true);
                            }
                            
                            std::cout << buf << " processed" << std::endl;
                        }

                        fclose(dirList);
                        
                        break;
                    }
                        
                    case 3: {
                        // check inversion
                        const std::string resultsFolderPath(argv[3]);
                        FILE *dirList = fopen((resultsFolderPath + "/folderList.txt").c_str(), "r");
                        assert(dirList);

                        char buf[BUFSIZ];
                        while((!feof(dirList)) && fscanf(dirList, "%s", buf)) {
                            std::string meshPath(resultsFolderPath + '/' + std::string(buf) + "/finalResult_mesh.obj");
                            Eigen::MatrixXd V, UV, N;
                            Eigen::MatrixXi F, FUV, FN;
                            igl::readOBJ(meshPath, V, UV, N, F, FUV, FN);
                            UV.conservativeResize(UV.rows(), 2);
                            TriangleSoup resultMesh(V, F, UV, FUV, false, 0.0);
                            
                            if(!resultMesh.checkInversion()) {
                                std::cout << buf << " inverted" << std::endl;
                            }
                        }
                        
                        fclose(dirList);
                        std::cout << "check finished" << std::endl;
                        
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
        }
    };
}

#endif /* Diagnostic_hpp */
