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
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/harmonic.h>

#include <cstdio>

extern std::string outputFolderPath;
extern igl::opengl::glfw::Viewer viewer;
extern bool viewUV;
extern bool showTexture;
extern int showDistortion;
extern double texScale;
extern std::vector<const FracCuts::TriangleSoup*> triSoup;
extern std::vector<FracCuts::Energy*> energyTerms;
extern std::vector<double> energyParams;
extern FracCuts::Optimizer* optimizer;
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
                        // compute and output our metric and visualization for AutoCuts output, also output AutoCuts' distortion for comparison
                        const std::string resultsFolderPath(argv[3]);
                        FILE *dirList = fopen((resultsFolderPath + "/folderList.txt").c_str(), "r");
                        assert(dirList);
                        
                        FILE *distFile = fopen((resultsFolderPath + "/distortion.txt").c_str(), "w");
                        assert(distFile);
                        
                        // for rendering:
                        energyTerms.emplace_back(new FracCuts::SymStretchEnergy());
                        energyParams.emplace_back(1.0);
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
                            
                            file << "0.0 0.0 " << worldTime << " 0.0 topo0.0 desc0.0 scaf0.0 enUp0.0" <<
                            " mtrComp0.0 mtrAssem0.0 symFac0.0 numFac0.0 backSolve0.0 lineSearch0.0" <<
                            " bSplit0.0 iSplit0.0 cMerge0.0" << std::endl;
                            
                            double seamLen;
                            resultMesh.computeSeamSparsity(seamLen, false);
                            double distortion;
                            SymStretchEnergy SD;
                            SD.computeEnergyVal(resultMesh, distortion);
                            file << distortion << " " << seamLen / resultMesh.virtualRadius << std::endl;
                            fprintf(distFile, "%s %lf\n", buf, distortion);
                            
                            resultMesh.outputStandardStretch(file);

                            file.close();
                            
                            triSoup[0] = triSoup[1] = &resultMesh;
                            texScale = 10.0 / (triSoup[0]->bbox.row(1) - triSoup[0]->bbox.row(0)).maxCoeff();
                            viewUV = true;
                            showTexture = false;
                            showDistortion = true;
                            optimizer = new FracCuts::Optimizer(*triSoup[0], energyTerms, energyParams, 0, false, false);
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
                            delete optimizer;
                        }

                        fclose(dirList);
                        fclose(distFile);
                        
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
                        
                    case 4: {
                        // output GI/Seamster results as input mesh files for our method
                        const std::string resultsFolderPath(argv[3]);
                        FILE *dirList = fopen((resultsFolderPath + "/folderList.txt").c_str(), "r");
                        assert(dirList);
                        
                        char buf[BUFSIZ];
                        while((!feof(dirList)) && fscanf(dirList, "%s", buf)) {
                            std::string resultName(buf);
//                            if(resultName.find("GeomImg") != std::string::npos)
                            {
                                std::string meshPath(resultsFolderPath + '/' + resultName + "/finalResult_mesh.obj");
                                Eigen::MatrixXd V, UV, N;
                                Eigen::MatrixXi F, FUV, FN;
                                igl::readOBJ(meshPath, V, UV, N, F, FUV, FN);
                                UV.conservativeResize(UV.rows(), 3);
                                UV.col(2) = Eigen::VectorXd::Zero(UV.rows());
                                
                                Eigen::VectorXi bnd;
                                igl::boundary_loop(FUV, bnd); // Find the open boundary
                                if(bnd.size()) {
                                    Eigen::MatrixXd bnd_uv;
                                    //            igl::map_vertices_to_circle(V, bnd, bnd_uv);
                                    FracCuts::IglUtils::map_vertices_to_circle(UV, bnd, bnd_uv);
                                    
                                    //            // Harmonic parametrization
                                    //            igl::harmonic(V, F, bnd, bnd_uv, 1, UV);
                                    
                                    // Harmonic map with uniform weights
                                    Eigen::SparseMatrix<double> A, M;
                                    FracCuts::IglUtils::computeUniformLaplacian(FUV, A);
                                    igl::harmonic(A, M, bnd, bnd_uv, 1, UV);
                                    //            FracCuts::IglUtils::computeMVCMtr(V, F, A);
                                    //            FracCuts::IglUtils::fixedBoundaryParam_MVC(A, bnd, bnd_uv, UV);
                                    
                                    igl::writeOBJ(outputFolderPath + resultName + ".obj",
                                                  V, F, N, FN, UV, FUV);
                                    
                                    std::cout << "mesh saved in " << outputFolderPath + resultName + ".obj" << std::endl;
                                }
                            }
                        }
                        
                        fclose(dirList);
                        std::cout << "output finished" << std::endl;
                        
                        break;
                    }
                        
                    case 5: {
                        // output for visualization
                        
                        const std::string resultsFolderPath(argv[3]);
                        FILE *dirList = fopen((resultsFolderPath + "/folderList.txt").c_str(), "r");
                        assert(dirList);
                        
                        char buf[BUFSIZ];
                        while((!feof(dirList)) && fscanf(dirList, "%s", buf)) {
                            std::string resultName(buf);
                            std::string meshPath(resultsFolderPath + '/' + resultName + "/finalResult_mesh.obj");
                            Eigen::MatrixXd V, UV, N;
                            Eigen::MatrixXi F, FUV, FN;
                            if(!igl::readOBJ(meshPath, V, UV, N, F, FUV, FN)) {
                                continue;
                            }
                            
                            // map texture to 0-1 and output UV boundary path in both 2D and 3D
                            if(UV.rows() == 0) {
                                std::cout << "no input UV" << std::endl;
                                continue;
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
                            
                            igl::writeOBJ(resultsFolderPath + '/' + resultName + "/output_with01UV.obj",
                                          V, F, N, FN, UV, FUV);
                            std::cout << "texture mapped to [0,1]^2 and saved into " << outputFolderPath << resultName << " folder" << std::endl;
                            
                            Eigen::MatrixXd V_UV;
                            if(meshPath.find("AutoCuts") != std::string::npos) {
                                V_UV = UV;
                                UV.conservativeResize(UV.rows(), 2);
                            }
                            else {
                                V_UV.resize(UV.rows(), 3);
                                V_UV << UV, Eigen::VectorXd::Zero(UV.rows());
                            }
                            igl::writeOBJ(resultsFolderPath + '/' + resultName + "/output_01UV.obj",
                                          V_UV, FUV, Eigen::MatrixXd(), Eigen::MatrixXi(), UV, FUV);
                            
                            
                            FracCuts::TriangleSoup temp(V, F, UV, FUV, false);
                            
                            std::vector<std::vector<int>> bnd_all;
                            igl::boundary_loop(temp.F, bnd_all);
                            
                            FILE *out = fopen((resultsFolderPath + '/' + resultName + "/output_with01UV.sp").c_str(), "w");
                            assert(out);
                            FILE *out_UV = fopen((resultsFolderPath + '/' + resultName + "/output_01UV.sp").c_str(), "w");
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
                        }
                        
                        fclose(dirList);
                        std::cout << "output finished" << std::endl;
                        
                        break;
                    }
                        
                    case 6: {
                        // output ExpInfo into js variables for local web visualization
                        const std::string resultsFolderPath(argv[3]);
                        FILE *dirList = fopen((resultsFolderPath + "/folderList.txt").c_str(), "r");
                        assert(dirList);
                        
                        FILE *outFile = fopen((resultsFolderPath + "/data_ExpInfo.js").c_str(), "w");
                        assert(outFile);
                        
                        std::string resultsFolderName;
                        int endI_substr = resultsFolderPath.find_last_of('/');
                        if(endI_substr == std::string::npos) {
                            resultsFolderName = resultsFolderPath;
                        }
                        else {
                            if(endI_substr == resultsFolderPath.length() - 1) {
                                endI_substr = resultsFolderPath.find_last_of('/', endI_substr - 1);
                                resultsFolderName = resultsFolderPath.substr(endI_substr + 1,
                                                                             resultsFolderPath.length() - endI_substr - 2);
                            }
                            else {
                                resultsFolderName = resultsFolderPath.substr(endI_substr + 1,
                                                                             resultsFolderPath.length() - endI_substr - 1);
                            }
                        }
                        
                        fprintf(outFile, "var %s = [\n", resultsFolderName.c_str());
                        
                        char buf[BUFSIZ];
                        while((!feof(dirList)) && fscanf(dirList, "%s", buf)) {
                            std::string resultName(buf);
                            std::string infoFilePath(resultsFolderPath + '/' + resultName + "/info.txt");
                            std::ifstream infoFile(infoFilePath);
                            if(infoFile.is_open()) {
                                fprintf(outFile, "\tnew ExpInfo(");
                                
                                std::string bypass;
                                int vertAmt, faceAmt;
                                infoFile >> vertAmt >> faceAmt;
                                fprintf(outFile, "%d, %d, ", vertAmt, faceAmt);
                                
                                int innerIterNum, outerIterNum;
                                double lambda_init, lambda_end;
                                infoFile >> innerIterNum >> outerIterNum >>
                                    bypass >> bypass >>
                                    lambda_init >> lambda_end;
                                fprintf(outFile, "%d, %d, 0, 0, %lf, %lf, ",
                                        innerIterNum, outerIterNum, lambda_init, lambda_end);
                                
                                double time, duration;
                                infoFile >> bypass >> bypass >> time >> duration;
                                for(int wordI = 0; wordI < 13; wordI++) {
                                    infoFile >> bypass;
                                }
                                fprintf(outFile, "0.0, 0.0, %lf, [], [], ", time);
                                
                                double E_d, E_s;
                                infoFile >> E_d >> E_s;
                                fprintf(outFile, "%lf, %lf, ", E_d, E_s);
                                
                                double l2Stretch, lInfStretch, l2Shear, lInfCompress;
                                infoFile >> l2Stretch >> lInfStretch >> l2Shear >> lInfCompress;
                                fprintf(outFile, "%lf, %lf, %lf, ", l2Stretch, lInfStretch, l2Shear);
                                
                                fprintf(outFile, "\"%s\"", buf);
                                
                                fprintf(outFile, "),\n");
                                
                                infoFile.close();
                            }
                            else {
                                std::cout << "can't open " << infoFilePath << std::endl;
                            }
                        }
                        fprintf(outFile, "];\n");
                        
                        fclose(dirList);
                        fclose(outFile);
                        std::cout << "output finished" << std::endl;
                        
                        break;
                    }
                        
                    case 7: {
                        // check whether oscillation detected or converged exactly
                        const std::string resultsFolderPath(argv[3]);
                        FILE *dirList = fopen((resultsFolderPath + "/folderList.txt").c_str(), "r");
                        assert(dirList);
                        
                        std::string resultsFolderName;
                        int endI_substr = resultsFolderPath.find_last_of('/');
                        if(endI_substr == std::string::npos) {
                            resultsFolderName = resultsFolderPath;
                        }
                        else {
                            if(endI_substr == resultsFolderPath.length() - 1) {
                                endI_substr = resultsFolderPath.find_last_of('/', endI_substr - 1);
                                resultsFolderName = resultsFolderPath.substr(endI_substr + 1,
                                                                             resultsFolderPath.length() - endI_substr - 2);
                            }
                            else {
                                resultsFolderName = resultsFolderPath.substr(endI_substr + 1,
                                                                             resultsFolderPath.length() - endI_substr - 1);
                            }
                        }
                        
                        char buf[BUFSIZ];
                        int oscAmt = 0, exactConvAmt = 0, rollbackAmt_osc = 0, rollbackAmt_conv = 0;
                        while((!feof(dirList)) && fscanf(dirList, "%s", buf)) {
                            std::string resultName(buf);
                            std::string infoFilePath(resultsFolderPath + '/' + resultName + "/log.txt");
                            FILE *infoFile = fopen(infoFilePath.c_str(), "r");
                            if(infoFile) {
                                std::vector<std::string> lines;
                                char line[BUFSIZ];
                                while((!feof(infoFile)) && fgets(line, BUFSIZ, infoFile)) {
                                    lines.emplace_back(line);
                                }
                                
                                bool oscillation = false, issue = true;
                                for(int i = 0; (i < 10) && (i < lines.size()); i++) {
                                    if(lines[lines.size() - 1 - i].find("oscillation") != std::string::npos) {
                                        issue = false;
                                        oscillation = true;
                                        oscAmt++;
                                        break;
                                    }
                                    if(lines[lines.size() - 1 - i].find("all converged") != std::string::npos) {
                                        issue = false;
                                        exactConvAmt++;
                                        break;
                                    }
                                }
                                
                                for(int i = 0; (i < 10) && (i < lines.size()); i++) {
                                    if(lines[lines.size() - 1 - i].find("rolled back") != std::string::npos) {
                                        if(oscillation) {
                                            rollbackAmt_osc++;
                                        }
                                        else {
                                            rollbackAmt_conv++;
                                        }
                                    }
                                }
                                
                                if(issue) {
                                    std::cout << resultName << std::endl;
                                }
                                
                                fclose(infoFile);
                            }
                            else {
                                std::cout << "can't open " << infoFilePath << std::endl;
                            }
                        }
                        
                        fclose(dirList);
                        std::cout << oscAmt << " oscillation (" << rollbackAmt_osc << " rollbacks), " <<
                            exactConvAmt << " exact convergence (" << rollbackAmt_conv << " rollbacks)." << std::endl;
                        
                        break;
                    }
                        
                    case 8: {
                        // output for sequence visualization
                        const std::string resultsFolderPath(argv[3]);
                        for(int frameI = 0; ; frameI++) {
                            std::string meshPath(resultsFolderPath + '/' + std::to_string(frameI) + "_mesh.obj");
                            Eigen::MatrixXd V, UV, N;
                            Eigen::MatrixXi F, FUV, FN;
                            if(!igl::readOBJ(meshPath, V, UV, N, F, FUV, FN)) {
                                break;
                            }
                            
                            // map texture to 0-1 and output UV boundary path in both 2D and 3D
                            if(UV.rows() == 0) {
                                std::cout << "no input UV" << std::endl;
                                continue;
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
                            
                            igl::writeOBJ(resultsFolderPath + '/' + std::to_string(frameI) + "_with01UV.obj",
                                          V, F, N, FN, UV, FUV);
                            std::cout << "texture mapped to [0,1]^2 and saved into " <<
                                resultsFolderPath << " folder" << std::endl;
                            
                            Eigen::MatrixXd V_UV;
                            if(meshPath.find("AutoCuts") != std::string::npos) {
                                V_UV = UV;
                                UV.conservativeResize(UV.rows(), 2);
                            }
                            else {
                                V_UV.resize(UV.rows(), 3);
                                V_UV << UV, Eigen::VectorXd::Zero(UV.rows());
                            }
                            igl::writeOBJ(resultsFolderPath + '/' + std::to_string(frameI) + "_01UV.obj",
                                          V_UV, FUV, Eigen::MatrixXd(), Eigen::MatrixXi(), UV, FUV);
                            
                            
                            FracCuts::TriangleSoup temp(V, F, UV, FUV, false);
                            
                            std::vector<std::vector<int>> bnd_all;
                            igl::boundary_loop(temp.F, bnd_all);
                            
                            FILE *out = fopen((resultsFolderPath + '/' + std::to_string(frameI) + "_with01UV.sp").c_str(), "w");
                            assert(out);
                            FILE *out_UV = fopen((resultsFolderPath + '/' + std::to_string(frameI) + "_01UV.sp").c_str(), "w");
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
                        }
                        
                        std::cout << "output finished" << std::endl;
                        
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
