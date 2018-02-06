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

#include <cstdio>

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
