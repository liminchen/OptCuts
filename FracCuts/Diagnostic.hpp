//
//  Diagnostic.hpp
//  FracCuts
//
//  Created by Minchen Li on 1/31/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#ifndef Diagnostic_hpp
#define Diagnostic_hpp

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
