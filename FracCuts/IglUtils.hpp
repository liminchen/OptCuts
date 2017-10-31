//
//  IglUtils.hpp
//  FracCuts
//
//  Created by Minchen Li on 8/30/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#ifndef IglUtils_hpp
#define IglUtils_hpp

#include <Eigen/Eigen>

#include <iostream>
#include <fstream>

namespace FracCuts {
    
    // a static class implementing basic geometry processing operations that are not provided in libIgl
    class IglUtils {
    public:
        static void computeGraphLaplacian(const Eigen::MatrixXi& F, Eigen::SparseMatrix<double>& graphL);
        
        // graph laplacian with half-weighted boundary edge, the computation is also faster
        static void computeUniformLaplacian(const Eigen::MatrixXi& F, Eigen::SparseMatrix<double>& graphL);
        
        static void mapTriangleTo2D(const Eigen::Vector3d v[3], Eigen::Vector2d u[3]);
        static void computeDeformationGradient(const Eigen::Vector3d v[3], const Eigen::Vector2d u[3], Eigen::Matrix2d& F);
        
        // to a circle with the perimeter equal to the length of the boundary on the mesh
        static void map_vertices_to_circle(const Eigen::MatrixXd& V, const Eigen::VectorXi& bnd, Eigen::MatrixXd& UV);
        
        static void mapScalarToColor_bin(const Eigen::VectorXd& scalar, Eigen::MatrixXd& color, double thres);
        static void mapScalarToColor(const Eigen::VectorXd& scalar, Eigen::MatrixXd& color, double lowerBound, double upperBound);
        
        static void addBlockToMatrix(Eigen::SparseMatrix<double>& mtr, const Eigen::MatrixXd& block,
                                     const Eigen::VectorXi& index, int dim);
        static void addBlockToMatrix(const Eigen::MatrixXd& block, const Eigen::VectorXi& index, int dim,
                                     Eigen::VectorXd* V, Eigen::VectorXi* I = NULL, Eigen::VectorXi* J = NULL);
        static void addDiagonalToMatrix(const Eigen::VectorXd& diagonal, const Eigen::VectorXi& index, int dim,
                                     Eigen::VectorXd* V, Eigen::VectorXi* I = NULL, Eigen::VectorXi* J = NULL);
        
        template<typename Scalar, int rows, int cols>
        static void symmetrizeMatrix(Eigen::Matrix<Scalar, rows, cols>& mtr) {
            if(rows != cols) {
                return;
            }
            
            for(int rowI = 0; rowI < rows; rowI++) {
                for(int colI = rowI + 1; colI < cols; colI++) {
                    double &a = mtr(rowI, colI), &b = mtr(colI, rowI);
                    a = b = (a + b) / 2.0;
                }
            }
        }
        
        template<typename Scalar, int size>
        static void makePD(Eigen::Matrix<Scalar, size, size>& mtr) {
            Eigen::JacobiSVD<Eigen::Matrix<Scalar, size, size>> svd(mtr, Eigen::ComputeFullV);
            
            mtr = 0.5 * (mtr + svd.matrixV() * Eigen::DiagonalMatrix<Scalar, size>(svd.singularValues()) * svd.matrixV().transpose());
//            Eigen::DiagonalMatrix<Scalar, size> sigma_clamp(svd.singularValues());
//            for(int i = 0; i < size; i++) {
//                if(sigma_clamp.diagonal()[i] < 0.0) {
//                    sigma_clamp.diagonal()[i] = 0.0;
//                }
//            }
//            mtr = svd.matrixV() * sigma_clamp * svd.matrixV().transpose();
            
            symmetrizeMatrix(mtr);
        }
        
        static void writeSparseMatrixToFile(const std::string& filePath, const Eigen::SparseMatrix<double>& mtr, bool MATLAB = false);
        static void writeSparseMatrixToFile(const std::string& filePath, const Eigen::VectorXi& I, const Eigen::VectorXi& J,
                                            const Eigen::VectorXd& V, bool MATLAB = false);
        static void loadSparseMatrixFromFile(const std::string& filePath, Eigen::SparseMatrix<double>& mtr);
        
        static void sparseMatrixToTriplet(const Eigen::SparseMatrix<double>& mtr,
                                          Eigen::VectorXi& I, Eigen::VectorXi& J, Eigen::VectorXd& V);
        static void sparseMatrixToTriplet(const Eigen::SparseMatrix<double>& mtr, Eigen::VectorXd& V);
        
        static const std::string rtos(double real);
        
        static void differentiate_normalize(const Eigen::Vector2d& var, Eigen::Matrix2d& deriv);
        static void differentiate_xxT(const Eigen::Vector2d& var, Eigen::Matrix<Eigen::RowVector2d, 2, 2>& deriv,
                                      double param = 1.0);
    };
    
}

#endif /* IglUtils_hpp */
