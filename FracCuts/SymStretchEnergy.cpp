//
//  SymStretchEnergy.cpp
//  FracCuts
//
//  Created by Minchen Li on 9/3/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#include "SymStretchEnergy.hpp"
#include "IglUtils.hpp"

#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>

#include <tbb/tbb.h> // maybe beneficial for very large models

#include <limits>
#include <fstream>
#include <cfloat>

extern std::ofstream logFile;

namespace FracCuts {
    
    void SymStretchEnergy::getEnergyValPerElem(const TriangleSoup& data, Eigen::VectorXd& energyValPerElem, bool uniformWeight) const
    {
        const double normalizer_div = data.surfaceArea;
        
        energyValPerElem.resize(data.F.rows());
        for(int triI = 0; triI < data.F.rows(); triI++) {
            const Eigen::Vector3i& triVInd = data.F.row(triI);
            
            const Eigen::Vector2d& U1 = data.V.row(triVInd[0]);
            const Eigen::Vector2d& U2 = data.V.row(triVInd[1]);
            const Eigen::Vector2d& U3 = data.V.row(triVInd[2]);
            
            const Eigen::Vector2d U2m1 = U2 - U1;
            const Eigen::Vector2d U3m1 = U3 - U1;
            
            const double area_U = 0.5 * (U2m1[0] * U3m1[1] - U2m1[1] * U3m1[0]);
            
            const double w = (uniformWeight ? 1.0 : (data.triArea[triI] / normalizer_div));
            energyValPerElem[triI] = w * (1.0 + data.triAreaSq[triI] / area_U / area_U) *
                ((U3m1.squaredNorm() * data.e0SqLen[triI] + U2m1.squaredNorm() * data.e1SqLen[triI]) / 4 / data.triAreaSq[triI] -
                U3m1.dot(U2m1) * data.e0dote1[triI] / 2 / data.triAreaSq[triI]);
        }
    }
    
    void SymStretchEnergy::computeGradient(const TriangleSoup& data, Eigen::VectorXd& gradient) const
    {
        const double normalizer_div = data.surfaceArea;
        
        gradient.resize(data.V.rows() * 2);
        gradient.setZero();
        for(int triI = 0; triI < data.F.rows(); triI++) {
            const Eigen::Vector3i& triVInd = data.F.row(triI);
            
            const Eigen::Vector2d& U1 = data.V.row(triVInd[0]);
            const Eigen::Vector2d& U2 = data.V.row(triVInd[1]);
            const Eigen::Vector2d& U3 = data.V.row(triVInd[2]);
            
            const Eigen::Vector2d U2m1 = U2 - U1;
            const Eigen::Vector2d U3m1 = U3 - U1;
            
            const double area_U = 0.5 * (U2m1[0] * U3m1[1] - U2m1[1] * U3m1[0]);
            
            const double leftTerm = 1.0 + data.triAreaSq[triI] / area_U / area_U;
            const double rightTerm = (U3m1.squaredNorm() * data.e0SqLen[triI] + U2m1.squaredNorm() * data.e1SqLen[triI]) /
                4 / data.triAreaSq[triI] - U3m1.dot(U2m1) * data.e0dote1[triI] / 2 / data.triAreaSq[triI];
            
            const double areaRatio = data.triAreaSq[triI] / area_U / area_U / area_U;
            const double w = data.triArea[triI] / normalizer_div;
            
            const Eigen::Vector2d edge_oppo1 = U3 - U2;
            const Eigen::Vector2d dLeft1 = areaRatio * Eigen::Vector2d(edge_oppo1[1], -edge_oppo1[0]);
            const Eigen::Vector2d dRight1 = ((data.e0dote1[triI] - data.e0SqLen[triI]) * U3m1 +
                (data.e0dote1[triI] - data.e1SqLen[triI]) * U2m1) / 2.0 / data.triAreaSq[triI];
            gradient.block(triVInd[0] * 2, 0, 2, 1) += w * (dLeft1 * rightTerm + dRight1 * leftTerm);
            
            const Eigen::Vector2d edge_oppo2 = U1 - U3;
            const Eigen::Vector2d dLeft2 = areaRatio * Eigen::Vector2d(edge_oppo2[1], -edge_oppo2[0]);
            const Eigen::Vector2d dRight2 = (data.e1SqLen[triI] * U2m1 - data.e0dote1[triI] * U3m1) / 2.0 / data.triAreaSq[triI];
            gradient.block(triVInd[1] * 2, 0, 2, 1) += w * (dLeft2 * rightTerm + dRight2 * leftTerm);
            
            const Eigen::Vector2d edge_oppo3 = U2 - U1;
            const Eigen::Vector2d dLeft3 = areaRatio * Eigen::Vector2d(edge_oppo3[1], -edge_oppo3[0]);
            const Eigen::Vector2d dRight3 = (data.e0SqLen[triI] * U3m1 - data.e0dote1[triI] * U2m1) / 2.0 / data.triAreaSq[triI];
            gradient.block(triVInd[2] * 2, 0, 2, 1) += w * (dLeft3 * rightTerm + dRight3 * leftTerm);
        }
        
        for(const auto fixedVI : data.fixedVert) {
            gradient[2 * fixedVI] = 0.0;
            gradient[2 * fixedVI + 1] = 0.0;
        }
    }
    
    void SymStretchEnergy::computePrecondMtr(const TriangleSoup& data, Eigen::SparseMatrix<double>& precondMtr) const
    {
//        precondMtr = data.LaplacianMtr;
        computeHessian(data, precondMtr);
//        IglUtils::writeSparseMatrixToFile("/Users/mincli/Desktop/FracCuts/mtr", precondMtr, true);
        
//        Eigen::BDCSVD<Eigen::MatrixXd> svd((Eigen::MatrixXd(precondMtr)));
//        logFile << "singular values of precondMtr_ESD:" << std::endl << svd.singularValues() << std::endl;
//        double det = 1.0;
//        for(int i = 0; i < svd.singularValues().size(); i++) {
//            det *= svd.singularValues()[i];
//        }
//        std::cout << "det(precondMtr_ESD) = " << det << std::endl;
    }
    
    void SymStretchEnergy::computePrecondMtr(const TriangleSoup& data, Eigen::VectorXd* V,
                                   Eigen::VectorXi* I, Eigen::VectorXi* J) const
    {
        const double normalizer_div = data.surfaceArea;
        
//        std::cout << "computing entry value..." << std::endl;
//        clock_t start = clock();
        for(int triI = 0; triI < data.F.rows(); triI++) {
            const Eigen::Vector3i& triVInd = data.F.row(triI);
            
            const Eigen::Vector2d& U1 = data.V.row(triVInd[0]);
            const Eigen::Vector2d& U2 = data.V.row(triVInd[1]);
            const Eigen::Vector2d& U3 = data.V.row(triVInd[2]);
            
            const Eigen::Vector2d U2m1 = U2 - U1;
            const Eigen::Vector2d U3m1 = U3 - U1;
            
            const double area_U = 0.5 * (U2m1[0] * U3m1[1] - U2m1[1] * U3m1[0]);
            const double areaRatio = data.triAreaSq[triI] / area_U / area_U / area_U;
            const double dAreaRatio_div_dArea_mult = 3.0 / 2.0 * areaRatio / area_U;
            
            const double w = data.triArea[triI] / normalizer_div;
            
            const double e0SqLen_div_dbAreaSq = data.e0SqLen_div_dbAreaSq[triI];
            const double e1SqLen_div_dbAreaSq = data.e1SqLen_div_dbAreaSq[triI];
            const double e0dote1_div_dbAreaSq = data.e0dote1_div_dbAreaSq[triI];
            
            // compute energy terms
            const double leftTerm = 1.0 + data.triAreaSq[triI] / area_U / area_U;
            const double rightTerm = (U3m1.squaredNorm() * e0SqLen_div_dbAreaSq + U2m1.squaredNorm() * e1SqLen_div_dbAreaSq) / 2. - U3m1.dot(U2m1) * e0dote1_div_dbAreaSq;
            
            const Eigen::Vector2d edge_oppo1 = U3 - U2;
            const Eigen::Vector2d edge_oppo2 = U1 - U3;
            const Eigen::Vector2d edge_oppo3 = U2 - U1;
            const Eigen::Vector2d edge_oppo1_Ortho = Eigen::Vector2d(edge_oppo1[1], -edge_oppo1[0]);
            const Eigen::Vector2d edge_oppo2_Ortho = Eigen::Vector2d(edge_oppo2[1], -edge_oppo2[0]);
            const Eigen::Vector2d edge_oppo3_Ortho = Eigen::Vector2d(edge_oppo3[1], -edge_oppo3[0]);
            Eigen::Matrix2d dOrtho_div_dU; dOrtho_div_dU << 0.0, -1.0, 1.0, 0.0;
            
            // compute 1st order derivatives
            const Eigen::Vector2d dLeft1 = areaRatio * edge_oppo1_Ortho;
            const Eigen::Vector2d dRight1 = ((e0dote1_div_dbAreaSq - e0SqLen_div_dbAreaSq) * U3m1 +
                                             (e0dote1_div_dbAreaSq - e1SqLen_div_dbAreaSq) * U2m1);
            
            const Eigen::Vector2d dLeft2 = areaRatio * edge_oppo2_Ortho;
            const Eigen::Vector2d dRight2 = (e1SqLen_div_dbAreaSq * U2m1 - e0dote1_div_dbAreaSq * U3m1);
            
            const Eigen::Vector2d dLeft3 = areaRatio * edge_oppo3_Ortho;
            const Eigen::Vector2d dRight3 = (e0SqLen_div_dbAreaSq * U3m1 - e0dote1_div_dbAreaSq * U2m1);
            
            Eigen::Matrix<double, 6, 6> curHessian;
            
            // compute second order derivatives for g_U1
            const Eigen::Matrix2d d2Left11 = dAreaRatio_div_dArea_mult * edge_oppo1_Ortho * edge_oppo1_Ortho.transpose();
            const double d2Right11 = (e0SqLen_div_dbAreaSq + e1SqLen_div_dbAreaSq - 2.0 * e0dote1_div_dbAreaSq);
            const Eigen::Matrix2d dLeft1dRight1T = dLeft1 * dRight1.transpose();
            curHessian.block(0, 0, 2, 2) = w * (d2Left11 * rightTerm + dLeft1dRight1T +
                                                d2Right11 * leftTerm * Eigen::Matrix2d::Identity() + dLeft1dRight1T.transpose());
            curHessian(0, 1) = curHessian(1, 0) = (curHessian(0, 1) + curHessian(1, 0)) / 2.0;
        
            const Eigen::Matrix2d d2Left12 = dAreaRatio_div_dArea_mult * edge_oppo1_Ortho * edge_oppo2_Ortho.transpose() +
            areaRatio * dOrtho_div_dU;
            const double d2Right12 = (e0dote1_div_dbAreaSq - e1SqLen_div_dbAreaSq);
            curHessian.block(0, 2, 2, 2) = w * (d2Left12 * rightTerm + dLeft1 * dRight2.transpose() +
                                                d2Right12 * leftTerm * Eigen::Matrix2d::Identity() + dRight1 * dLeft2.transpose());
            curHessian.block(2, 0, 2, 2) = curHessian.block(0, 2, 2, 2).transpose();
    
            const Eigen::Matrix2d d2Left13 = dAreaRatio_div_dArea_mult * edge_oppo1_Ortho * edge_oppo3_Ortho.transpose() +
            areaRatio * (-dOrtho_div_dU);
            const double d2Right13 = (e0dote1_div_dbAreaSq - e0SqLen_div_dbAreaSq);
            curHessian.block(0, 4, 2, 2) = w * (d2Left13 * rightTerm + dLeft1 * dRight3.transpose() +
                                                d2Right13 * leftTerm * Eigen::Matrix2d::Identity() + dRight1 * dLeft3.transpose());
            curHessian.block(4, 0, 2, 2) = curHessian.block(0, 4, 2, 2).transpose();
        
            // compute second order derivatives for g_U2
            const Eigen::Matrix2d d2Left22 = dAreaRatio_div_dArea_mult * edge_oppo2_Ortho * edge_oppo2_Ortho.transpose();
            const double d2Right22 = e1SqLen_div_dbAreaSq;
            curHessian.block(2, 2, 2, 2) = w * (d2Left22 * rightTerm + dLeft2 * dRight2.transpose() +
                                                d2Right22 * leftTerm * Eigen::Matrix2d::Identity() + dRight2 * dLeft2.transpose());
            curHessian(2, 3) = curHessian(3, 2) = (curHessian(2, 3) + curHessian(3, 2)) / 2.0;
        
            const Eigen::Matrix2d d2Left23 = dAreaRatio_div_dArea_mult * edge_oppo2_Ortho * edge_oppo3_Ortho.transpose() +
            areaRatio * dOrtho_div_dU;
            const double d2Right23 = -e0dote1_div_dbAreaSq;
            curHessian.block(2, 4, 2, 2) = w * (d2Left23 * rightTerm + dLeft2 * dRight3.transpose() +
                                                d2Right23 * leftTerm * Eigen::Matrix2d::Identity() + dRight2 * dLeft3.transpose());
            curHessian.block(4, 2, 2, 2) = curHessian.block(2, 4, 2, 2).transpose();
        
            // compute second order derivatives for g_U3
            const Eigen::Matrix2d d2Left33 = dAreaRatio_div_dArea_mult * edge_oppo3_Ortho * edge_oppo3_Ortho.transpose();
            const double d2Right33 = e0SqLen_div_dbAreaSq;
            curHessian.block(4, 4, 2, 2) = w * (d2Left33 * rightTerm + dLeft3 * dRight3.transpose() +
                                                d2Right33 * leftTerm * Eigen::Matrix2d::Identity() + dRight3 * dLeft3.transpose());
            curHessian(4, 5) = curHessian(5, 4) = (curHessian(4, 5) + curHessian(5, 4)) / 2.0;
            
            // project to nearest SPD matrix
            Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> svd(curHessian, Eigen::ComputeFullV);
            Eigen::Matrix<double, 6, 6> curHessian_SPD = 0.5 * (curHessian +
                svd.matrixV() * Eigen::DiagonalMatrix<double, 6>(svd.singularValues()) * svd.matrixV().transpose());
            IglUtils::symmetrizeMatrix(curHessian_SPD);
//            for(int corI = 0; (curHessian_SPD.determinant() <= 0.0) && (corI < 1); corI++) {
//                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eigenSolver(curHessian_SPD);
//                curHessian_SPD.diagonal() -= std::nextafter(eigenSolver.eigenvalues()[0], DBL_MIN) * Eigen::VectorXd::Ones(6);
//            }
            
            Eigen::VectorXi vInd = triVInd;
            for(int vI = 0; vI < 3; vI++) {
                if(data.fixedVert.find(vInd[vI]) != data.fixedVert.end()) {
                    vInd[vI] = -1;
                }
            }
            IglUtils::addBlockToMatrix(curHessian_SPD, vInd, 2, V, I, J);
//            IglUtils::addBlockToMatrix(curHessian, vInd, 2, V, I, J);
        }
//        std::cout << static_cast<double>(clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;
        
        Eigen::VectorXi fixedVertInd;
        fixedVertInd.resize(data.fixedVert.size());
        int fVI = 0;
        for(const auto fixedVI : data.fixedVert) {
            fixedVertInd[fVI++] = fixedVI;
        }
        IglUtils::addDiagonalToMatrix(Eigen::VectorXd::Ones(data.fixedVert.size() * 2),
                                      fixedVertInd, 2, V, I, J);
    }
    
    void SymStretchEnergy::computeHessian(const TriangleSoup& data, Eigen::SparseMatrix<double>& hessian) const
    {
        const double normalizer_div = data.surfaceArea;
        
        hessian.conservativeResize(data.V.rows() * 2, data.V.rows() * 2);
        hessian.reserve(data.V.rows() * 7 * 4);
//        hessian.setZero();

        std::vector<Eigen::Matrix<double, 6, 6>> triHessian_SPD(data.F.rows());
        Eigen::MatrixXi triVInd_withFixed = data.F;
        std::cout << "computing entry value..." << std::endl;
        clock_t start = clock();
        for(int triI = 0; triI < data.F.rows(); triI++) {
//        tbb::parallel_for(0, static_cast<int>(data.F.rows()), 1, [&](int triI) {
                    
            const Eigen::Vector3i& triVInd = data.F.row(triI);
            
            const Eigen::Vector2d& U1 = data.V.row(triVInd[0]);
            const Eigen::Vector2d& U2 = data.V.row(triVInd[1]);
            const Eigen::Vector2d& U3 = data.V.row(triVInd[2]);
            
            const Eigen::Vector2d U2m1 = U2 - U1;
            const Eigen::Vector2d U3m1 = U3 - U1;
            
            const double area_U = 0.5 * (U2m1[0] * U3m1[1] - U2m1[1] * U3m1[0]);
            const double areaRatio = data.triAreaSq[triI] / area_U / area_U / area_U;
            const double dAreaRatio_div_dArea_mult = 3.0 / 2.0 * areaRatio / area_U;
            
            const double w = data.triArea[triI] / normalizer_div;
            
            const double e0SqLen_div_dbAreaSq = data.e0SqLen_div_dbAreaSq[triI];
            const double e1SqLen_div_dbAreaSq = data.e1SqLen_div_dbAreaSq[triI];
            const double e0dote1_div_dbAreaSq = data.e0dote1_div_dbAreaSq[triI];
            
            // compute energy terms
            const double leftTerm = 1.0 + data.triAreaSq[triI] / area_U / area_U;
//            const double rightTerm = (U3m1.squaredNorm() * data.e0SqLen[triI] + U2m1.squaredNorm() * data.e1SqLen[triI]) /
//                4. / data.triAreaSq[triI] - U3m1.dot(U2m1) * data.e0dote1[triI] / 2. / data.triAreaSq[triI];
            const double rightTerm = (U3m1.squaredNorm() * e0SqLen_div_dbAreaSq + U2m1.squaredNorm() * e1SqLen_div_dbAreaSq) / 2. - U3m1.dot(U2m1) * e0dote1_div_dbAreaSq;
            
            const Eigen::Vector2d edge_oppo1 = U3 - U2;
            const Eigen::Vector2d edge_oppo2 = U1 - U3;
            const Eigen::Vector2d edge_oppo3 = U2 - U1;
            const Eigen::Vector2d edge_oppo1_Ortho = Eigen::Vector2d(edge_oppo1[1], -edge_oppo1[0]);
            const Eigen::Vector2d edge_oppo2_Ortho = Eigen::Vector2d(edge_oppo2[1], -edge_oppo2[0]);
            const Eigen::Vector2d edge_oppo3_Ortho = Eigen::Vector2d(edge_oppo3[1], -edge_oppo3[0]);
            Eigen::Matrix2d dOrtho_div_dU; dOrtho_div_dU << 0.0, -1.0, 1.0, 0.0;
            
            // compute 1st order derivatives
            const Eigen::Vector2d dLeft1 = areaRatio * edge_oppo1_Ortho;
//            const Eigen::Vector2d dRight1 = ((data.e0dote1[triI] - data.e0SqLen[triI]) * U3m1 +
//                                             (data.e0dote1[triI] - data.e1SqLen[triI]) * U2m1) / 2.0 / data.triAreaSq[triI];
            const Eigen::Vector2d dRight1 = ((e0dote1_div_dbAreaSq - e0SqLen_div_dbAreaSq) * U3m1 +
                                            (e0dote1_div_dbAreaSq - e1SqLen_div_dbAreaSq) * U2m1);
            
            const Eigen::Vector2d dLeft2 = areaRatio * edge_oppo2_Ortho;
//            const Eigen::Vector2d dRight2 = (data.e1SqLen[triI] * U2m1 - data.e0dote1[triI] * U3m1) / 2.0 / data.triAreaSq[triI];
            const Eigen::Vector2d dRight2 = (e1SqLen_div_dbAreaSq * U2m1 - e0dote1_div_dbAreaSq * U3m1);
            
            const Eigen::Vector2d dLeft3 = areaRatio * edge_oppo3_Ortho;
//            const Eigen::Vector2d dRight3 = (data.e0SqLen[triI] * U3m1 - data.e0dote1[triI] * U2m1) / 2.0 / data.triAreaSq[triI];
            const Eigen::Vector2d dRight3 = (e0SqLen_div_dbAreaSq * U3m1 - e0dote1_div_dbAreaSq * U2m1);
            
            Eigen::Matrix<double, 6, 6> curHessian;
            
            // compute second order derivatives for g_U1
            const Eigen::Matrix2d d2Left11 = dAreaRatio_div_dArea_mult * edge_oppo1_Ortho * edge_oppo1_Ortho.transpose();
//            const double d2Right11 = (data.e0SqLen[triI] + data.e1SqLen[triI] - 2.0 * data.e0dote1[triI]) / 2.0 / data.triAreaSq[triI];
            const double d2Right11 = (e0SqLen_div_dbAreaSq + e1SqLen_div_dbAreaSq - 2.0 * e0dote1_div_dbAreaSq);
            const Eigen::Matrix2d dLeft1dRight1T = dLeft1 * dRight1.transpose();
            curHessian.block(0, 0, 2, 2) = w * (d2Left11 * rightTerm + dLeft1dRight1T +
                d2Right11 * leftTerm * Eigen::Matrix2d::Identity() + dLeft1dRight1T.transpose());
            
            const Eigen::Matrix2d d2Left12 = dAreaRatio_div_dArea_mult * edge_oppo1_Ortho * edge_oppo2_Ortho.transpose() +
                areaRatio * dOrtho_div_dU;
//            const double d2Right12 = (data.e0dote1[triI] - data.e1SqLen[triI]) / 2.0 / data.triAreaSq[triI];
            const double d2Right12 = (e0dote1_div_dbAreaSq - e1SqLen_div_dbAreaSq);
            curHessian.block(0, 2, 2, 2) = w * (d2Left12 * rightTerm + dLeft1 * dRight2.transpose() +
                d2Right12 * leftTerm * Eigen::Matrix2d::Identity() + dRight1 * dLeft2.transpose());
            
            const Eigen::Matrix2d d2Left13 = dAreaRatio_div_dArea_mult * edge_oppo1_Ortho * edge_oppo3_Ortho.transpose() +
                areaRatio * (-dOrtho_div_dU);
//            const double d2Right13 = (data.e0dote1[triI] - data.e0SqLen[triI]) / 2.0 / data.triAreaSq[triI];
            const double d2Right13 = (e0dote1_div_dbAreaSq - e0SqLen_div_dbAreaSq);
            curHessian.block(0, 4, 2, 2) = w * (d2Left13 * rightTerm + dLeft1 * dRight3.transpose() +
                d2Right13 * leftTerm * Eigen::Matrix2d::Identity() + dRight1 * dLeft3.transpose());
            
            // compute second order derivatives for g_U2
            const Eigen::Matrix2d d2Left21 = dAreaRatio_div_dArea_mult * edge_oppo2_Ortho * edge_oppo1_Ortho.transpose() +
                areaRatio * (-dOrtho_div_dU);
//            const double d2Right21 = (data.e0dote1[triI] - data.e1SqLen[triI]) / 2.0 / data.triAreaSq[triI];
            const double d2Right21 = (e0dote1_div_dbAreaSq - e1SqLen_div_dbAreaSq);
            curHessian.block(2, 0, 2, 2) = w * (d2Left21 * rightTerm + dLeft2 * dRight1.transpose() +
                d2Right21 * leftTerm * Eigen::Matrix2d::Identity() + dRight2 * dLeft1.transpose());
            
            const Eigen::Matrix2d d2Left22 = dAreaRatio_div_dArea_mult * edge_oppo2_Ortho * edge_oppo2_Ortho.transpose();
//            const double d2Right22 = data.e1SqLen[triI] / 2.0 / data.triAreaSq[triI];
            const double d2Right22 = e1SqLen_div_dbAreaSq;
            curHessian.block(2, 2, 2, 2) = w * (d2Left22 * rightTerm + dLeft2 * dRight2.transpose() +
                d2Right22 * leftTerm * Eigen::Matrix2d::Identity() + dRight2 * dLeft2.transpose());
            
            const Eigen::Matrix2d d2Left23 = dAreaRatio_div_dArea_mult * edge_oppo2_Ortho * edge_oppo3_Ortho.transpose() +
                areaRatio * dOrtho_div_dU;
//            const double d2Right23 = -data.e0dote1[triI] / 2.0 / data.triAreaSq[triI];
            const double d2Right23 = -e0dote1_div_dbAreaSq;
            curHessian.block(2, 4, 2, 2) = w * (d2Left23 * rightTerm + dLeft2 * dRight3.transpose() +
                d2Right23 * leftTerm * Eigen::Matrix2d::Identity() + dRight2 * dLeft3.transpose());
            
            // compute second order derivatives for g_U3
            const Eigen::Matrix2d d2Left31 = dAreaRatio_div_dArea_mult * edge_oppo3_Ortho * edge_oppo1_Ortho.transpose() +
                areaRatio * dOrtho_div_dU;
//            const double d2Right31 = (data.e0dote1[triI] - data.e0SqLen[triI]) / 2.0 / data.triAreaSq[triI];
            const double d2Right31 = (e0dote1_div_dbAreaSq - e0SqLen_div_dbAreaSq);
            curHessian.block(4, 0, 2, 2) = w * (d2Left31 * rightTerm + dLeft3 * dRight1.transpose() +
                d2Right31 * leftTerm * Eigen::Matrix2d::Identity() + dRight3 * dLeft1.transpose());
            
            const Eigen::Matrix2d d2Left32 = dAreaRatio_div_dArea_mult * edge_oppo3_Ortho * edge_oppo2_Ortho.transpose() +
                areaRatio * (-dOrtho_div_dU);
            const double d2Right32 = d2Right23;
            curHessian.block(4, 2, 2, 2) = w * (d2Left32 * rightTerm + dLeft3 * dRight2.transpose() +
                d2Right32 * leftTerm * Eigen::Matrix2d::Identity() + dRight3 * dLeft2.transpose());
            
            const Eigen::Matrix2d d2Left33 = dAreaRatio_div_dArea_mult * edge_oppo3_Ortho * edge_oppo3_Ortho.transpose();
//            const double d2Right33 = data.e0SqLen[triI] / 2.0 / data.triAreaSq[triI];
            const double d2Right33 = e0SqLen_div_dbAreaSq;
            curHessian.block(4, 4, 2, 2) = w * (d2Left33 * rightTerm + dLeft3 * dRight3.transpose() +
                d2Right33 * leftTerm * Eigen::Matrix2d::Identity() + dRight3 * dLeft3.transpose());
            
            const Eigen::Matrix<double, 6, 6> curHessian_sym = (curHessian + curHessian.transpose()) / 2.0;
            Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> svd(curHessian_sym, Eigen::ComputeFullV);
//            const Eigen::Matrix<double, 6, 6> curHessian_SPD = 0.5 * (curHessian_sym +
//                svd.matrixV() * Eigen::DiagonalMatrix<double, 6>(svd.singularValues()) * svd.matrixV().transpose());
            triHessian_SPD[triI] = 0.5 * (curHessian_sym +
                svd.matrixV() * Eigen::DiagonalMatrix<double, 6>(svd.singularValues()) * svd.matrixV().transpose());
//            triHessian_SPD[triI] = curHessian;
            
//            Eigen::VectorXi vInd = triVInd;
//            for(int vI = 0; vI < 3; vI++) {
//                if(data.fixedVert.find(vInd[vI]) != data.fixedVert.end()) {
//                    vInd[vI] = -1;
//                }
//            }
//            IglUtils::addBlockToMatrix(hessian, curHessian_SPD, vInd, 2);
////            IglUtils::addBlockToMatrix(hessian, curHessian, vInd, 2);
            for(int vI = 0; vI < 3; vI++) {
                if(data.fixedVert.find(triVInd[vI]) != data.fixedVert.end()) {
                    triVInd_withFixed(triI, vI) = -1;
                }
            }
        }
//        });
        std::cout << static_cast<double>(clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;
        
        std::cout << "inserting elements..." << std::endl;
        start = clock();
        for(int triI = 0; triI < data.F.rows(); triI++) {
            IglUtils::addBlockToMatrix(hessian, triHessian_SPD[triI], triVInd_withFixed.row(triI), 2);
        }
        std::cout << static_cast<double>(clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;
        
        for(const auto fixedVI : data.fixedVert) {
//            hessian.insert(2 * fixedVI, 2 * fixedVI) = 1.0;
//            hessian.insert(2 * fixedVI + 1, 2 * fixedVI + 1) = 1.0;
            hessian.coeffRef(2 * fixedVI, 2 * fixedVI) += 1.0;
            hessian.coeffRef(2 * fixedVI + 1, 2 * fixedVI + 1) += 1.0;
        }
        
        hessian.makeCompressed();
    }
    
    void SymStretchEnergy::initStepSize(const TriangleSoup& data, const Eigen::VectorXd& searchDir, double& stepSize) const
    {
        double left = 1.0, right = 0.0;
        for(int triI = 0; triI < data.F.rows(); triI++)
        {
            const Eigen::Vector3i& triVInd = data.F.row(triI);
            
            const Eigen::Vector2d& U1 = data.V.row(triVInd[0]);
            const Eigen::Vector2d& U2 = data.V.row(triVInd[1]);
            const Eigen::Vector2d& U3 = data.V.row(triVInd[2]);
            
            const Eigen::Vector2d V1(searchDir[triVInd[0] * 2], searchDir[triVInd[0] * 2 + 1]);
            const Eigen::Vector2d V2(searchDir[triVInd[1] * 2], searchDir[triVInd[1] * 2 + 1]);
            const Eigen::Vector2d V3(searchDir[triVInd[2] * 2], searchDir[triVInd[2] * 2 + 1]);
            
            const Eigen::Vector2d U2m1 = U2 - U1;
            const Eigen::Vector2d U3m1 = U3 - U1;
            const Eigen::Vector2d V2m1 = V2 - V1;
            const Eigen::Vector2d V3m1 = V3 - V1;
            
            const double a = V2m1[0] * V3m1[1] - V2m1[1] * V3m1[0];
            const double b = U2m1[0] * V3m1[1] - U2m1[1] * V3m1[0] + V2m1[0] * U3m1[1] - V2m1[1] * U3m1[0];
            const double c = U2m1[0] * U3m1[1] - U2m1[1] * U3m1[0];
            assert(c > 0.0);
            const double delta = b * b - 4.0 * a * c;
            double bound = stepSize;
            if(a > 0.0) {
                if((b < 0.0) && (delta > 0.0)) {
                    const double r_left = (-b - sqrt(delta)) / 2.0 / a;
                    assert(r_left > 0.0);
                    const double r_right = (-b + sqrt(delta)) / 2.0 / a;
                    if(r_left < left) {
                        left = r_left;
                    }
                    if(r_right > right) {
                        right = r_right;
                    }
                }
            }
            else if(a < 0.0) {
                assert(delta > 0.0);
                bound = (-b - sqrt(delta)) / 2.0 / a;
            }
            else {
                if(b < 0.0) {
                    bound = -c / b;
                }
            }
            if(bound < stepSize) {
                stepSize = bound;
            }
        }
        
        if((stepSize < right) && (stepSize > left)) {
            stepSize = left;
        }
        assert(stepSize > 0.0);
    }
    
    void SymStretchEnergy::checkEnergyVal(const TriangleSoup& data) const
    {
        logFile << "check energyVal computation..." << std::endl;
        
        const double normalizer_div = data.surfaceArea;
        
        Eigen::VectorXd energyValPerTri;
        energyValPerTri.resize(data.F.rows());
        double err = 0.0;
        for(int triI = 0; triI < data.F.rows(); triI++) {
            const Eigen::Vector3i& triVInd = data.F.row(triI);
            
            const Eigen::Vector3d& P1 = data.V_rest.row(triVInd[0]);
            const Eigen::Vector3d& P2 = data.V_rest.row(triVInd[1]);
            const Eigen::Vector3d& P3 = data.V_rest.row(triVInd[2]);
            
            // fake isometric UV coordinates
            Eigen::Vector3d P[3] = { P1, P2, P3 };
            Eigen::Vector2d U[3]; IglUtils::mapTriangleTo2D(P, U);
            const Eigen::Vector2d U2m1 = U[1];//(P2m1.norm(), 0.0);
            const Eigen::Vector2d U3m1 = U[2];//(P3m1.dot(P2m1) / U2m1[0], P3m1.cross(P2m1).norm() / U2m1[0]);
            
            const double area_U = 0.5 * (U2m1[0] * U3m1[1] - U2m1[1] * U3m1[0]);
            logFile << "areas: " << data.triArea[triI] << ", " << area_U << std::endl;
            
            const double w = data.triArea[triI] / normalizer_div;
            energyValPerTri[triI] = w * (1.0 + data.triAreaSq[triI] / area_U / area_U) *
                ((U3m1.squaredNorm() * data.e0SqLen[triI] + U2m1.squaredNorm() * data.e1SqLen[triI]) / 4 / data.triAreaSq[triI] -
                U3m1.dot(U2m1) * data.e0dote1[triI] / 2 / data.triAreaSq[triI]);
            err += energyValPerTri[triI] - w * 4.0;
        }
        std::cout << "energyVal computation error = " << err << std::endl;
        logFile << "energyVal computation error = " << err << std::endl;
    }
    
    SymStretchEnergy::SymStretchEnergy(void) :
        Energy(true)
    {
        
    }
    
    void SymStretchEnergy::computeStressTensor(const Eigen::Vector3d v[3], const Eigen::Vector2d u[3], Eigen::Matrix2d& stressTensor)
    {
        Eigen::Matrix2d F;
        IglUtils::computeDeformationGradient(v, u, F);
        Eigen::Matrix2d FmT = (F.transpose()).inverse();
        stressTensor = F - FmT * FmT.transpose() * FmT;
    }
}
