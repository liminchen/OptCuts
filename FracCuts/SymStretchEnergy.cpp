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

#include <limits>
#include <fstream>

extern std::ofstream logFile;

namespace FracCuts {
    
    void SymStretchEnergy::computeEnergyVal(const TriangleSoup& data, double& energyVal) const
    {
        //TODO: precomputation of quantities related to P
        
        energyVal = 0.0;
        for(int triI = 0; triI < data.F.rows(); triI++) {
            const Eigen::Vector3i& triVInd = data.F.row(triI);
            
            const Eigen::Vector3d& P1 = data.V_rest.row(triVInd[0]);
            const Eigen::Vector3d& P2 = data.V_rest.row(triVInd[1]);
            const Eigen::Vector3d& P3 = data.V_rest.row(triVInd[2]);
            const Eigen::Vector2d& U1 = data.V.row(triVInd[0]);
            const Eigen::Vector2d& U2 = data.V.row(triVInd[1]);
            const Eigen::Vector2d& U3 = data.V.row(triVInd[2]);
            
            const Eigen::Vector3d P2m1 = P2 - P1;
            const Eigen::Vector3d P3m1 = P3 - P1;
            const Eigen::Vector2d U2m1 = U2 - U1;
            const Eigen::Vector2d U3m1 = U3 - U1;
            
            const double area_P = 0.5 * P2m1.cross(P3m1).norm();
            const Eigen::Vector3d U2m1_3D(U2m1[0], U2m1[1], 0.0);
            const Eigen::Vector3d U3m1_3D(U3m1[0], U3m1[1], 0.0);
            const double area_U = 0.5 * U2m1_3D.cross(U3m1_3D).norm();
            
            const double w = area_P;
            energyVal += w * (1.0 + area_P * area_P / area_U / area_U) *
                ((U3m1.squaredNorm() * P2m1.squaredNorm() + U2m1.squaredNorm() * P3m1.squaredNorm()) / 4 / area_P / area_P -
                U3m1.dot(U2m1) * P3m1.dot(P2m1) / 2 / area_P / area_P);
        }
    }
    
    void SymStretchEnergy::checkEnergyVal(const TriangleSoup& data) const
    {
        logFile << "check energyVal computation..." << std::endl;
        
        Eigen::VectorXd energyValPerTri;
        energyValPerTri.resize(data.F.rows());
        double err = 0.0;
        for(int triI = 0; triI < data.F.rows(); triI++) {
            const Eigen::Vector3i& triVInd = data.F.row(triI);
            
            const Eigen::Vector3d& P1 = data.V_rest.row(triVInd[0]);
            const Eigen::Vector3d& P2 = data.V_rest.row(triVInd[1]);
            const Eigen::Vector3d& P3 = data.V_rest.row(triVInd[2]);
            
            const Eigen::Vector3d P2m1 = P2 - P1;
            const Eigen::Vector3d P3m1 = P3 - P1;
            
            // fake isotropic UV coordinates
            Eigen::Vector3d P[3] = { P1, P2, P3 };
            Eigen::Vector2d U[3]; IglUtils::mapTriangleTo2D(P, U);
            const Eigen::Vector2d U2m1 = U[1];//(P2m1.norm(), 0.0);
            const Eigen::Vector2d U3m1 = U[2];//(P3m1.dot(P2m1) / U2m1[0], P3m1.cross(P2m1).norm() / U2m1[0]);
            
            const double area_P = 0.5 * P2m1.cross(P3m1).norm();
            const Eigen::Vector3d U2m1_3D(U2m1[0], U2m1[1], 0.0);
            const Eigen::Vector3d U3m1_3D(U3m1[0], U3m1[1], 0.0);
            const double area_U = 0.5 * U2m1_3D.cross(U3m1_3D).norm();
            logFile << "areas: " << area_P << ", " << area_U << std::endl;
            
            const double w = area_P;
            energyValPerTri[triI] = w * (1.0 + area_P * area_P / area_U / area_U) *
                ((U3m1.squaredNorm() * P2m1.squaredNorm() + U2m1.squaredNorm() * P3m1.squaredNorm()) / 4 / area_P / area_P -
                U3m1.dot(U2m1) * P3m1.dot(P2m1) / 2 / area_P / area_P);
            err += energyValPerTri[triI] - w * 4.0;
        }
        std::cout << "energyVal computation error = " << err << std::endl;
        logFile << "energyVal computation error = " << err << std::endl;
    }
    
    void SymStretchEnergy::computeGradient(const TriangleSoup& data, Eigen::VectorXd& gradient) const
    {
        //TODO: precomputation of quantities related to P
        
        Eigen::MatrixXd cotVals;
        igl::cotmatrix_entries(data.V_rest, data.F, cotVals);
        
        gradient.resize(data.V.rows() * 2);
        gradient.setZero();
        for(int triI = 0; triI < data.F.rows(); triI++) {
            const Eigen::Vector3i& triVInd = data.F.row(triI);
            
            const Eigen::Vector3d& P1 = data.V_rest.row(triVInd[0]);
            const Eigen::Vector3d& P2 = data.V_rest.row(triVInd[1]);
            const Eigen::Vector3d& P3 = data.V_rest.row(triVInd[2]);
            const Eigen::Vector2d& U1 = data.V.row(triVInd[0]);
            const Eigen::Vector2d& U2 = data.V.row(triVInd[1]);
            const Eigen::Vector2d& U3 = data.V.row(triVInd[2]);
            
            const Eigen::Vector3d P2m1 = P2 - P1;
            const Eigen::Vector3d P3m1 = P3 - P1;
            const Eigen::Vector2d U2m1 = U2 - U1;
            const Eigen::Vector2d U3m1 = U3 - U1;
            
            const double area_P = 0.5 * P2m1.cross(P3m1).norm();
            const Eigen::Vector3d U2m1_3D(U2m1[0], U2m1[1], 0.0);
            const Eigen::Vector3d U3m1_3D(U3m1[0], U3m1[1], 0.0);
            const double area_U = 0.5 * U2m1_3D.cross(U3m1_3D).norm();
            
            const double leftTerm = 1.0 + area_P * area_P / area_U / area_U;
            const double rightTerm = (U3m1.squaredNorm() * P2m1.squaredNorm() + U2m1.squaredNorm() * P3m1.squaredNorm()) / 4 / area_P / area_P -
                U3m1.dot(U2m1) * P3m1.dot(P2m1) / 2 / area_P / area_P;
            
            const double areaRatio = area_P * area_P / area_U / area_U / area_U;
            const double w = area_P;
            
            const Eigen::Vector2d edge_oppo1 = U3 - U2;
            const Eigen::Vector2d dLeft1 = areaRatio * Eigen::Vector2d(edge_oppo1[1], -edge_oppo1[0]);
            const Eigen::Vector2d dRight1 = ((P3m1.dot(P2m1) - P2m1.squaredNorm()) * U3m1 + (P3m1.dot(P2m1) - P3m1.squaredNorm()) * U2m1) / 2.0 / area_P / area_P;
            gradient.block(triVInd[0] * 2, 0, 2, 1) += w * (dLeft1 * rightTerm + dRight1 * leftTerm);
            
            const Eigen::Vector2d edge_oppo2 = U1 - U3;
            const Eigen::Vector2d dLeft2 = areaRatio * Eigen::Vector2d(edge_oppo2[1], -edge_oppo2[0]);
            const Eigen::Vector2d dRight2 = (P3m1.squaredNorm() * U2m1 - P3m1.dot(P2m1) * U3m1) / 2.0 / area_P / area_P;
            gradient.block(triVInd[1] * 2, 0, 2, 1) += w * (dLeft2 * rightTerm + dRight2 * leftTerm);
            
            const Eigen::Vector2d edge_oppo3 = U2 - U1;
            const Eigen::Vector2d dLeft3 = areaRatio * Eigen::Vector2d(edge_oppo3[1], -edge_oppo3[0]);
            const Eigen::Vector2d dRight3 = (P2m1.squaredNorm() * U3m1 - P3m1.dot(P2m1) * U2m1) / 2.0 / area_P / area_P;
            gradient.block(triVInd[2] * 2, 0, 2, 1) += w * (dLeft3 * rightTerm + dRight3 * leftTerm);
        }
    }
    
    void SymStretchEnergy::computePrecondMtr(const TriangleSoup& data, Eigen::SparseMatrix<double>& precondMtr) const
    {
//        Eigen::SparseMatrix<double> M;
//        massmatrix(data.V_rest, data.F, igl::MASSMATRIX_TYPE_DEFAULT, M);
        Eigen::SparseMatrix<double> L;
        igl::cotmatrix(data.V_rest, data.F, L);
        precondMtr.resize(data.V.rows() * 2, data.V.rows() * 2);
        for (int k = 0; k < L.outerSize(); ++k)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it)
            {
                precondMtr.insert(it.row() * 2, it.col() * 2) = -it.value();// * M.coeffRef(it.row(), it.row());
                precondMtr.insert(it.row() * 2 + 1, it.col() * 2 + 1) = -it.value();// * M.coeffRef(it.row(), it.row());
            }
        }
        precondMtr.makeCompressed();
    }
    
    void SymStretchEnergy::lineSearch(const TriangleSoup& data, const Eigen::VectorXd& searchDir, double& stepSize)
    {
        stepSize = 1.0;
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
            const double delta = b * b - 4.0 * a * c;
            double bound = stepSize;
            if(a > 0.0) {
                if((b < 0.0) && (delta > 0.0)) {
                    const double r_left = (-b - sqrt(delta)) / 2.0 / a;
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
    }
    
    void SymStretchEnergy::computeHessian(const TriangleSoup& data, Eigen::SparseMatrix<double>& hessian) const
    {
        assert(0 && "no hessian computation for this energy");
    }
    
    
    void SymStretchEnergy::getEnergyValPerElem(const TriangleSoup& data, Eigen::VectorXd& energyValPerElem) const
    {
        //TODO:
    }
}
