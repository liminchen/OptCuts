//
//  SymDirichletEnergy.cpp
//  OptCuts
//
//  Created by Minchen Li on 9/3/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#include "SymDirichletEnergy.hpp"
#include "IglUtils.hpp"

#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>

#include <tbb/tbb.h>

#include <limits>
#include <fstream>
#include <cfloat>

extern std::ofstream logFile;

namespace OptCuts {
    
    void SymDirichletEnergy::getEnergyValPerElem(const TriMesh& data, Eigen::VectorXd& energyValPerElem, bool uniformWeight) const
    {
        const double normalizer_div = data.surfaceArea;
        
        energyValPerElem.resize(data.F.rows());
        for(int triI = 0; triI < data.F.rows(); triI++) {
            const Eigen::Vector3i& triVInd = data.F.row(triI);
            
            const Eigen::RowVector2d& U1 = data.V.row(triVInd[0]);
            const Eigen::RowVector2d& U2 = data.V.row(triVInd[1]);
            const Eigen::RowVector2d& U3 = data.V.row(triVInd[2]);
            
            const Eigen::RowVector2d U2m1 = U2 - U1;
            const Eigen::RowVector2d U3m1 = U3 - U1;
            
            const double area_U = 0.5 * (U2m1[0] * U3m1[1] - U2m1[1] * U3m1[0]);
            
            const double w = (uniformWeight ? 1.0 : (data.triArea[triI] / normalizer_div));
            energyValPerElem[triI] = w * (1.0 + data.triAreaSq[triI] / area_U / area_U) *
                ((U3m1.squaredNorm() * data.e0SqLen[triI] + U2m1.squaredNorm() * data.e1SqLen[triI]) / 4 / data.triAreaSq[triI] -
                U3m1.dot(U2m1) * data.e0dote1[triI] / 2 / data.triAreaSq[triI]);
        }
    }
    
    void SymDirichletEnergy::getEnergyValByElemID(const TriMesh& data, int elemI, double& energyVal, bool uniformWeight) const
    {
        const double normalizer_div = data.surfaceArea;
        
        int triI = elemI;
        const Eigen::Vector3i& triVInd = data.F.row(triI);
        
        const Eigen::Vector2d& U1 = data.V.row(triVInd[0]);
        const Eigen::Vector2d& U2 = data.V.row(triVInd[1]);
        const Eigen::Vector2d& U3 = data.V.row(triVInd[2]);
        
        const Eigen::Vector2d U2m1 = U2 - U1;
        const Eigen::Vector2d U3m1 = U3 - U1;
        
        const double area_U = 0.5 * (U2m1[0] * U3m1[1] - U2m1[1] * U3m1[0]);
        
        const double w = (uniformWeight ? 1.0 : (data.triArea[triI] / normalizer_div));
        energyVal = w * (1.0 + data.triAreaSq[triI] / area_U / area_U) *
        ((U3m1.squaredNorm() * data.e0SqLen[triI] + U2m1.squaredNorm() * data.e1SqLen[triI]) / 4 / data.triAreaSq[triI] -
         U3m1.dot(U2m1) * data.e0dote1[triI] / 2 / data.triAreaSq[triI]);
    }
    
    void SymDirichletEnergy::getEnergyValPerVert(const TriMesh& data, Eigen::VectorXd& energyValPerVert) const
    {
        Eigen::VectorXd energyValPerElem;
        getEnergyValPerElem(data, energyValPerElem);
        
        Eigen::VectorXd totalWeight;
        totalWeight.resize(data.V_rest.rows());
        totalWeight.setZero();
        energyValPerVert.resize(data.V_rest.rows());
        energyValPerVert.setZero();
        for(int triI = 0; triI < data.F.rows(); triI++) {
            for(int i = 0; i < 3; i++) {
                energyValPerVert[data.F(triI, i)] += energyValPerElem[triI];
                totalWeight[data.F(triI, i)] += data.triArea[triI];
                //TODO: verify the scale if the value will be used rather than the rank!
            }
        }
        for(int vI = 0; vI < data.V_rest.rows(); vI++) {
            energyValPerVert[vI] /= totalWeight[vI]; //!!! is normalization needed?
        }
    }
    
    void SymDirichletEnergy::getMaxUnweightedEnergyValPerVert(const TriMesh& data, Eigen::VectorXd& MaxUnweightedEnergyValPerVert) const
    {
        Eigen::VectorXd energyValPerElem;
        getEnergyValPerElem(data, energyValPerElem, true);

        MaxUnweightedEnergyValPerVert.resize(data.V_rest.rows());
        MaxUnweightedEnergyValPerVert.setZero();
        for(int triI = 0; triI < data.F.rows(); triI++) {
            for(int i = 0; i < 3; i++) {
                if(MaxUnweightedEnergyValPerVert[data.F(triI, i)] < energyValPerElem[triI]) {
                    MaxUnweightedEnergyValPerVert[data.F(triI, i)] = energyValPerElem[triI];
                }
            }
        }
    }
    
    void SymDirichletEnergy::computeDivGradPerVert(const TriMesh& data, Eigen::VectorXd& divGradPerVert) const
    {
        Eigen::MatrixXd localGradients;
        computeLocalGradient(data, localGradients);
        
#define STANDARD_DEVIATION_FILTERING 1
#ifdef STANDARD_DEVIATION_FILTERING
        //NOTE: no need to weight by area since gradient already contains area information
        //TODO: don't need to compute mean because it's always zero at stationary, go back to the simpler version?
        Eigen::MatrixXd mean = Eigen::MatrixXd::Zero(data.V_rest.rows(), 2);
        Eigen::VectorXi incTriAmt = Eigen::VectorXi::Zero(data.V_rest.rows());
        for(int triI = 0; triI < data.F.rows(); triI++) {
            const Eigen::RowVector3i& triVInd = data.F.row(triI);
            int locGradStartInd = triI * 3;
            for(int i = 0; i < 3; i++) {
                mean.row(triVInd[i]) += localGradients.row(locGradStartInd + i);
                incTriAmt[triVInd[i]]++;
            }
        }
        for(int vI = 0; vI < data.V_rest.rows(); vI++) {
            mean.row(vI) /= incTriAmt[vI];
        }
        
        Eigen::VectorXd standardDeviation = Eigen::VectorXd::Zero(data.V_rest.rows());
        for(int triI = 0; triI < data.F.rows(); triI++) {
            const Eigen::RowVector3i& triVInd = data.F.row(triI);
            int locGradStartInd = triI * 3;
            for(int i = 0; i < 3; i++) {
                standardDeviation[triVInd[i]] += (localGradients.row(locGradStartInd + i) - mean.row(triVInd[i])).squaredNorm();
            }
        }
        
        divGradPerVert = Eigen::VectorXd::Zero(data.V_rest.rows());
        for(int vI = 0; vI < data.V_rest.rows(); vI++) {
            if(incTriAmt[vI] == 1) {
                // impossible to be splitted
                divGradPerVert[vI] = 0.0;
            }
            else {
                divGradPerVert[vI] = std::sqrt(standardDeviation[vI] / (incTriAmt[vI] - 1.0));
            }
        }
#else
        divGradPerVert = Eigen::VectorXd::Zero(data.V_rest.rows());
        for(int triI = 0; triI < data.F.rows(); triI++) {
            const Eigen::RowVector3i& triVInd = data.F.row(triI);
            const Eigen::RowVector2d eDir[3] = {
                (data.V.row(triVInd[1]) - data.V.row(triVInd[0])).normalized(),
                (data.V.row(triVInd[2]) - data.V.row(triVInd[1])).normalized(),
                (data.V.row(triVInd[0]) - data.V.row(triVInd[2])).normalized()
//                (data.V.row(triVInd[1]) - data.V.row(triVInd[0])),
//                (data.V.row(triVInd[2]) - data.V.row(triVInd[1])),
//                (data.V.row(triVInd[0]) - data.V.row(triVInd[2]))
            };
            int locGradStartInd = triI * 3;
            for(int i = 0; i < 3; i++) {
                const Eigen::RowVector2d centralDir = (eDir[i] - eDir[(i+2) % 3]).normalized();
//                const Eigen::RowVector2d centralDir(eDir[(i+1) % 3][1], -eDir[(i+1) % 3][0]);
                const double w = std::acos(std::max(-1.0, std::min(1.0, eDir[i].dot(-eDir[(i+2) % 3]))));
                divGradPerVert[triVInd[i]] += w * -localGradients.row(locGradStartInd + i).dot(centralDir);
//                divGradPerVert[triVInd[i]] += -localGradients.row(locGradStartInd + i).dot(centralDir);
                
                //TODO: when querying boundary verts, no need to compute for interior verts
                //TODO: if want to compare boundary with interior, w needs to be /pi or /2pi
            }
        }
        
//        double minDiv = divGradPerVert.minCoeff();
//        minDiv -= std::abs(minDiv) * 1.0e-3;
        for(int vI = 0; vI < data.V_rest.rows(); vI++) {
            // prefer to split stretched regions:
//            if(data.vNeighbor[vI].size() <= 2) {
//                // impossible to be splitted
//                divGradPerVert[vI] = minDiv;
//            }
            
            // if want to split both compressed and stretched regions:
            if(data.vNeighbor[vI].size() <= 2) {
                // impossible to be splitted
                divGradPerVert[vI] = 0.0;
            }
            else {
                divGradPerVert[vI] = std::abs(divGradPerVert[vI]);
            }
        }
#endif
    }
    
    void SymDirichletEnergy::getDivGradPerElem(const TriMesh& data, Eigen::VectorXd& divGradPerElem) const
    {
        Eigen::VectorXd divGrad_vert;
        computeDivGradPerVert(data, divGrad_vert);
        
        // filter out interior vertices for visualizing boundary verts
//        for(int vI = 0; vI < divGrad_vert.size(); vI++) {
//            if(!data.isBoundaryVert(data.edge2Tri, data.vNeighbor, vI)) {
//                divGrad_vert[vI] = 0.0;
//            }
//        }
        
        divGradPerElem.resize(data.F.rows());
        for(int triI = 0; triI < data.F.rows(); triI++) {
            const Eigen::RowVector3i& triVInd = data.F.row(triI);
            divGradPerElem[triI] = (divGrad_vert[triVInd[0]] + divGrad_vert[triVInd[1]] + divGrad_vert[triVInd[2]]) / 3.0;
        }
    }
    
    void SymDirichletEnergy::computeLocalGradient(const TriMesh& data, Eigen::MatrixXd& localGradients) const
    {
        const double normalizer_div = data.surfaceArea;
        
        localGradients.resize(data.F.rows() * 3, 2);
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
            const int startRowI = triI * 3;
            
            const Eigen::Vector2d edge_oppo1 = U3 - U2;
            const Eigen::Vector2d dLeft1 = areaRatio * Eigen::Vector2d(edge_oppo1[1], -edge_oppo1[0]);
            const Eigen::Vector2d dRight1 = ((data.e0dote1[triI] - data.e0SqLen[triI]) * U3m1 +
                                             (data.e0dote1[triI] - data.e1SqLen[triI]) * U2m1) / 2.0 / data.triAreaSq[triI];
            localGradients.row(startRowI) = w * (dLeft1 * rightTerm + dRight1 * leftTerm);
            
            const Eigen::Vector2d edge_oppo2 = U1 - U3;
            const Eigen::Vector2d dLeft2 = areaRatio * Eigen::Vector2d(edge_oppo2[1], -edge_oppo2[0]);
            const Eigen::Vector2d dRight2 = (data.e1SqLen[triI] * U2m1 - data.e0dote1[triI] * U3m1) / 2.0 / data.triAreaSq[triI];
            localGradients.row(startRowI + 1) = w * (dLeft2 * rightTerm + dRight2 * leftTerm);
            
            const Eigen::Vector2d edge_oppo3 = U2 - U1;
            const Eigen::Vector2d dLeft3 = areaRatio * Eigen::Vector2d(edge_oppo3[1], -edge_oppo3[0]);
            const Eigen::Vector2d dRight3 = (data.e0SqLen[triI] * U3m1 - data.e0dote1[triI] * U2m1) / 2.0 / data.triAreaSq[triI];
            localGradients.row(startRowI + 2) = w * (dLeft3 * rightTerm + dRight3 * leftTerm);
        }
    }
    
    void SymDirichletEnergy::computeGradient(const TriMesh& data, Eigen::VectorXd& gradient, bool uniformWeight) const
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
            const double w = (uniformWeight ? 1.0 : (data.triArea[triI] / normalizer_div));
            
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
    
    void SymDirichletEnergy::computeHessian(const TriMesh& data,
                                          Eigen::MatrixXd& Hessian,
                                          bool uniformWeight) const
    {
        const double normalizer_div = data.surfaceArea;
        
        Hessian.resize(data.V.rows() * 2, data.V.rows() * 2);
        Hessian.setZero();
        
        std::vector<Eigen::Matrix<double, 6, 6>> triHessians(data.F.rows());
        std::vector<Eigen::VectorXi> vInds(data.F.rows());
        tbb::parallel_for(0, (int)data.F.rows(), 1, [&](int triI) {
            //        for(int triI = 0; triI < data.F.rows(); triI++) {
            const Eigen::Vector3i& triVInd = data.F.row(triI);
            
            const Eigen::Vector2d& U1 = data.V.row(triVInd[0]);
            const Eigen::Vector2d& U2 = data.V.row(triVInd[1]);
            const Eigen::Vector2d& U3 = data.V.row(triVInd[2]);
            
            const Eigen::Vector2d U2m1 = U2 - U1;
            const Eigen::Vector2d U3m1 = U3 - U1;
            
            const double area_U = 0.5 * (U2m1[0] * U3m1[1] - U2m1[1] * U3m1[0]);
            const double areaRatio = data.triAreaSq[triI] / area_U / area_U / area_U;
            const double dAreaRatio_div_dArea_mult = 3.0 / 2.0 * areaRatio / area_U;
            
            const double w = (uniformWeight ? 1.0 : (data.triArea[triI] / normalizer_div));
            
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
            
            Eigen::Matrix<double, 6, 6>& curHessian = triHessians[triI];
            
            // compute second order derivatives for g_U1
            const Eigen::Matrix2d d2Left11 = dAreaRatio_div_dArea_mult * edge_oppo1_Ortho * edge_oppo1_Ortho.transpose();
            const double d2Right11 = (e0SqLen_div_dbAreaSq + e1SqLen_div_dbAreaSq - 2.0 * e0dote1_div_dbAreaSq);
            const Eigen::Matrix2d dLeft1dRight1T = dLeft1 * dRight1.transpose();
            curHessian.block(0, 0, 2, 2) = w * (d2Left11 * rightTerm + dLeft1dRight1T +
                                                d2Right11 * leftTerm * Eigen::Matrix2d::Identity() + dLeft1dRight1T.transpose());
            
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
            
            // project to nearest SPD matrix
            IglUtils::makePD(curHessian);
            
            Eigen::VectorXi& vInd = vInds[triI];
            vInd = triVInd;
            for(int vI = 0; vI < 3; vI++) {
                if(data.fixedVert.find(vInd[vI]) != data.fixedVert.end()) {
                    vInd[vI] = -1;
                }
            }
            //        }
        });
        for(int triI = 0; triI < data.F.rows(); triI++) {
            IglUtils::addBlockToMatrix(triHessians[triI], vInds[triI], 2, Hessian);
        }
        
        Eigen::VectorXi fixedVertInd;
        fixedVertInd.resize(data.fixedVert.size());
        int fVI = 0;
        for(const auto fixedVI : data.fixedVert) {
            fixedVertInd[fVI++] = fixedVI;
        }
        IglUtils::addDiagonalToMatrix(Eigen::VectorXd::Ones(data.fixedVert.size() * 2),
                                      fixedVertInd, 2, Hessian);
    }

    void SymDirichletEnergy::computeHessian(const TriMesh& data, Eigen::VectorXd* V,
                                            Eigen::VectorXi* I, Eigen::VectorXi* J, bool uniformWeight) const
    {
        const double normalizer_div = data.surfaceArea;
        
//        std::cout << "computing entry value..." << std::endl;
//        clock_t start = clock();
        std::vector<Eigen::Matrix<double, 6, 6>> triHessians(data.F.rows());
        std::vector<Eigen::VectorXi> vInds(data.F.rows());
        tbb::parallel_for(0, (int)data.F.rows(), 1, [&](int triI) {
//        for(int triI = 0; triI < data.F.rows(); triI++) {
            const Eigen::Vector3i& triVInd = data.F.row(triI);
            
            const Eigen::Vector2d& U1 = data.V.row(triVInd[0]);
            const Eigen::Vector2d& U2 = data.V.row(triVInd[1]);
            const Eigen::Vector2d& U3 = data.V.row(triVInd[2]);
            
            const Eigen::Vector2d U2m1 = U2 - U1;
            const Eigen::Vector2d U3m1 = U3 - U1;
            
            const double area_U = 0.5 * (U2m1[0] * U3m1[1] - U2m1[1] * U3m1[0]);
            const double areaRatio = data.triAreaSq[triI] / area_U / area_U / area_U;
            const double dAreaRatio_div_dArea_mult = 3.0 / 2.0 * areaRatio / area_U;
            
            const double w = (uniformWeight ? 1.0 : (data.triArea[triI] / normalizer_div));
            
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
            
            Eigen::Matrix<double, 6, 6>& curHessian = triHessians[triI];
            
            // compute second order derivatives for g_U1
            const Eigen::Matrix2d d2Left11 = dAreaRatio_div_dArea_mult * edge_oppo1_Ortho * edge_oppo1_Ortho.transpose();
            const double d2Right11 = (e0SqLen_div_dbAreaSq + e1SqLen_div_dbAreaSq - 2.0 * e0dote1_div_dbAreaSq);
            const Eigen::Matrix2d dLeft1dRight1T = dLeft1 * dRight1.transpose();
            curHessian.block(0, 0, 2, 2) = w * (d2Left11 * rightTerm + dLeft1dRight1T +
                                                d2Right11 * leftTerm * Eigen::Matrix2d::Identity() + dLeft1dRight1T.transpose());
        
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
            
            // project to nearest SPD matrix
            IglUtils::makePD(curHessian);
            
            Eigen::VectorXi& vInd = vInds[triI];
            vInd = triVInd;
            for(int vI = 0; vI < 3; vI++) {
                if(data.fixedVert.find(vInd[vI]) != data.fixedVert.end()) {
                    vInd[vI] = -1;
                }
            }
//        }
        });
        for(int triI = 0; triI < data.F.rows(); triI++) {
            IglUtils::addBlockToMatrix(triHessians[triI], vInds[triI], 2, V, I, J);
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
    
    void SymDirichletEnergy::initStepSize(const TriMesh& data, const Eigen::VectorXd& searchDir, double& stepSize) const
    {
        assert(stepSize > 0.0);
        
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
                if((b < 0.0) && (delta >= 0.0)) {
                    bound = 2.0 * c / (-b + sqrt(delta));
                    // (same in math as (-b - sqrt(delta)) / 2.0 / a
                    //  but smaller numerical error when b < 0.0)
                    assert(bound > 0.0);
                }
            }
            else if(a < 0.0) {
                assert(delta > 0.0);
                if(b < 0.0) {
                    bound = 2.0 * c / (-b + sqrt(delta));
                    // (same in math as (-b - sqrt(delta)) / 2.0 / a
                    //  but smaller numerical error when b < 0.0)
                }
                else {
                    bound = (-b - sqrt(delta)) / 2.0 / a;
                }
                assert(bound > 0.0);
            }
            else {
                if(b < 0.0) {
                    bound = -c / b;
                    assert(bound > 0.0);
                }
            }
            
            if(bound < stepSize) {
                stepSize = bound;
            }
        }
    }
    
    void SymDirichletEnergy::checkEnergyVal(const TriMesh& data) const
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
    
    SymDirichletEnergy::SymDirichletEnergy(void) :
        Energy(true)
    {
        
    }
}
