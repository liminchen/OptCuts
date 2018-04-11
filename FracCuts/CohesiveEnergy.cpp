//
//  CohesiveEnergy.cpp
//  FracCuts
//
//  Created by Minchen Li on 9/20/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#include "CohesiveEnergy.hpp"
#include "IglUtils.hpp"

namespace FracCuts
{
    
    void CohesiveEnergy::getEnergyValPerElem(const TriangleSoup& data, Eigen::VectorXd& energyValPerElem, bool uniformWeight) const
    {
        energyValPerElem.resize(data.cohE.rows());
        for(int cohI = 0; cohI < data.cohE.rows(); cohI++)
        {
            if(data.boundaryEdge[cohI]) {
                energyValPerElem[cohI] = 0.0;
            }
            else {
                const Eigen::RowVector2d& xa = data.V.row(data.cohE(cohI, 0));
                const Eigen::RowVector2d& xb = data.V.row(data.cohE(cohI, 1));
                const Eigen::RowVector2d& xc = data.V.row(data.cohE(cohI, 2));
                const Eigen::RowVector2d& xd = data.V.row(data.cohE(cohI, 3));
                
                Eigen::Matrix2d P = alpha * Eigen::Matrix2d::Identity();
                const Eigen::RowVector2d xhat_bn = xa + xc - xb - xd;
                const double xhat_bn_norm = xhat_bn.norm();
                if(xhat_bn_norm > 1.0e-3 * data.avgEdgeLen) {
                    const Eigen::RowVector2d xhat = xhat_bn / xhat_bn_norm;
                    P += ((1.0 - 2.0 * alpha) * xhat.transpose()) * xhat;
                }
                Eigen::RowVector4d difVec; difVec << xa - xc, xb - xd;
                Eigen::Matrix4d m; m << P, P / 2.0, P / 2.0, P;
                const double w = (uniformWeight ? 1.0 : (data.edgeLen[cohI] / 3.0));
                
                energyValPerElem[cohI] = w * lambda * difVec * m * difVec.transpose();
                if(energyValPerElem[cohI] > tau) {
                    energyValPerElem[cohI] = tau;
                }
            }
        }
    }
    
    void CohesiveEnergy::computeGradient(const TriangleSoup& data, Eigen::VectorXd& gradient, bool uniformWeight) const
    {
        gradient.resize(data.V.rows() * 2);
        gradient.setZero();
        for(int cohI = 0; cohI < data.cohE.rows(); cohI++)
        {
            if(!data.boundaryEdge[cohI]) {
                const Eigen::RowVector2d& xa = data.V.row(data.cohE(cohI, 0));
                const Eigen::RowVector2d& xb = data.V.row(data.cohE(cohI, 1));
                const Eigen::RowVector2d& xc = data.V.row(data.cohE(cohI, 2));
                const Eigen::RowVector2d& xd = data.V.row(data.cohE(cohI, 3));
                
                Eigen::Matrix2d P = alpha * Eigen::Matrix2d::Identity();
                const Eigen::RowVector2d xhat_bn = xa + xc - xb - xd;
                const double xhat_bn_norm = xhat_bn.norm();
                Eigen::RowVector2d xhat;
                bool dropxxT = true;
                if(xhat_bn_norm > 1.0e-3 * data.avgEdgeLen) {
                    dropxxT = false;
                    xhat = xhat_bn / xhat_bn_norm;
                    P += ((1.0 - 2.0 * alpha) * xhat.transpose()) * xhat;
                }
                
                Eigen::RowVector4d difVec; difVec << xa - xc, xb - xd;
                Eigen::Matrix4d m; m << P, P / 2.0, P / 2.0, P;
                const double w = (uniformWeight ? 1.0 : (data.edgeLen[cohI] / 3.0));
                if(w * lambda * difVec * m * difVec.transpose() > tau) {
                    continue;
                }
                
                Eigen::Vector4d dd_div_dv = w * 2.0 * lambda * m * difVec.transpose();
                gradient.block(data.cohE(cohI, 0) * 2, 0, 2, 1) += dd_div_dv.block(0, 0, 2, 1);
                gradient.block(data.cohE(cohI, 1) * 2, 0, 2, 1) += dd_div_dv.block(2, 0, 2, 1);
                gradient.block(data.cohE(cohI, 2) * 2, 0, 2, 1) -= dd_div_dv.block(0, 0, 2, 1);
                gradient.block(data.cohE(cohI, 3) * 2, 0, 2, 1) -= dd_div_dv.block(2, 0, 2, 1);
                
                if(!dropxxT) {
                    Eigen::Matrix<Eigen::RowVector2d, 2, 2> dP_div_dxhat;
                    IglUtils::differentiate_xxT(xhat, dP_div_dxhat, 1.0 - 2.0 * alpha);
                    Eigen::Matrix2d dxhat_div_dx_base;
                    IglUtils::differentiate_normalize(xhat_bn, dxhat_div_dx_base);
                    Eigen::Matrix<Eigen::RowVector2d, 2, 2> dP_div_dx_base;
                    for(int rowI = 0; rowI < 2; rowI++) {
                        for(int colI = 0; colI < 2; colI++) {
                            dP_div_dx_base(rowI, colI) = dP_div_dxhat(rowI, colI) * dxhat_div_dx_base;
                        }
                    }
                    Eigen::Vector2d dd_div_dx_M_base;
                    compute_dd_div_dx_M(difVec, dP_div_dx_base, dd_div_dx_M_base, w);
                    
                    gradient.block(data.cohE(cohI, 0) * 2, 0, 2, 1) += dd_div_dx_M_base;
                    gradient.block(data.cohE(cohI, 1) * 2, 0, 2, 1) -= dd_div_dx_M_base;
                    gradient.block(data.cohE(cohI, 2) * 2, 0, 2, 1) += dd_div_dx_M_base;
                    gradient.block(data.cohE(cohI, 3) * 2, 0, 2, 1) -= dd_div_dx_M_base;
                }
            }
        }
        for(const auto fixedVI : data.fixedVert) {
            gradient[2 * fixedVI] = 0.0;
            gradient[2 * fixedVI + 1] = 0.0;
        }
    }
    
    void CohesiveEnergy::computePrecondMtr(const TriangleSoup& data, Eigen::SparseMatrix<double>& precondMtr, bool uniformWeight) const
    {
        precondMtr.resize(data.V.rows() * 2, data.V.rows() * 2);
        precondMtr.reserve(data.V.rows() * 5 * 4);
        precondMtr.setZero();
        for(int cohI = 0; cohI < data.cohE.rows(); cohI++)
        {
            if(!data.boundaryEdge[cohI]) {
                const Eigen::RowVector2d& xa = data.V.row(data.cohE(cohI, 0));
                const Eigen::RowVector2d& xb = data.V.row(data.cohE(cohI, 1));
                const Eigen::RowVector2d& xc = data.V.row(data.cohE(cohI, 2));
                const Eigen::RowVector2d& xd = data.V.row(data.cohE(cohI, 3));
                
                Eigen::Matrix2d P = alpha * Eigen::Matrix2d::Identity();
                const Eigen::RowVector2d xhat_bn = xa + xc - xb - xd;
                const double xhat_bn_norm = xhat_bn.norm();
                if(xhat_bn_norm > 1.0e-3 * data.avgEdgeLen) {
                    const Eigen::RowVector2d xhat = xhat_bn / xhat_bn_norm;
                    P += ((1.0 - 2.0 * alpha) * xhat.transpose()) * xhat;
                }
                Eigen::RowVector4d difVec; difVec << xa - xc, xb - xd;
                Eigen::Matrix4d m; m << P, P / 2.0, P / 2.0, P;
                const double w = (uniformWeight ? 1.0 : (data.edgeLen[cohI] / 3.0));
                
                //                if(w * lambda * difVec * m * difVec.transpose() > tau) {
                //                    continue;
                //                }
                
                const Eigen::Matrix4d wl2m = w * lambda * 2.0 * m;
                Eigen::MatrixXd proxy;
                proxy.resize(8, 8);
                proxy << wl2m, -wl2m, -wl2m, wl2m;
                
                bool fixed[4];
                int fixedAmt = 0;
                for(int vI = 0; vI < 4; vI++) {
                    if(data.fixedVert.find(data.cohE(cohI, vI)) != data.fixedVert.end()) {
                        fixed[vI] = true;
                        fixedAmt++;
                    }
                    else {
                        fixed[vI] = false;
                    }
                }
                if(fixedAmt == 0) {
                    IglUtils::addBlockToMatrix(precondMtr, proxy, data.cohE.row(cohI), 2);
                }
                else {
                    Eigen::MatrixXd proxy_sub;
                    proxy_sub.resize(2 * (4 - fixedAmt), 2 * (4 - fixedAmt));
                    Eigen::VectorXi index;
                    index.resize(4 - fixedAmt);
                    int curBRowI = 0;
                    for(int blockRowI = 0; blockRowI < 4; blockRowI++) {
                        if(fixed[blockRowI]) {
                            continue;
                        }
                        
                        int curBColI = 0;
                        for(int blockColI = 0; blockColI < 4; blockColI++) {
                            if(fixed[blockColI]) {
                                continue;
                            }
                            
                            proxy_sub.block(curBRowI * 2, curBColI * 2, 2, 2) = proxy.block(blockRowI * 2, blockColI * 2, 2, 2);
                            curBColI++;
                        }
                        index[curBRowI] = data.cohE(cohI, blockRowI);
                        curBRowI++;
                    }
                    IglUtils::addBlockToMatrix(precondMtr, proxy_sub, index, 2);
                }
            }
        }
        for(const auto fixedVI : data.fixedVert) {
            precondMtr.insert(2 * fixedVI, 2 * fixedVI) = 1.0;
            precondMtr.insert(2 * fixedVI + 1, 2 * fixedVI + 1) = 1.0;
        }
        precondMtr.makeCompressed();
    }
    
    void CohesiveEnergy::computeHessian(const TriangleSoup& data, Eigen::SparseMatrix<double>& hessian, bool uniformWeight) const
    {
        assert(0 && "no hessian computation for this energy");
    }
    
    void CohesiveEnergy::checkEnergyVal(const TriangleSoup& data) const
    {
        
    }
    
    CohesiveEnergy::CohesiveEnergy(double avgEdgeLen, double p_tau_param, double p_alpha, double p_lambda) :
    tau_param(p_tau_param), alpha(p_alpha), lambda(p_lambda), Energy(false)
    {
        tau_base = lambda / 3.0 * avgEdgeLen * avgEdgeLen * avgEdgeLen;
        tau = tau_param * tau_base;
    }
    
    void CohesiveEnergy::compute_dd_div_dx_M(const Eigen::RowVector4d& difVec, const Eigen::Matrix<Eigen::RowVector2d, 2, 2>& dP_div_dx,
                                             Eigen::Vector2d& result, double elemWeight) const
    {
        result.setZero();
        const Eigen::Matrix4d vvT = difVec.transpose() * difVec;
        for(int rowI = 0; rowI < 4; rowI++) {
            for(int colI = 0; colI < 4; colI++) {
                double w = 1.0;
                if(rowI / 2 != colI / 2) {
                    w = 0.5;
                }
                result += w * vvT(rowI, colI) * dP_div_dx(rowI % 2, colI % 2).transpose();
            }
        }
        result *= elemWeight * lambda;
    }
    
}
