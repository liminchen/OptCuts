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
                
                const Eigen::RowVector2d xhat = (xa + xc - xb - xd).normalized();
                const Eigen::Matrix2d P = alpha * Eigen::Matrix2d::Identity() + ((1.0 - 2.0 * alpha) * xhat.transpose()) * xhat;
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
    
    void CohesiveEnergy::computeGradient(const TriangleSoup& data, Eigen::VectorXd& gradient) const
    {
        //TODO: fix vert
        
        gradient.resize(data.V.rows() * 2);
        gradient.setZero();
        for(int cohI = 0; cohI < data.cohE.rows(); cohI++)
        {
            if(!data.boundaryEdge[cohI]) {
                const Eigen::RowVector2d& xa = data.V.row(data.cohE(cohI, 0));
                const Eigen::RowVector2d& xb = data.V.row(data.cohE(cohI, 1));
                const Eigen::RowVector2d& xc = data.V.row(data.cohE(cohI, 2));
                const Eigen::RowVector2d& xd = data.V.row(data.cohE(cohI, 3));
                
                const Eigen::RowVector2d xhat_bn = xa + xc - xb - xd;
                const Eigen::RowVector2d xhat = xhat_bn.normalized();
                const Eigen::Matrix2d P = alpha * Eigen::Matrix2d::Identity() + ((1.0 - 2.0 * alpha) * xhat.transpose()) * xhat;
                Eigen::RowVector4d difVec; difVec << xa - xc, xb - xd;
                Eigen::Matrix4d m; m << P, P / 2.0, P / 2.0, P;
                const double w = data.edgeLen[cohI] / 3.0;
                
                if(w * lambda * difVec * m * difVec.transpose() > tau) {
                    continue;
                }
                
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
                
                Eigen::Vector4d dd_div_dv = w * 2.0 * lambda * m * difVec.transpose();
                
                gradient.block(data.cohE(cohI, 0) * 2, 0, 2, 1) += dd_div_dv.block(0, 0, 2, 1) + dd_div_dx_M_base;
                gradient.block(data.cohE(cohI, 1) * 2, 0, 2, 1) += dd_div_dv.block(2, 0, 2, 1) - dd_div_dx_M_base;
                gradient.block(data.cohE(cohI, 2) * 2, 0, 2, 1) += -dd_div_dv.block(0, 0, 2, 1) + dd_div_dx_M_base;
                gradient.block(data.cohE(cohI, 3) * 2, 0, 2, 1) += -dd_div_dv.block(2, 0, 2, 1) - dd_div_dx_M_base;
            }
        }
    }
    
    void CohesiveEnergy::computePrecondMtr(const TriangleSoup& data, Eigen::SparseMatrix<double>& precondMtr) const
    {
        //TODO: fix vert, clamped precondMtr?
        
    }
    
    void CohesiveEnergy::computeHessian(const TriangleSoup& data, Eigen::SparseMatrix<double>& hessian) const
    {
        assert(0 && "no hessian computation for this energy");
    }
    
    void CohesiveEnergy::checkEnergyVal(const TriangleSoup& data) const
    {
        
    }
    
    CohesiveEnergy::CohesiveEnergy(double p_tau, double p_alpha, double p_lambda) :
        tau(p_tau), alpha(p_alpha), lambda(p_lambda)
    {
    
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
