//
//  SeparationEnergy.cpp
//  FracCuts
//
//  Created by Minchen Li on 9/8/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#include "SeparationEnergy.hpp"
#include "IglUtils.hpp"

#include <iostream>
#include <fstream>

extern std::ofstream logFile;

namespace FracCuts {
    
    void SeparationEnergy::computeEnergyVal(const TriangleSoup& data, double& energyVal) const
    {
        Eigen::VectorXd energyValPerElem;
        getEnergyValPerElem(data, energyValPerElem);
        energyVal = energyValPerElem.sum();
    }
    
    void SeparationEnergy::computeGradient(const TriangleSoup& data, Eigen::VectorXd& gradient) const
    {
        gradient.resize(data.V.rows() * 2);
        gradient.setZero();
        for(int cohI = 0; cohI < data.cohE.rows(); cohI++)
        {
            if(!data.boundaryEdge[cohI]) {
                const double w = data.edgeLen[cohI];
                const Eigen::Vector2d xamc = data.V.row(data.cohE(cohI, 0)) - data.V.row(data.cohE(cohI, 2));
                const Eigen::Vector2d xbmd = data.V.row(data.cohE(cohI, 1)) - data.V.row(data.cohE(cohI, 3));
                const double kG_ac = w * kernelGradient(xamc.squaredNorm());
                const double kG_bd = w * kernelGradient(xbmd.squaredNorm());
                gradient.block(data.cohE(cohI, 0) * 2, 0, 2, 1) += kG_ac * 2.0 * xamc;
                gradient.block(data.cohE(cohI, 2) * 2, 0, 2, 1) -= kG_ac * 2.0 * xamc;
                gradient.block(data.cohE(cohI, 1) * 2, 0, 2, 1) += kG_bd * 2.0 * xbmd;
                gradient.block(data.cohE(cohI, 3) * 2, 0, 2, 1) -= kG_bd * 2.0 * xbmd;
            }
        }
    }
    
    void SeparationEnergy::computePrecondMtr(const TriangleSoup& data, Eigen::SparseMatrix<double>& precondMtr) const
    {
        computeHessian(data, precondMtr);
    }
    
    void SeparationEnergy::computeHessian(const TriangleSoup& data, Eigen::SparseMatrix<double>& hessian) const
    {
        hessian.resize(data.V.rows() * 2, data.V.rows() * 2);
        hessian.setZero();
        for(int cohI = 0; cohI < data.cohE.rows(); cohI++)
        {
            if(!data.boundaryEdge[cohI]) {
                const Eigen::Vector2d xamc = data.V.row(data.cohE(cohI, 0)) - data.V.row(data.cohE(cohI, 2));
                const Eigen::Vector2d xbmd = data.V.row(data.cohE(cohI, 1)) - data.V.row(data.cohE(cohI, 3));
                
                Eigen::VectorXd dtddx_ac;
                dtddx_ac.resize(4);
                dtddx_ac << 2 * xamc, -2 * xamc;
                Eigen::VectorXd dtddx_bd;
                dtddx_bd.resize(4);
                dtddx_bd << 2 * xbmd, -2 * xbmd;
                
                Eigen::MatrixXd dt2dd2x;
                dt2dd2x.resize(4, 4);
                dt2dd2x <<
                2.0, 0.0, -2.0, 0.0,
                0.0, 2.0, 0.0, -2.0,
                -2.0, 0.0, 2.0, 0.0,
                0.0, -2.0, 0.0, 2.0;
                
                const double w = data.edgeLen[cohI];
                
                Eigen::MatrixXd hessian_ac;
                hessian_ac.resize(4, 4);
                const double sqn_xamc = xamc.squaredNorm();
                hessian_ac = w * (kernelHessian(sqn_xamc) * dtddx_ac * dtddx_ac.transpose() +
                                  kernelGradient(sqn_xamc) * dt2dd2x);
                IglUtils::addBlockToMatrix(hessian, hessian_ac, Eigen::Vector2i(data.cohE(cohI, 0), data.cohE(cohI, 2)), 2);
                
                Eigen::MatrixXd hessian_bd;
                hessian_bd.resize(4, 4);
                const double sqn_xbmd = xamc.squaredNorm();
                hessian_bd = w * (kernelHessian(sqn_xbmd) * dtddx_bd * dtddx_bd.transpose() +
                                  kernelGradient(sqn_xbmd) * dt2dd2x);
                IglUtils::addBlockToMatrix(hessian, hessian_bd, Eigen::Vector2i(data.cohE(cohI, 1), data.cohE(cohI, 3)), 2);
            }
        }
        hessian.makeCompressed();
    }
    
    void SeparationEnergy::checkEnergyVal(const TriangleSoup& data) const
    {
        
    }
    
    void SeparationEnergy::getEnergyValPerElem(const TriangleSoup& data, Eigen::VectorXd& energyValPerElem) const
    {
        energyValPerElem.resize(data.cohE.rows());
        for(int cohI = 0; cohI < data.cohE.rows(); cohI++)
        {
            if(data.boundaryEdge[cohI]) {
                energyValPerElem[cohI] = 0.0;
            }
            else {
                const double w = data.edgeLen[cohI];
                energyValPerElem[cohI] = w * kernel((data.V.row(data.cohE(cohI, 0)) - data.V.row(data.cohE(cohI, 2))).squaredNorm());
                energyValPerElem[cohI] += w * kernel((data.V.row(data.cohE(cohI, 1)) - data.V.row(data.cohE(cohI, 3))).squaredNorm());
            }
        }
    }
    
    SeparationEnergy::SeparationEnergy(double p_sigma_base, double p_sigma_param) :
        sigma_base(p_sigma_base), sigma_param(p_sigma_param)
    {
        sigma = sigma_param * sigma_base;
    }
    
    void SeparationEnergy::decreaseSigma(void)
    {
        if(sigma_param > 1.0e-6) {
            sigma_param /= 2.0;
            sigma = sigma_param * sigma_base;
            std::cout << "sigma decreased to ";
        }
        else {
            std::cout << "sigma stays at it's predefined minimum ";
        }
        std::cout << sigma_param << " * " << sigma_base << " = " << sigma << std::endl;
    }
    
    double SeparationEnergy::kernel(double t) const
    {
//        return t * t / (t * t + sigma);
        return t / (t + sigma);
    }
    
    double SeparationEnergy::kernelGradient(double t) const
    {
//        const double denom_sqrt = t * t + sigma;
//        return 2 * t * sigma / (denom_sqrt * denom_sqrt);
        const double denom_sqrt = t + sigma;
        return sigma / (denom_sqrt * denom_sqrt);
    }
    
    double SeparationEnergy::kernelHessian(double t) const
    {
//        const double t2 = t * t;
//        return -2 * sigma * (3 * t2 - sigma) * (t2 + sigma) / std::pow(t2 + sigma, 4);
        const double tpsigma = t + sigma;
        return -2 * sigma / std::pow(tpsigma, 3);
    }
    
}
