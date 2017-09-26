//
//  SeparationEnergy.hpp
//  FracCuts
//
//  Created by Minchen Li on 9/8/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#ifndef SeparationEnergy_hpp
#define SeparationEnergy_hpp

#include "Energy.hpp"

namespace FracCuts {
    
    class SeparationEnergy : public Energy
    {
    public:
        virtual void getEnergyValPerElem(const TriangleSoup& data, Eigen::VectorXd& energyValPerElem, bool uniformWeight = false) const;
        virtual void computeGradient(const TriangleSoup& data, Eigen::VectorXd& gradient) const;
        virtual void computePrecondMtr(const TriangleSoup& data, Eigen::SparseMatrix<double>& precondMtr) const;
        virtual void computeHessian(const TriangleSoup& data, Eigen::SparseMatrix<double>& hessian) const;
        
        virtual void checkEnergyVal(const TriangleSoup& data) const;
        
    public:
        SeparationEnergy(double p_sigma_base, double p_sigma_param);
        
    public:
        bool decreaseSigma(void); // decrease sigma by half for homotopy optimization
        double getSigmaParam(void) const;
        
    protected:
        double kernel(double t) const;
        double kernelGradient(double t) const;
        double kernelHessian(double t) const;
        
    protected:
        double sigma_param;
        double sigma_base; // usually set to the average edge length of the mesh
        double sigma; // sigma = sigma_param * sigma_base
    };
    
}

#endif /* SeparationEnergy_hpp */
