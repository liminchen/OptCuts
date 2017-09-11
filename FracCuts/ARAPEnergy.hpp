//
//  ARAPEnergy.hpp
//  FracCuts
//
//  Created by Minchen Li on 9/5/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#ifndef ARAPEnergy_hpp
#define ARAPEnergy_hpp

#include "Energy.hpp"

namespace FracCuts
{
    
    class ARAPEnergy : public Energy
    {
    public:
        virtual void computeEnergyVal(const TriangleSoup& data, double& energyVal) const;
        virtual void computeGradient(const TriangleSoup& data, Eigen::VectorXd& gradient) const;
        virtual void computePrecondMtr(const TriangleSoup& data, Eigen::SparseMatrix<double>& precondMtr) const;
        virtual void computeHessian(const TriangleSoup& data, Eigen::SparseMatrix<double>& hessian) const;
        
        virtual void checkEnergyVal(const TriangleSoup& data) const;
        
        virtual void getEnergyValPerElem(const TriangleSoup& data, Eigen::VectorXd& energyValPerElem) const; // for visualization
    };
    
}

#endif /* ARAPEnergy_hpp */
