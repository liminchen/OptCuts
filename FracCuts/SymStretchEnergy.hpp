//
//  SymStretchEnergy.hpp
//  FracCuts
//
//  Created by Minchen Li on 9/3/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#ifndef SymStretchEnergy_hpp
#define SymStretchEnergy_hpp

#include "Energy.hpp"

namespace FracCuts {
    
    class SymStretchEnergy : public Energy {
    public:
        virtual void computeEnergyVal(const TriangleSoup& data, double& energyVal) const;
        virtual void computeGradient(const TriangleSoup& data, Eigen::VectorXd& gradient) const;
        virtual void computePrecondMtr(const TriangleSoup& data, Eigen::SparseMatrix<double>& precondMtr) const;
        virtual void computeHessian(const TriangleSoup& data, Eigen::SparseMatrix<double>& hessian) const;
        
        virtual void checkEnergyVal(const TriangleSoup& data) const; // check with isometric case
        
        virtual void getEnergyValPerElem(const TriangleSoup& data, Eigen::VectorXd& energyValPerElem) const;
        
        // to prevent element inversion
        static void lineSearch(const TriangleSoup& data, const Eigen::VectorXd& searchDir, double& stepSize);
    };
    
}

#endif /* SymStretchEnergy_hpp */
