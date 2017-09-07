//
//  Energy.hpp
//  FracCuts
//
//  Created by Minchen Li on 8/31/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#ifndef Energy_hpp
#define Energy_hpp

#include "TriangleSoup.hpp"

namespace FracCuts {
    
    // an abstract class for energy terms in the objective of an optimization problem
    class Energy {
    public:
        virtual ~Energy(void);
        
    public:
        virtual void computeEnergyVal(const TriangleSoup& data, double& energyVal) const = 0;
        virtual void computeGradient(const TriangleSoup& data, Eigen::VectorXd& gradient) const = 0;
        virtual void computePrecondMtr(const TriangleSoup& data, Eigen::SparseMatrix<double>& precondMtr) const = 0;
        virtual void computeHessian(const TriangleSoup& data, Eigen::SparseMatrix<double>& hessian) const = 0;
        
        virtual void checkEnergyVal(const TriangleSoup& data) const = 0;
        virtual void checkGradient(const TriangleSoup& data) const; // check with finite difference method
        
        virtual bool checkInversion(const TriangleSoup& data) const;
        
        virtual void getEnergyValPerElem(const TriangleSoup& data, Eigen::VectorXd& energyValPerElem) const = 0;
    };
    
}

#endif /* Energy_hpp */
