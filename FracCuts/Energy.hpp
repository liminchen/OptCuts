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
    protected:
        const bool needRefactorize;
        
    public:
        Energy(bool p_needRefactorize);
        virtual ~Energy(void);
        
    public:
        bool getNeedRefactorize(void) const;
        
    public:
        virtual void computeEnergyVal(const TriangleSoup& data, double& energyVal, bool uniformWeight = false) const;
        virtual void getEnergyValPerElem(const TriangleSoup& data, Eigen::VectorXd& energyValPerElem, bool uniformWeight = false) const = 0;
        virtual void getEnergyValByElemID(const TriangleSoup& data, int elemI, double& energyVal, bool uniformWeight = false) const;
        virtual void computeGradient(const TriangleSoup& data, Eigen::VectorXd& gradient, bool uniformWeight = false) const = 0;
        virtual void computePrecondMtr(const TriangleSoup& data, Eigen::SparseMatrix<double>& precondMtr, bool uniformWeight = false) const = 0;
        virtual void computePrecondMtr(const TriangleSoup& data, Eigen::VectorXd* V,
                                       Eigen::VectorXi* I = NULL, Eigen::VectorXi* J = NULL, bool uniformWeight = false) const;
        virtual void computeHessian(const TriangleSoup& data, Eigen::SparseMatrix<double>& hessian, bool uniformWeight = false) const = 0;
        
        virtual void checkEnergyVal(const TriangleSoup& data) const = 0;
        
        virtual void checkGradient(const TriangleSoup& data) const; // check with finite difference method, according to energyVal
        virtual void checkHessian(const TriangleSoup& data, bool triplet = false) const; // check with finite difference method, according to gradient
        
        virtual void initStepSize(const TriangleSoup& data, const Eigen::VectorXd& searchDir, double& stepSize) const;
    };
    
}

#endif /* Energy_hpp */
