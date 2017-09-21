//
//  CohesiveEnergy.hpp
//  FracCuts
//
//  Created by Minchen Li on 9/20/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#ifndef CohesiveEnergy_hpp
#define CohesiveEnergy_hpp

#include "Energy.hpp"

namespace FracCuts {
    class CohesiveEnergy : public Energy
    {
    public:
        virtual void getEnergyValPerElem(const TriangleSoup& data, Eigen::VectorXd& energyValPerElem, bool uniformWeight = false) const;
        virtual void computeGradient(const TriangleSoup& data, Eigen::VectorXd& gradient) const;
        virtual void computePrecondMtr(const TriangleSoup& data, Eigen::SparseMatrix<double>& precondMtr) const;
        virtual void computeHessian(const TriangleSoup& data, Eigen::SparseMatrix<double>& hessian) const;
        
        virtual void checkEnergyVal(const TriangleSoup& data) const;
        
    public:
        CohesiveEnergy(double p_tau = 1.0, double p_alpha = 0.5, double p_lambda = 1.0);
        
    protected:
        void compute_dd_div_dx_M(const Eigen::RowVector4d& difVec, const Eigen::Matrix<Eigen::RowVector2d, 2, 2>& dP_div_dx,
                                 Eigen::Vector2d& result, double elemWeight = 1.0) const;
        
    protected:
        double tau;
        double alpha;
        double lambda;
    };
}

#endif /* CohesiveEnergy_hpp */
