//
//  Optimizer.cpp
//  FracCuts
//
//  Created by Minchen Li on 8/31/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#include "Optimizer.hpp"

namespace FracCuts {
    
    Optimizer::Optimizer(const TriangleSoup& p_data0, const std::vector<Energy>& p_energyTerms, const std::vector<double>& p_energyParams) :
    data0(p_data0), energyTerms(p_energyTerms), energyParams(p_energyParams)
    {
        assert(energyTerms.size() == energyParams.size());
    }
    
    void Optimizer::precompute(void)
    {
        computePrecondMtr(data0, precondMtr);
        cholSolver.compute(precondMtr);
        result = data0;
    }
    
    void Optimizer::solve(void)
    {
        const int maxIter = 100;
        const double targetRes = 1.0e-6;
        for(int iterI = 0; iterI < maxIter; iterI++)
        {
            computeGradient(result, gradient);
            if(gradient.squaredNorm() < targetRes) {
                // converged
                return;
            }
            else
            {
                solve_oneStep();
            }
        }
    }
    
    void Optimizer::solve_oneStep(void)
    {
        searchDir = cholSolver.solve(-gradient);
        lineSearch();
    }
    
    void Optimizer::lineSearch(void)
    {
        //TODO: implement element inversion preventing condition
        //TODO: implement Armijo/Wolfe condition?
        
        const double eps = 1.0e-3;
        double lastE;
        computeEnergyVal(result, lastE);
        
        TriangleSoup testingData = result;
        testingData.V += searchDir;
        double stepSize = 1.0;
        double testingE;
        computeEnergyVal(testingData, testingE);
        while(testingE > lastE) {
            stepSize /= 2.0;
            if(stepSize < eps) {
                //TODO: converged?
                break;
            }
            
            testingData.V = result.V + stepSize * searchDir;
            computeEnergyVal(testingData, testingE);
        }
        result.V = testingData.V;
    }
    
    void Optimizer::computeEnergyVal(const TriangleSoup& data, double& energyVal) const
    {
        energyTerms[0].computeEnergyVal(data, energyVal);
        energyVal *= energyParams[0];
        for(int eI = 1; eI < energyTerms.size(); eI++) {
            double energyValI;
            energyTerms[eI].computeEnergyVal(data, energyValI);
            energyVal += energyParams[eI] * energyValI;
        }
    }
    void Optimizer::computeGradient(const TriangleSoup& data, Eigen::VectorXd& gradient) const
    {
        energyTerms[0].computeGradient(data, gradient);
        gradient *= energyParams[0];
        for(int eI = 1; eI < energyTerms.size(); eI++) {
            Eigen::VectorXd gradientI;
            energyTerms[eI].computeGradient(data, gradientI);
            gradient += energyParams[eI] * gradientI;
        }
    }
    void Optimizer::computePrecondMtr(const TriangleSoup& data, Eigen::SparseMatrix<double>& precondMtr) const
    {
        energyTerms[0].computePrecondMtr(data, precondMtr);
        precondMtr *= energyParams[0];
        for(int eI = 1; eI < energyTerms.size(); eI++) {
            Eigen::SparseMatrix<double> precondMtrI;
            energyTerms[eI].computePrecondMtr(data, precondMtrI);
            precondMtr += energyParams[eI] * precondMtrI;
        }
    }
    void Optimizer::computeHessian(const TriangleSoup& data, Eigen::SparseMatrix<double>& hessian) const
    {
        energyTerms[0].computeHessian(data, hessian);
        hessian *= energyParams[0];
        for(int eI = 1; eI < energyTerms.size(); eI++) {
            Eigen::SparseMatrix<double> hessianI;
            energyTerms[eI].computeHessian(data, hessianI);
            hessian += energyParams[eI] * hessianI;
        }
    }
}
