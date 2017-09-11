//
//  Optimizer.cpp
//  FracCuts
//
//  Created by Minchen Li on 8/31/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#include "Optimizer.hpp"
#include "SymStretchEnergy.hpp"

#include <igl/avg_edge_length.h>

#include <fstream>
#include <iostream>
#include <string>

extern std::string outputFolderPath;
extern std::ofstream logFile;

namespace FracCuts {
    
    Optimizer::Optimizer(const TriangleSoup& p_data0, const std::vector<Energy*>& p_energyTerms, const std::vector<double>& p_energyParams) : data0(p_data0), energyTerms(p_energyTerms), energyParams(p_energyParams)
    {
        assert(energyTerms.size() == energyParams.size());
        
        file_energyValPerIter.open(outputFolderPath + "energyValPerIter.txt");
        
        if(!energyTerms[0]->checkInversion(data0))
        {
            std::cout << "***Warning: element inversion detected in initial configuration!" << std::endl;
        }
    }
    
    Optimizer::~Optimizer(void)
    {
        file_energyValPerIter.close();
    }
    
    void Optimizer::precompute(void)
    {
        computePrecondMtr(data0, precondMtr);
        cholSolver.compute(precondMtr);
        if(cholSolver.info() != Eigen::Success) {
            assert(0 && "Cholesky decomposition failed!");
        }
        result = data0;
        computeEnergyVal(result, lastEnergyVal);
        file_energyValPerIter << lastEnergyVal << std::endl;
        std::cout << "E_initial = " << lastEnergyVal << std::endl;
    }
    
    const TriangleSoup& Optimizer::solve(int maxIter)
    {
        const double targetRes = 1.0e-12;
        for(int iterI = 0; iterI < maxIter; iterI++)
        {
            computeGradient(result, gradient);
            if(gradient.squaredNorm() < targetRes) {
                // converged
                break;
            }
            else
            {
                solve_oneStep();
            }
        }
        return result;
    }
    
    void Optimizer::solve_oneStep(void)
    {
        //!! for the changing hessian
        computePrecondMtr(result, precondMtr);
        cholSolver.compute(precondMtr);
        if(cholSolver.info() != Eigen::Success) {
            assert(0 && "Cholesky decomposition failed!");
        }
        
        searchDir = cholSolver.solve(-gradient);
        if(cholSolver.info() != Eigen::Success) {
            assert(0 && "Cholesky solve failed!");
        }
        lineSearch();
    }
    
    void Optimizer::lineSearch(void)
    {
        const double eps = 1.0e-12;
        double stepSize = 1.0;
        initStepSize(result, stepSize);
        stepSize *= 0.99; // producing degenerated element is not allowed
        std::cout << "stepSize: " << stepSize << " -> ";
        
        const double m = searchDir.dot(gradient);
        const double c1m = 1.0e-4 * m, c2m = 0.9 * m;
        TriangleSoup testingData = result;
        stepForward(testingData, stepSize);
        double testingE;
        Eigen::VectorXd testingG;
        computeEnergyVal(testingData, testingE);
        computeGradient(testingData, testingG);
        while((testingE > lastEnergyVal + stepSize * c1m) ||
              (searchDir.dot(testingG) < c2m)) // Wolfe condition
//        while(testingE > lastEnergyVal + stepSize * c1m) // Armijo condition
//        while(0)
        {
            stepSize /= 2.0;
            if(stepSize < eps) {
                //TODO: converged?
                break;
            }
            
            stepForward(testingData, stepSize);
            computeEnergyVal(testingData, testingE);
        }
        result.V = testingData.V;
        lastEnergyVal = testingE;
        
        std::cout << stepSize << std::endl;
        std::cout << "E_cur = " << testingE << std::endl;
        
        file_energyValPerIter << lastEnergyVal << std::endl;
    }
    
    void Optimizer::stepForward(TriangleSoup& data, double stepSize) const
    {
        assert(data.V.rows() * 2 == searchDir.size());
        assert(data.V.rows() == result.V.rows());
        
        for(int vI = 0; vI < data.V.rows(); vI++) {
            data.V(vI, 0) = result.V(vI, 0) + stepSize * searchDir[vI * 2];
            data.V(vI, 1) = result.V(vI, 1) + stepSize * searchDir[vI * 2 + 1];
        }
    }
    
    void Optimizer::getGradientVisual(Eigen::MatrixXd& arrowVec) const
    {
        assert(result.V.rows() * 2 == gradient.size());
        arrowVec.resize(result.V.rows(), result.V.cols());
        for(int vI = 0; vI < result.V.rows(); vI++) {
            arrowVec(vI, 0) = gradient[vI * 2];
            arrowVec(vI, 1) = gradient[vI * 2 + 1];
            arrowVec.row(vI).normalize();
        }
        arrowVec *= igl::avg_edge_length(result.V, result.F);
    }
    
    void Optimizer::initStepSize(const TriangleSoup& data, double& stepSize) const
    {
        for(int eI = 0; eI < energyTerms.size(); eI++) {
            energyTerms[eI]->initStepSize(data, searchDir, stepSize);
        }
    }
    
    void Optimizer::computeEnergyVal(const TriangleSoup& data, double& energyVal) const
    {
        energyTerms[0]->computeEnergyVal(data, energyVal);
        energyVal *= energyParams[0];
        for(int eI = 1; eI < energyTerms.size(); eI++) {
            double energyValI;
            energyTerms[eI]->computeEnergyVal(data, energyValI);
            energyVal += energyParams[eI] * energyValI;
        }
    }
    void Optimizer::computeGradient(const TriangleSoup& data, Eigen::VectorXd& gradient) const
    {
        energyTerms[0]->computeGradient(data, gradient);
        gradient *= energyParams[0];
        for(int eI = 1; eI < energyTerms.size(); eI++) {
            Eigen::VectorXd gradientI;
            energyTerms[eI]->computeGradient(data, gradientI);
            gradient += energyParams[eI] * gradientI;
        }
    }
    void Optimizer::computePrecondMtr(const TriangleSoup& data, Eigen::SparseMatrix<double>& precondMtr) const
    {
        energyTerms[0]->computePrecondMtr(data, precondMtr);
        precondMtr *= energyParams[0];
        for(int eI = 1; eI < energyTerms.size(); eI++) {
            Eigen::SparseMatrix<double> precondMtrI;
            energyTerms[eI]->computePrecondMtr(data, precondMtrI);
            precondMtr += energyParams[eI] * precondMtrI;
        }
//        std::cout << "det(precondMtr) = " << Eigen::MatrixXd(precondMtr).determinant() << std::endl;
//        logFile << precondMtr << std::endl;
    }
    void Optimizer::computeHessian(const TriangleSoup& data, Eigen::SparseMatrix<double>& hessian) const
    {
        energyTerms[0]->computeHessian(data, hessian);
        hessian *= energyParams[0];
        for(int eI = 1; eI < energyTerms.size(); eI++) {
            Eigen::SparseMatrix<double> hessianI;
            energyTerms[eI]->computeHessian(data, hessianI);
            hessian += energyParams[eI] * hessianI;
        }
    }
}
