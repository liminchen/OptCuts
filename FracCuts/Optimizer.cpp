//
//  Optimizer.cpp
//  FracCuts
//
//  Created by Minchen Li on 8/31/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#include "Optimizer.hpp"
#include "SymStretchEnergy.hpp"
#include "SeparationEnergy.hpp"
#include "IglUtils.hpp"

#include <igl/avg_edge_length.h>

//#include <omp.h>

#include <fstream>
#include <iostream>
#include <string>

extern std::string outputFolderPath;
extern std::ofstream logFile;

namespace FracCuts {
    
    Optimizer::Optimizer(const TriangleSoup& p_data0, const std::vector<Energy*>& p_energyTerms, const std::vector<double>& p_energyParams) : data0(p_data0), energyTerms(p_energyTerms), energyParams(p_energyParams)
    {
        assert(energyTerms.size() == energyParams.size());
        
        gradient_ET.resize(energyTerms.size());
        energyVal_ET.resize(energyTerms.size());
        
        file_energyValPerIter.open(outputFolderPath + "energyValPerIter.txt");
        file_gradientPerIter.open(outputFolderPath + "gradientPerIter.txt");
        
        assert(data0.checkInversion());
        
        globalIterNum = 0;
        
        needRefactorize = false;
        for(const auto& energyTermI : energyTerms) {
            if(energyTermI->getNeedRefactorize()) {
                needRefactorize = true;
                break;
            }
        }
        
        pardisoThreadAmt = 1;
    }
    
    Optimizer::~Optimizer(void)
    {
        file_energyValPerIter.close();
        file_gradientPerIter.close();
    }
    
    void Optimizer::computeLastEnergyVal(void)
    {
        computeEnergyVal(result, lastEnergyVal);
    }
    
    const TriangleSoup& Optimizer::getResult(void) const {
        return result;
    }
    
    int Optimizer::getIterNum(void) const {
        return globalIterNum;
    }
    
    void Optimizer::precompute(void)
    {
        computePrecondMtr(data0, precondMtr);
        
        if(!pardisoThreadAmt) {
            cholSolver.analyzePattern(precondMtr);
            cholSolver.factorize(precondMtr);
            if(cholSolver.info() != Eigen::Success) {
                assert(0 && "Cholesky decomposition failed!");
            }
        }
        else {
            pardisoSolver.set_type(pardisoThreadAmt, 2);
            Eigen::VectorXi I, J;
            Eigen::VectorXd V;
            IglUtils::sparseMatrixToTriplet(precondMtr, I, J, V);
            pardisoSolver.set_pattern(I, J, V);
            pardisoSolver.analyze_pattern();
            pardisoSolver.factorize();
        }
        
        result = data0;
        targetGRes = data0.V_rest.rows() * 1.0e-12 * data0.avgEdgeLen * data0.avgEdgeLen;
//        targetGRes = data0.V_rest.rows() * 1.0e-10 * data0.avgEdgeLen * data0.avgEdgeLen;
        computeEnergyVal(result, lastEnergyVal);
        file_energyValPerIter << lastEnergyVal;
        for(int eI = 0; eI < energyTerms.size(); eI++) {
            file_energyValPerIter << " " << energyVal_ET[eI];
        }
        double seamSparsity;
        result.computeSeamSparsity(seamSparsity);
        file_energyValPerIter << " " << seamSparsity << std::endl;
        std::cout << "E_initial = " << lastEnergyVal << std::endl;
    }
    
    bool Optimizer::solve(int maxIter)
    {
        for(int iterI = 0; iterI < maxIter; iterI++)
        {
            createFracture(-1.0); //DEBUG
            computeGradient(result, gradient);
            const double sqn_g = gradient.squaredNorm();
            std::cout << "||gradient||^2 = " << sqn_g << ", targetGRes = " << targetGRes << std::endl;
            file_gradientPerIter << sqn_g;
            for(int eI = 0; eI < energyTerms.size(); eI++) {
                file_gradientPerIter << " " << gradient_ET[eI].squaredNorm();
            }
            file_gradientPerIter << std::endl;
            if(sqn_g < targetGRes) {
                // converged
                file_energyValPerIter << lastEnergyVal;
                for(int eI = 0; eI < energyTerms.size(); eI++) {
                    file_energyValPerIter << " " << energyVal_ET[eI];
                }
                double seamSparsity;
                result.computeSeamSparsity(seamSparsity);
                file_energyValPerIter << " " << seamSparsity << std::endl;
                globalIterNum++;
                return true;
            }
            else {
                if(solve_oneStep()) {
                    globalIterNum++;
                    return true;
                }
            }
            globalIterNum++;
        }
        return false;
    }
    
    void Optimizer::updatePrecondMtrAndFactorize(void)
    {
        if(needRefactorize) {
            // don't need to call this function
            return;
        }
        
        std::cout << "recompute proxy/Hessian matrix and factorize..." << std::endl;
        computePrecondMtr(result, precondMtr);
        if(!pardisoThreadAmt) {
            cholSolver.factorize(precondMtr);
            if(cholSolver.info() != Eigen::Success) {
                IglUtils::writeSparseMatrixToFile(outputFolderPath + "precondMtr_decomposeFailed", precondMtr);
                assert(0 && "Cholesky decomposition failed!");
            }
        }
        else {
            Eigen::VectorXd V;
            IglUtils::sparseMatrixToTriplet(precondMtr, V);
            pardisoSolver.update_a(V);
            pardisoSolver.factorize();
        }
    }
    
    void Optimizer::separateTriangles(double energyThres)
    {
        Eigen::VectorXd distortionPerElem;
        energyTerms[0]->getEnergyValPerElem(result, distortionPerElem, true);
        bool changed = result.separateTriangle(distortionPerElem, energyThres);
        logFile << result.cohE; //DEBUG
        if(changed) {
            targetGRes = result.V_rest.rows() * 1.0e-6 * data0.avgEdgeLen * data0.avgEdgeLen;
            
            // compute energy and output
            computeEnergyVal(result, lastEnergyVal);
            file_energyValPerIter << lastEnergyVal;
            for(int eI = 0; eI < energyTerms.size(); eI++) {
                file_energyValPerIter << " " << energyVal_ET[eI];
            }
            double seamSparsity;
            result.computeSeamSparsity(seamSparsity);
            file_energyValPerIter << " " << seamSparsity << std::endl;
            globalIterNum++;
            
            // compute gradient and output
            computeGradient(result, gradient);
            file_gradientPerIter << gradient.squaredNorm();
            for(int eI = 0; eI < energyTerms.size(); eI++) {
                file_gradientPerIter << " " << gradient_ET[eI].squaredNorm();
            }
            file_gradientPerIter << std::endl;
            
            // for the changing hessian
            std::cout << "recompute proxy/Hessian matrix and factorize..." << std::endl;
            computePrecondMtr(result, precondMtr);
            if(!pardisoThreadAmt) {
                cholSolver.analyzePattern(precondMtr);
                if(!needRefactorize) {
                    cholSolver.factorize(precondMtr);
                    if(cholSolver.info() != Eigen::Success) {
                        IglUtils::writeSparseMatrixToFile(outputFolderPath + "precondMtr_decomposeFailed", precondMtr);
                        assert(0 && "Cholesky decomposition failed!");
                    }
                }
            }
            else {
                pardisoSolver = PardisoSolver<Eigen::VectorXi, Eigen::VectorXd>(); //TODO: make it cheaper!
                pardisoSolver.set_type(pardisoThreadAmt, 2);
                Eigen::VectorXi I, J;
                Eigen::VectorXd V;
                IglUtils::sparseMatrixToTriplet(precondMtr, I, J, V);
                pardisoSolver.set_pattern(I, J, V);
                pardisoSolver.analyze_pattern();
                if(!needRefactorize) {
                    pardisoSolver.factorize();
                }
            }
        }
    }
    
    void Optimizer::createFracture(double stressThres)
    {
//        bool changed = result.splitVertex(Eigen::VectorXd::Zero(result.V.rows()), stressThres); //DEBUG
//        bool changed = result.splitEdge(); //DEBUG
        logFile << result.V.rows() << std::endl;
        bool changed = (result.mergeEdge() | result.splitEdge()); //DEBUG
        if(changed) {
//            logFile << result.F << std::endl; //DEBUG
//            logFile << result.cohE << std::endl; //DEBUG
            
            targetGRes = result.V_rest.rows() * 1.0e-12 * data0.avgEdgeLen * data0.avgEdgeLen;
            
            // compute energy and output
            computeEnergyVal(result, lastEnergyVal);
            
            // compute gradient and output
            computeGradient(result, gradient);
            
            // for the changing hessian
            std::cout << "recompute proxy/Hessian matrix and factorize..." << std::endl;
            computePrecondMtr(result, precondMtr);
            if(!pardisoThreadAmt) {
                cholSolver.analyzePattern(precondMtr);
                if(!needRefactorize) {
                    cholSolver.factorize(precondMtr);
                    if(cholSolver.info() != Eigen::Success) {
                        IglUtils::writeSparseMatrixToFile(outputFolderPath + "precondMtr_decomposeFailed", precondMtr);
                        assert(0 && "Cholesky decomposition failed!");
                    }
                }
            }
            else {
                pardisoSolver = PardisoSolver<Eigen::VectorXi, Eigen::VectorXd>(); //TODO: make it cheaper!
                pardisoSolver.set_type(pardisoThreadAmt, 2);
                Eigen::VectorXi I, J;
                Eigen::VectorXd V;
                IglUtils::sparseMatrixToTriplet(precondMtr, I, J, V);
                pardisoSolver.set_pattern(I, J, V);
                pardisoSolver.analyze_pattern();
                if(!needRefactorize) {
                    pardisoSolver.factorize();
                }
            }
        }
    }
    
    bool Optimizer::solve_oneStep(void)
    {
        if(needRefactorize) {
            // for the changing hessian
            std::cout << "recompute proxy/Hessian matrix and factorize..." << std::endl;
            computePrecondMtr(result, precondMtr);
            
            if(!pardisoThreadAmt) {
                cholSolver.factorize(precondMtr);
                if(cholSolver.info() != Eigen::Success) {
                    IglUtils::writeSparseMatrixToFile(outputFolderPath + "precondMtr_decomposeFailed", precondMtr);
                    assert(0 && "Cholesky decomposition failed!");
                }
            }
            else {
                Eigen::VectorXd V;
                IglUtils::sparseMatrixToTriplet(precondMtr, V);
                pardisoSolver.update_a(V);
                pardisoSolver.factorize();
            }
        }
        
        if(!pardisoThreadAmt) {
            searchDir = cholSolver.solve(-gradient);
            if(cholSolver.info() != Eigen::Success) {
                assert(0 && "Cholesky solve failed!");
            }
        }
        else {
            Eigen::VectorXd minusG = -gradient;
            pardisoSolver.solve(minusG, searchDir);
        }
        
        bool stopped = lineSearch();
        if(stopped) {
            IglUtils::writeSparseMatrixToFile(outputFolderPath + "precondMtr_stopped_" + std::to_string(globalIterNum), precondMtr);
        }
        return stopped;
    }
    
    bool Optimizer::lineSearch(void)
    {
        bool stopped = false;
        double stepSize = 1.0;
        initStepSize(result, stepSize);
        stepSize *= 0.99; // producing degenerated element is not allowed
        std::cout << "stepSize: " << stepSize << " -> ";
        
        const double m = searchDir.dot(gradient);
        const double c1m = 1.0e-4 * m, c2m = 0.9 * m;
        TriangleSoup testingData = result;
        stepForward(testingData, stepSize);
//        double stepLen = (stepSize * searchDir).squaredNorm();
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
//            stepLen = (stepSize * searchDir).squaredNorm();
//            if(stepLen < targetGRes) {
            if(stepSize < 1e-12) {
                stopped = true;
                break;
            }
            
            stepForward(testingData, stepSize);
            computeEnergyVal(testingData, testingE);
        }
        while(!testingData.checkInversion()) {
            stepSize /= 2.0;
            if(stepSize < 1e-12) {
                stopped = true;
                break;
            }
            
            stepForward(testingData, stepSize);
            computeEnergyVal(testingData, testingE);
        }
        result.V = testingData.V;
        lastEnergyVal = testingE;
        
        std::cout << stepSize << std::endl;
        std::cout << "stepLen = " << (stepSize * searchDir).squaredNorm() << std::endl;
        std::cout << "E_cur = " << testingE << std::endl;
        
        file_energyValPerIter << lastEnergyVal;
        for(int eI = 0; eI < energyTerms.size(); eI++) {
            file_energyValPerIter << " " << energyVal_ET[eI];
        }
        double seamSparsity;
        result.computeSeamSparsity(seamSparsity);
        file_energyValPerIter << " " << seamSparsity << std::endl;
        
        return stopped;
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
    
    void Optimizer::computeEnergyVal(const TriangleSoup& data, double& energyVal)
    {
        energyTerms[0]->computeEnergyVal(data, energyVal_ET[0]);
        energyVal_ET[0] *= energyParams[0];
        energyVal = energyVal_ET[0];
        for(int eI = 1; eI < energyTerms.size(); eI++) {
            energyTerms[eI]->computeEnergyVal(data, energyVal_ET[eI]);
            energyVal_ET[eI] *= energyParams[eI];
            energyVal += energyVal_ET[eI];
        }
    }
    void Optimizer::computeGradient(const TriangleSoup& data, Eigen::VectorXd& gradient)
    {
        energyTerms[0]->computeGradient(data, gradient_ET[0]);
        gradient_ET[0] *= energyParams[0];
        gradient = gradient_ET[0];
        for(int eI = 1; eI < energyTerms.size(); eI++) {
            energyTerms[eI]->computeGradient(data, gradient_ET[eI]);
            gradient_ET[eI] *= energyParams[eI];
            gradient += gradient_ET[eI];
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
        
//        Eigen::BDCSVD<Eigen::MatrixXd> svd((Eigen::MatrixXd(precondMtr)));
//        logFile << "singular values of precondMtr_E:" << std::endl << svd.singularValues() << std::endl;
//        double det = 1.0;
//        for(int i = svd.singularValues().size() - 1; i >= 0; i--) {
//            det *= svd.singularValues()[i];
//        }
//        std::cout << "det(precondMtr_E) = " << det << std::endl;
        
//        const double det = Eigen::MatrixXd(precondMtr).determinant();
//        logFile << det << std::endl;
//        if(det <= 1e-10) {
//            std::cout << "***Warning: Indefinte hessian!" << std::endl;
//        }
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
