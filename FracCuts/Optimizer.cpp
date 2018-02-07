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
#include <numeric>

extern FracCuts::MethodType methodType;
extern const std::string outputFolderPath;
extern const bool fractureMode;
extern std::ofstream logFile;
extern clock_t ticksPast_frac;

namespace FracCuts {
    
    Optimizer::Optimizer(const TriangleSoup& p_data0, const std::vector<Energy*>& p_energyTerms, const std::vector<double>& p_energyParams,
        bool p_propagateFracture, bool p_mute) : data0(p_data0), energyTerms(p_energyTerms), energyParams(p_energyParams)
    {
        assert(energyTerms.size() == energyParams.size());
        
        energyParamSum = 0.0;
        for(const auto& ePI : energyParams) {
            energyParamSum += ePI;
        }
        
        gradient_ET.resize(energyTerms.size());
        energyVal_ET.resize(energyTerms.size());
        
        allowEDecRelTol = true;
        propagateFracture = p_propagateFracture;
        mute = p_mute;
        
        if(!mute) {
            file_energyValPerIter.open(outputFolderPath + "energyValPerIter.txt");
            file_gradientPerIter.open(outputFolderPath + "gradientPerIter.txt");
        }
        
        if(!data0.checkInversion()) {
            exit(-1);
        }
        
        globalIterNum = 0;
        relGL2Tol = 1.0e-8;
        topoIter = 0;
        
        needRefactorize = false;
        for(const auto& energyTermI : energyTerms) {
            if(energyTermI->getNeedRefactorize()) {
                needRefactorize = true;
                break;
            }
        }
        
//        pardisoThreadAmt = 0;
        pardisoThreadAmt = 1; //TODO: use more threads!
    }
    
    Optimizer::~Optimizer(void)
    {
        if(file_energyValPerIter.is_open()) {
            file_energyValPerIter.close();
        }
        if(file_gradientPerIter.is_open()) {
            file_gradientPerIter.close();
        }
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
    
    int Optimizer::getTopoIter(void) const {
        return topoIter;
    }
    
    void Optimizer::setRelGL2Tol(double p_relTol)
    {
        assert(p_relTol > 0.0);
        relGL2Tol = p_relTol;
        updateTargetGRes();
    }
    
    void Optimizer::setAllowEDecRelTol(bool p_allowEDecRelTol)
    {
        allowEDecRelTol = p_allowEDecRelTol;
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
            pardisoSolver.set_type(pardisoThreadAmt, -2);
            pardisoSolver.set_pattern(I_mtr, J_mtr, V_mtr);
            pardisoSolver.analyze_pattern();
            try {
                pardisoSolver.factorize();
            }
            catch(std::exception e) {
                IglUtils::writeSparseMatrixToFile(outputFolderPath + "mtr_factorizeFail", I_mtr, J_mtr, V_mtr, true);
                exit(-1);
            }
        }
        
        lastEDec = 0.0;
        result = data0;
        updateTargetGRes();
        computeEnergyVal(result, lastEnergyVal);
        if(!mute) {
            double seamSparsity;
            result.computeSeamSparsity(seamSparsity, !fractureMode);
            seamSparsity /= result.virtualRadius;
            if(fractureMode) {
                file_energyValPerIter << lastEnergyVal + (1.0 - energyParams[0]) * seamSparsity;
            }
            else {
                file_energyValPerIter << lastEnergyVal;
            }
            for(int eI = 0; eI < energyTerms.size(); eI++) {
                file_energyValPerIter << " " << energyVal_ET[eI];
            }
            file_energyValPerIter << " " << seamSparsity << std::endl;
            std::cout << "E_initial = " << lastEnergyVal << std::endl;
        }
    }
    
    bool Optimizer::solve(int maxIter)
    {
        for(int iterI = 0; iterI < maxIter; iterI++)
        {
            computeGradient(result, gradient);
            const double sqn_g = gradient.squaredNorm();
            if(!mute) {
                std::cout << "||gradient||^2 = " << sqn_g << ", targetGRes = " << targetGRes << std::endl;
                file_gradientPerIter << sqn_g;
                for(int eI = 0; eI < energyTerms.size(); eI++) {
                    file_gradientPerIter << " " << gradient_ET[eI].squaredNorm();
                }
                file_gradientPerIter << std::endl;
            }
            if(sqn_g < targetGRes) {
                // converged
                lastEDec = 0.0;
                if(!mute) {
                    double seamSparsity;
                    result.computeSeamSparsity(seamSparsity, !fractureMode);
                    seamSparsity /= result.virtualRadius;
                    if(fractureMode) {
                        file_energyValPerIter << lastEnergyVal + (1.0 - energyParams[0]) * seamSparsity;
                    }
                    else {
                        file_energyValPerIter << lastEnergyVal;
                    }
                    for(int eI = 0; eI < energyTerms.size(); eI++) {
                        file_energyValPerIter << " " << energyVal_ET[eI];
                    }
                    file_energyValPerIter << " " << seamSparsity << std::endl;
                }
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
            
            if(propagateFracture) {
                if(!createFracture(lastEDec, false)) {
                    propagateFracture = false;
                }
            }
        }
        return false;
    }
    
    void Optimizer::updatePrecondMtrAndFactorize(void)
    {
        if(needRefactorize) {
            // don't need to call this function
            return;
        }
        
        if(!mute) {
            std::cout << "recompute proxy/Hessian matrix and factorize..." << std::endl;
        }
        computePrecondMtr(result, precondMtr);
        if(!pardisoThreadAmt) {
            cholSolver.factorize(precondMtr);
            if(cholSolver.info() != Eigen::Success) {
                IglUtils::writeSparseMatrixToFile(outputFolderPath + "precondMtr_decomposeFailed", precondMtr);
                assert(0 && "Cholesky decomposition failed!");
            }
        }
        else {
            pardisoSolver.update_a(V_mtr);
            pardisoSolver.factorize();
        }
    }
    
    void Optimizer::setConfig(const TriangleSoup& config)
    {
        result = config; //!!! is it able to copy all?
        
        updateTargetGRes();
        
        // compute energy and output
        computeEnergyVal(result, lastEnergyVal);
        
        // compute gradient and output
        computeGradient(result, gradient);
        
        // for the changing hessian
        if(!mute) {
            std::cout << "recompute proxy/Hessian matrix and factorize..." << std::endl;
        }
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
            pardisoSolver.set_type(pardisoThreadAmt, -2);
            pardisoSolver.set_pattern(I_mtr, J_mtr, V_mtr);
            pardisoSolver.analyze_pattern();
            if(!needRefactorize) {
                pardisoSolver.factorize();
            }
        }
    }
    
    bool Optimizer::createFracture(double stressThres, bool initiation, bool allowPropagate, bool allowInSplit)
    {
        if(initiation) {
            topoIter++;
        }
        
        clock_t tickStart = clock();
        bool changed = false;
        switch(methodType) {
            case MT_OURS: {
//                changed = result.splitEdge(1.0 - energyParams[0], stressThres, !initiation, allowInSplit);
                changed = result.splitOrMerge(1.0 - energyParams[0], stressThres, !initiation, allowInSplit);
                break;
            }
                
            case MT_GEOMIMG:
                result.geomImgCut();
                allowPropagate = false;
                changed = true;
                break;
                
            default:
                assert(0 && "Fracture forbiddened for current method type!");
                break;
        }
//        logFile << result.V.rows() << std::endl;
        if(changed) {
//            logFile << result.F << std::endl; //DEBUG
//            logFile << result.cohE << std::endl; //DEBUG
//            while(result.splitEdge(0.0, true)) {} // propagate
            
            updateTargetGRes();
            
            // compute energy and output
            computeEnergyVal(result, lastEnergyVal);
            
            // compute gradient and output
            computeGradient(result, gradient);
            if(gradient.squaredNorm() < targetGRes) {
                logFile << "||g||^2 = " << gradient.squaredNorm() << " after fracture initiation!" << std::endl;
            }
            
            // for the changing hessian
            if(!mute) {
                std::cout << "recompute proxy/Hessian matrix and factorize..." << std::endl;
            }
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
                pardisoSolver.set_type(pardisoThreadAmt, -2);
                pardisoSolver.set_pattern(I_mtr, J_mtr, V_mtr);
                pardisoSolver.analyze_pattern();
                if(!needRefactorize) {
                    pardisoSolver.factorize();
                }
            }
            
            if(allowPropagate && initiation) {
//                solve(1);
                propagateFracture = true;
            }
        }
        ticksPast_frac += clock() - tickStart;
        return changed;
    }
    
    bool Optimizer::solve_oneStep(void)
    {
        if(needRefactorize) {
            // for the changing hessian
            if(!mute) {
                std::cout << "recompute proxy/Hessian matrix..." << std::endl;
            }
            computePrecondMtr(result, precondMtr);
            
            if(!mute) {
                std::cout << "factorizing proxy/Hessian matrix..." << std::endl;
            }
            if(!pardisoThreadAmt) {
                cholSolver.factorize(precondMtr);
                if(cholSolver.info() != Eigen::Success) {
                    IglUtils::writeSparseMatrixToFile(outputFolderPath + "precondMtr_decomposeFailed", precondMtr);
                    assert(0 && "Cholesky decomposition failed!");
                }
            }
            else {
                pardisoSolver.update_a(V_mtr);
                try {
                    pardisoSolver.factorize();
                }
                catch(std::exception e) {
                    IglUtils::writeSparseMatrixToFile(outputFolderPath + "mtr", I_mtr, J_mtr, V_mtr, true);
                    exit(-1);
                }
            }
        }
        
        //TODO: half matrix size when you can
//        if(precondMtr.rows() == result.V.rows() * 2) {
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
//        }
//        else {
//            assert(precondMtr.rows() == result.V.rows());
//            
//            Eigen::VectorXd gradient_x, gradient_y;
//            gradient_x.resize(result.V.rows());
//            gradient_y.resize(result.V.rows());
//            for(int vI = 0; vI < result.V.rows(); vI++) {
//                int startInd = vI * 2;
//                gradient_x[vI] = gradient[startInd];
//                gradient_y[vI] = gradient[startInd + 1];
//            }
//            
//            Eigen::VectorXd searchDir_x, searchDir_y;
//            if(!pardisoThreadAmt) {
//                searchDir_x = cholSolver.solve(-gradient_x);
//                if(cholSolver.info() != Eigen::Success) {
//                    assert(0 && "Cholesky solve failed!");
//                }
//                searchDir_y = cholSolver.solve(-gradient_y);
//                if(cholSolver.info() != Eigen::Success) {
//                    assert(0 && "Cholesky solve failed!");
//                }
//            }
//            else {
//                Eigen::VectorXd minusG_x = -gradient_x;
//                pardisoSolver.solve(minusG_x, searchDir_x);
//                Eigen::VectorXd minusG_y = -gradient_y;
//                pardisoSolver.solve(minusG_y, searchDir_y);
//            }
//
//            searchDir.resize(result.V.rows() * 2);
//            for(int vI = 0; vI < result.V.rows(); vI++) {
//                int startInd = vI * 2;
//                searchDir[startInd] = searchDir_x[vI];
//                searchDir[startInd + 1] = searchDir_y[vI];
//            }
//        }
        
        bool stopped = lineSearch();
        if(stopped) {
//            IglUtils::writeSparseMatrixToFile(outputFolderPath + "precondMtr_stopped_" + std::to_string(globalIterNum), precondMtr);
//            logFile << "descent step stopped at overallIter" << globalIterNum << " for no prominent energy decrease." << std::endl;
        }
        return stopped;
    }
    
    bool Optimizer::lineSearch(void)
    {
        bool stopped = false;
        double stepSize = 1.0;
        initStepSize(result, stepSize);
        stepSize *= 0.99; // producing degenerated element is not allowed
        if(!mute) {
            std::cout << "stepSize: " << stepSize << " -> ";
        }
        
        const double m = searchDir.dot(gradient);
        const double c1m = 1.0e-4 * m;
//        const double c2m = (1.0 - 1.0e-6) * m;
        TriangleSoup testingData = result;
        stepForward(testingData, stepSize);
//        double stepLen = (stepSize * searchDir).squaredNorm();
        double testingE;
//        Eigen::VectorXd testingG;
        computeEnergyVal(testingData, testingE);
//        computeGradient(testingData, testingG);
//        if(!mute) {
//            logFile << "searchDir " << searchDir.norm() << std::endl;
//            logFile << "testingE" << globalIterNum << " " << testingE << " > " << lastEnergyVal << " " << stepSize * c1m << std::endl;
//            logFile << "testingG" << globalIterNum << " " << searchDir.dot(testingG) << " < " << c2m << std::endl;
//        }
//        while((testingE > lastEnergyVal + stepSize * c1m) ||
//              (searchDir.dot(testingG) < c2m)) // Wolfe condition
        while(testingE > lastEnergyVal + stepSize * c1m) // Armijo condition
//        while(0)
        {
            stepSize /= 2.0;
//            stepLen = (stepSize * searchDir).squaredNorm();
//            if(stepLen < targetGRes) {
            if(stepSize == 0.0) {
                stopped = true;
                if(!mute) {
                    logFile << "testingE" << globalIterNum << " " << testingE << " > " << lastEnergyVal << " " << stepSize * c1m << std::endl;
//                    logFile << "testingG" << globalIterNum << " " << searchDir.dot(testingG) << " < " << c2m << std::endl;
                }
                break;
            }
            
            stepForward(testingData, stepSize);
            computeEnergyVal(testingData, testingE);
//            computeGradient(testingData, testingG);
        }
//        if(!mute) {
//            logFile << "testingE" << globalIterNum << " " << testingE << " > " << lastEnergyVal << " " << stepSize * c1m << std::endl;
//            logFile << "testingG" << globalIterNum << " " << searchDir.dot(testingG) << " < " << c2m << std::endl;
//        }
        while(!testingData.checkInversion()) {
            stepSize /= 2.0;
            if(stepSize == 0.0) {
                assert(0 && "line search failed!");
                stopped = true;
                break;
            }
            
            stepForward(testingData, stepSize);
            computeEnergyVal(testingData, testingE);
        }
        result.V = testingData.V;
        lastEDec = lastEnergyVal - testingE;
        if(allowEDecRelTol && (lastEDec / lastEnergyVal / stepSize < 1.0e-6)) {
            // no prominent energy decrease, stop for accelerating the process
            stopped = true;
//            std::cout << "no prominant energy decrease, optimization stops" << std::endl;
        }
        lastEnergyVal = testingE;
        
        if(!mute) {
            std::cout << stepSize << std::endl;
            std::cout << "stepLen = " << (stepSize * searchDir).squaredNorm() << std::endl;
            std::cout << "E_cur_smooth = " << testingE << std::endl;
            
            double seamSparsity;
            result.computeSeamSparsity(seamSparsity, !fractureMode);
            seamSparsity /= result.virtualRadius;
            if(fractureMode) {
                file_energyValPerIter << lastEnergyVal + (1.0 - energyParams[0]) * seamSparsity;
            }
            else {
                file_energyValPerIter << lastEnergyVal;
            }
            for(int eI = 0; eI < energyTerms.size(); eI++) {
                file_energyValPerIter << " " << energyVal_ET[eI];
            }
            file_energyValPerIter << " " << seamSparsity << std::endl;
        }
        
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
    
    void Optimizer::updateTargetGRes(void)
    {
//        targetGRes = energyParamSum * (data0.V_rest.rows() - data0.fixedVert.size()) * relGL2Tol * data0.avgEdgeLen * data0.avgEdgeLen;
        targetGRes = energyParamSum * static_cast<double>(data0.V_rest.rows() - data0.fixedVert.size()) / static_cast<double>(data0.V_rest.rows()) * relGL2Tol;
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
        energyVal = energyParams[0] * energyVal_ET[0];
        for(int eI = 1; eI < energyTerms.size(); eI++) {
            energyTerms[eI]->computeEnergyVal(data, energyVal_ET[eI]);
            energyVal += energyParams[eI] * energyVal_ET[eI];
        }
    }
    void Optimizer::computeGradient(const TriangleSoup& data, Eigen::VectorXd& gradient)
    {
        energyTerms[0]->computeGradient(data, gradient_ET[0]);
        gradient = energyParams[0] * gradient_ET[0];
        for(int eI = 1; eI < energyTerms.size(); eI++) {
            energyTerms[eI]->computeGradient(data, gradient_ET[eI]);
            gradient += energyParams[eI] * gradient_ET[eI];
        }
    }
    void Optimizer::computePrecondMtr(const TriangleSoup& data, Eigen::SparseMatrix<double>& precondMtr)
    {
        if(pardisoThreadAmt) {
            I_mtr.resize(0);
            J_mtr.resize(0);
            V_mtr.resize(0);
            for(int eI = 0; eI < energyTerms.size(); eI++) {
                Eigen::VectorXi I, J;
                Eigen::VectorXd V;
                energyTerms[eI]->computePrecondMtr(data, &V, &I, &J);
                V *= energyParams[eI];
                I_mtr.conservativeResize(I_mtr.size() + I.size());
                I_mtr.bottomRows(I.size()) = I;
                J_mtr.conservativeResize(J_mtr.size() + J.size());
                J_mtr.bottomRows(J.size()) = J;
                V_mtr.conservativeResize(V_mtr.size() + V.size());
                V_mtr.bottomRows(V.size()) = V;
            }
//            IglUtils::writeSparseMatrixToFile("/Users/mincli/Desktop/FracCuts/mtr", I_mtr, J_mtr, V_mtr, true);
        }
        else {
            //TODO: triplet representation for eigen matrices
            precondMtr.setZero();
            energyTerms[0]->computePrecondMtr(data, precondMtr);
            precondMtr *= energyParams[0];
            for(int eI = 1; eI < energyTerms.size(); eI++) {
                Eigen::SparseMatrix<double> precondMtrI;
                energyTerms[eI]->computePrecondMtr(data, precondMtrI);
                if(precondMtrI.rows() == precondMtr.rows() * 2) {
                    precondMtrI *= energyParams[eI];
                    for (int k = 0; k < precondMtr.outerSize(); ++k)
                    {
                        for (Eigen::SparseMatrix<double>::InnerIterator it(precondMtr, k); it; ++it)
                        {
                            precondMtrI.coeffRef(it.row() * 2, it.col() * 2) += it.value();
                            precondMtrI.coeffRef(it.row() * 2 + 1, it.col() * 2 + 1) += it.value();
                        }
                    }
                    precondMtr = precondMtrI;
                }
                else if(precondMtrI.rows() * 2 == precondMtr.rows()) {
                    for (int k = 0; k < precondMtrI.outerSize(); ++k)
                    {
                        for (Eigen::SparseMatrix<double>::InnerIterator it(precondMtrI, k); it; ++it)
                        {
                            precondMtr.coeffRef(it.row() * 2, it.col() * 2) += energyParams[eI] * it.value();
                            precondMtr.coeffRef(it.row() * 2 + 1, it.col() * 2 + 1) += energyParams[eI] * it.value();
                        }
                    }
                }
                else {
                    assert(precondMtrI.rows() == precondMtr.rows());
                    precondMtr += energyParams[eI] * precondMtrI;
                }
            }
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
    
    double Optimizer::getLastEnergyVal(void) const
    {
        return lastEnergyVal;
    }
}
