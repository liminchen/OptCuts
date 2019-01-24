//
//  Optimizer.cpp
//  OptCuts
//
//  Created by Minchen Li on 8/31/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#include "Optimizer.hpp"
#include "SymDirichletEnergy.hpp"
#include "IglUtils.hpp"
#include "Timer.hpp"

#ifdef LINSYSSOLVER_USE_CHOLMOD
#include "CHOLMODSolver.hpp"
#elif defined(LINSYSSOLVER_USE_PARDISO)
#include "PardisoSolver.hpp"
#else
#include "EigenLibSolver.hpp"
#endif

#include <igl/avg_edge_length.h>

#include <fstream>
#include <iostream>
#include <string>
#include <numeric>

extern OptCuts::MethodType methodType;
extern const std::string outputFolderPath;
extern const bool fractureMode;

extern std::ofstream logFile;
extern Timer timer, timer_step;

namespace OptCuts {
    
    Optimizer::Optimizer(const TriMesh& p_data0,
                         const std::vector<Energy*>& p_energyTerms, const std::vector<double>& p_energyParams,
                         int p_propagateFracture, bool p_mute, bool p_scaffolding,
                         const Eigen::MatrixXd& UV_bnds, const Eigen::MatrixXi& E, const Eigen::VectorXi& bnd,
                         bool p_useDense) :
        data0(p_data0), energyTerms(p_energyTerms), energyParams(p_energyParams)
    {
        assert(energyTerms.size() == energyParams.size());
        
        useDense = p_useDense;
        
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
        relGL2Tol = 1.0e-12;
        topoIter = 0;
        
        needRefactorize = false;
        for(const auto& energyTermI : energyTerms) {
            if(energyTermI->getNeedRefactorize()) {
                needRefactorize = true;
                break;
            }
        }
        
        pardisoThreadAmt = 4;
        
        scaffolding = p_scaffolding;
        UV_bnds_scaffold = UV_bnds;
        E_scaffold = E;
        bnd_scaffold = bnd;
        w_scaf = energyParams[0] * 0.01;
        
#ifdef LINSYSSOLVER_USE_CHOLMOD
        linSysSolver = new CHOLMODSolver<Eigen::VectorXi, Eigen::VectorXd>();
#elif defined(LINSYSSOLVER_USE_PARDISO)
        linSysSolver = new PardisoSolver<Eigen::VectorXi, Eigen::VectorXd>();
#else
        linSysSolver = new EigenLibSolver<Eigen::VectorXi, Eigen::VectorXd>();
#endif
    }
    
    Optimizer::~Optimizer(void)
    {
        if(file_energyValPerIter.is_open()) {
            file_energyValPerIter.close();
        }
        if(file_gradientPerIter.is_open()) {
            file_gradientPerIter.close();
        }
        delete linSysSolver;
    }
    
    void Optimizer::computeLastEnergyVal(void)
    {
        computeEnergyVal(result, scaffold, lastEnergyVal);
    }
    
    TriMesh& Optimizer::getResult(void) {
        return result;
    }
    
    const Scaffold& Optimizer::getScaffold(void) const {
        return scaffold;
    }
    
    const TriMesh& Optimizer::getAirMesh(void) const {
        return scaffold.airMesh;
    }
    
    bool Optimizer::isScaffolding(void) const {
        return scaffolding;
    }
    
    const TriMesh& Optimizer::getData_findExtrema(void) const {
        return data_findExtrema;
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
        result = data0;
        if(scaffolding) {
            scaffold = Scaffold(result, UV_bnds_scaffold, E_scaffold, bnd_scaffold);
            result.scaffold = &scaffold;
            scaffold.mergeVNeighbor(result.vNeighbor, vNeighbor_withScaf);
            scaffold.mergeFixedV(result.fixedVert, fixedV_withScaf);
        }
        
        computeHessian(result, scaffold);
        
        if(useDense) {
            if(!needRefactorize) {
                denseSolver = Hessian.ldlt();
            }
        }
        else {
            if(!mute) { timer_step.start(1); }
            linSysSolver->set_type(pardisoThreadAmt, -2);
            linSysSolver->set_pattern(scaffolding ? vNeighbor_withScaf : result.vNeighbor,
                                      scaffolding ? fixedV_withScaf : result.fixedVert);
            linSysSolver->update_a(I_mtr, J_mtr, V_mtr);
            if(!mute) { timer_step.stop(); timer_step.start(2); }
            linSysSolver->analyze_pattern();
            if(!mute) { timer_step.stop(); }
            if(!needRefactorize) {
                try {
                    if(!mute) { timer_step.start(3); }
                    linSysSolver->factorize();
                    if(!mute) { timer_step.stop(); }
                }
                catch(std::exception e) {
                    IglUtils::writeSparseMatrixToFile(outputFolderPath + "mtr_factorizeFail", I_mtr, J_mtr, V_mtr, true);
                    exit(-1);
                }
            }
        }
        
        lastEDec = 0.0;
        data_findExtrema = data0;
        updateTargetGRes();
        computeEnergyVal(result, scaffold, lastEnergyVal);
        if(!mute) {
            writeEnergyValToFile(true);
            std::cout << "E_initial = " << lastEnergyVal << std::endl;
        }
    }
    
    int Optimizer::solve(int maxIter)
    {
        static bool lastPropagate = false;
        for(int iterI = 0; iterI < maxIter; iterI++)
        {
            if(!mute) { timer.start(1); }
            computeGradient(result, scaffold, gradient);
            const double sqn_g = gradient.squaredNorm();
            if(!mute) {
                std::cout << "||gradient||^2 = " << sqn_g << ", targetGRes = " << targetGRes << std::endl;
                writeGradL2NormToFile(false);
            }
            if(sqn_g < targetGRes) {
                // converged
                lastEDec = 0.0;
                globalIterNum++;
                if(!mute) { timer.stop(); }
                return 1;
            }
            else {
                if(solve_oneStep()) {
                    globalIterNum++;
                    if(!mute) { timer.stop(); }
                    return 1;
                }
            }
            globalIterNum++;
            if(!mute) { timer.stop(); }
            
            if(propagateFracture > 0) {
                if(!createFracture(lastEDec, propagateFracture)) {
                    // always perform the one decreasing E_w more
                    if(scaffolding) {
                        scaffold = Scaffold(result, UV_bnds_scaffold, E_scaffold, bnd_scaffold);
                        result.scaffold = &scaffold;
                        scaffold.mergeVNeighbor(result.vNeighbor, vNeighbor_withScaf);
                        scaffold.mergeFixedV(result.fixedVert, fixedV_withScaf);
                    }
                    
                    if(lastPropagate) {
                        lastPropagate = false;
                        return 2; // for saving screenshots
                    }
                }
                else {
                    lastPropagate = true;
                }
            }
            else {
                if(scaffolding) {
                    scaffold = Scaffold(result, UV_bnds_scaffold, E_scaffold, bnd_scaffold);
                    result.scaffold = &scaffold;
                    scaffold.mergeVNeighbor(result.vNeighbor, vNeighbor_withScaf);
                    scaffold.mergeFixedV(result.fixedVert, fixedV_withScaf);
                }
            }
        }
        return 0;
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
        computeHessian(result, scaffold);
        
        if(useDense) {
            denseSolver = Hessian.ldlt();
        }
        else {
            if(!mute) { timer_step.start(1); }
            linSysSolver->update_a(I_mtr, J_mtr, V_mtr);
            if(!mute) { timer_step.stop(); timer_step.start(3); }
            linSysSolver->factorize();
            if(!mute) { timer_step.stop(); }
        }
    }
    
    void Optimizer::setConfig(const TriMesh& config, int iterNum, int p_topoIter)
    {
        topoIter = p_topoIter;
        globalIterNum = iterNum;
        result = config;
        if(scaffolding) {
            scaffold = Scaffold(result, UV_bnds_scaffold, E_scaffold, bnd_scaffold);
            result.scaffold = &scaffold;
            scaffold.mergeVNeighbor(result.vNeighbor, vNeighbor_withScaf);
            scaffold.mergeFixedV(result.fixedVert, fixedV_withScaf);
        }
        
        updateEnergyData();
    }
    
    void Optimizer::setPropagateFracture(bool p_prop)
    {
        propagateFracture = p_prop;
    }
    
    void Optimizer::setScaffolding(bool p_scaffolding)
    {
        scaffolding = p_scaffolding;
        if(scaffolding) {
            scaffold = Scaffold(result, UV_bnds_scaffold, E_scaffold, bnd_scaffold);
            result.scaffold = &scaffold;
            scaffold.mergeVNeighbor(result.vNeighbor, vNeighbor_withScaf);
            scaffold.mergeFixedV(result.fixedVert, fixedV_withScaf);
        }
    }
    
    void Optimizer::setUseDense(bool p_useDense)
    {
        useDense = p_useDense;
    }
    
    void Optimizer::updateEnergyData(bool updateEVal, bool updateGradient, bool updateHessian)
    {
        energyParamSum = 0.0;
        for(const auto& ePI : energyParams) {
            energyParamSum += ePI;
        }
        updateTargetGRes();
        
        if(updateEVal) {
            // compute energy and output
            computeEnergyVal(result, scaffold, lastEnergyVal);
        }
        
        if(updateGradient) {
            // compute gradient and output
            computeGradient(result, scaffold, gradient);
            if(gradient.squaredNorm() < targetGRes) {
                logFile << "||g||^2 = " << gradient.squaredNorm() << " after fracture initiation!" << std::endl;
            }
        }
        
        if(updateHessian) {
            // for the changing hessian
            if(!mute) {
                std::cout << "recompute proxy/Hessian matrix and factorize..." << std::endl;
            }
            computeHessian(result, scaffold);
            
            if(useDense) {
                if(!needRefactorize) {
                    denseSolver = Hessian.ldlt();
                }
            }
            else {
                if(!mute) { timer_step.start(1); }
                linSysSolver->set_pattern(scaffolding ? vNeighbor_withScaf : result.vNeighbor,
                                          scaffolding ? fixedV_withScaf : result.fixedVert);
                linSysSolver->update_a(I_mtr, J_mtr, V_mtr);
                if(!mute) { timer_step.stop(); timer_step.start(2); }
                linSysSolver->analyze_pattern();
                if(!mute) { timer_step.stop(); }
                if(!needRefactorize) {
                    if(!mute) { timer_step.start(3); }
                    linSysSolver->factorize();
                    if(!mute) { timer_step.stop(); }
                }
            }
        }
    }
    
    bool Optimizer::createFracture(int opType, const std::vector<int>& path, const Eigen::MatrixXd& newVertPos, bool allowPropagate)
    {
        assert(methodType == MT_OPTCUTS);
        
        topoIter++;
        
        timer.start(0);
        bool isMerge = false;
        data_findExtrema = result; // potentially time-consuming
        switch(opType) {
            case 0: // boundary split
                std::cout << "boundary split without querying again" << std::endl;
                result.splitEdgeOnBoundary(std::pair<int, int>(path[0], path[1]), newVertPos);
                logFile << "boundary edge splitted without querying again" << std::endl;
                //TODO: process fractail here!
                result.updateFeatures();
                break;
                
            case 1: // interior split
                std::cout << "Interior split without querying again" << std::endl;
                result.cutPath(path, true, 1, newVertPos);
                logFile << "interior edge splitted without querying again" << std::endl;
                result.fracTail.insert(path[0]);
                result.fracTail.insert(path[2]);
                result.curInteriorFracTails.first = path[0];
                result.curInteriorFracTails.second = path[2];
                result.curFracTail = -1;
                break;
                
            case 2: // merge
                std::cout << "corner edge merged without querying again" << std::endl;
                result.mergeBoundaryEdges(std::pair<int, int>(path[0], path[1]),
                                          std::pair<int, int>(path[1], path[2]), newVertPos.row(0));
                logFile << "corner edge merged without querying again" << std::endl;
                
                result.computeFeatures(); //TODO: only update locally
                isMerge = true;
                break;
                
            default:
                assert(0);
                break;
        }
        timer.stop();
        
        if(scaffolding) {
            scaffold = Scaffold(result, UV_bnds_scaffold, E_scaffold, bnd_scaffold);
            result.scaffold = &scaffold;
            scaffold.mergeVNeighbor(result.vNeighbor, vNeighbor_withScaf);
            scaffold.mergeFixedV(result.fixedVert, fixedV_withScaf);
        }
        
        timer.start(3);
        updateEnergyData(true, false, true);
        timer.stop();
        fractureInitiated = true;
        if(!mute) {
            writeEnergyValToFile(false);
        }
        
        if(allowPropagate) {
            propagateFracture = 1 + isMerge;
        }

        return true;
    }
    
    bool Optimizer::createFracture(double stressThres, int propType, bool allowPropagate, bool allowInSplit)
    {
        if(propType == 0) {
            topoIter++;
        }
        
        timer.start(0);
        bool changed = false;
        bool isMerge = false;
        switch(methodType) {
            case MT_OPTCUTS_NODUAL:
            case MT_OPTCUTS: {
                data_findExtrema = result;
                switch(propType) {
                    case 0: // initiation
                        changed = result.splitOrMerge(1.0 - energyParams[0], stressThres, false, allowInSplit, isMerge);
                        break;
                        
                    case 1: // propagate split
                        changed = result.splitEdge(1.0 - energyParams[0], stressThres, true, allowInSplit);
                        break;
                        
                    case 2: // propagate merge
                        changed = result.mergeEdge(1.0 - energyParams[0], stressThres, true);
                        isMerge = true;
                        break;
                }
                break;
            }
                
            case MT_EBCUTS:
                result.geomImgCut(data_findExtrema);
                allowPropagate = false;
                changed = true;
                break;
                
            default:
                assert(0 && "Fracture forbiddened for current method type!");
                break;
        }
        timer.stop();
        
        if(changed) {
            if(scaffolding) {
                scaffold = Scaffold(result, UV_bnds_scaffold, E_scaffold, bnd_scaffold);
                result.scaffold = &scaffold;
                scaffold.mergeVNeighbor(result.vNeighbor, vNeighbor_withScaf);
                scaffold.mergeFixedV(result.fixedVert, fixedV_withScaf);
            }
            
            timer.start(3);
            updateEnergyData(true, false, true);
            timer.stop();
            fractureInitiated = true;
            if((!mute) && (propType == 0)) {
                writeEnergyValToFile(false);
            }
            
            if(allowPropagate && (propType == 0)) {
                propagateFracture = 1 + isMerge;
            }
        }
        return changed;
    }
    
    bool Optimizer::solve_oneStep(void)
    {
        if(needRefactorize) {
            // for the changing hessian
            if(!mute) {
                std::cout << "recompute proxy/Hessian matrix..." << std::endl;
            }
            if(!fractureInitiated) {
                computeHessian(result, scaffold);
            }
            
            if(!mute) {
                std::cout << "factorizing proxy/Hessian matrix..." << std::endl;
            }
            
            if(!fractureInitiated) {
                if(!useDense) {
                    if(scaffolding) {
                        if(!mute) { timer_step.start(1); }
                        linSysSolver->set_pattern(scaffolding ? vNeighbor_withScaf : result.vNeighbor,
                                                  scaffolding ? fixedV_withScaf : result.fixedVert);
                        linSysSolver->update_a(I_mtr, J_mtr, V_mtr);
                        if(!mute) { timer_step.stop(); timer_step.start(2); }
                        linSysSolver->analyze_pattern();
                        if(!mute) { timer_step.stop(); }
                    }
                    else {
                        if(!mute) { timer_step.start(1); }
                        linSysSolver->update_a(I_mtr, J_mtr, V_mtr);
                        if(!mute) { timer_step.stop(); }
                    }
                }
            }
            try {
                if(!mute) { timer_step.start(3); }
                if(useDense) {
                    denseSolver = Hessian.ldlt();
                }
                else {
                    linSysSolver->factorize();
                }
                if(!mute) { timer_step.stop(); }
            }
            catch(std::exception e) {
                if(!useDense) {
                    IglUtils::writeSparseMatrixToFile(outputFolderPath + "mtr",
                                                      I_mtr, J_mtr, V_mtr, true);
                }
                exit(-1);
            }
        }
        
        Eigen::VectorXd minusG = -gradient;
        if(!mute) { timer_step.start(4); }
        if(useDense) {
            searchDir = denseSolver.solve(minusG);
        }
        else {
            linSysSolver->solve(minusG, searchDir);
        }
        if(!mute) { timer_step.stop(); }
        
        fractureInitiated = false;
        
        if(!mute) { timer_step.start(5); }
        bool stopped = lineSearch();
        if(!mute) { timer_step.stop(); }
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
        
        double lastEnergyVal_scaffold = 0.0;
        Eigen::MatrixXd resultV0 = result.V;
        Eigen::MatrixXd scaffoldV0;
        if(scaffolding) {
            scaffoldV0 = scaffold.airMesh.V;
            computeEnergyVal(result, scaffold, lastEnergyVal); // this update is necessary since scaffold changes
            lastEnergyVal_scaffold = energyVal_scaffold;
        }
        stepForward(resultV0, scaffoldV0, result, scaffold, stepSize);
        double testingE;
        computeEnergyVal(result, scaffold, testingE);
        
        while(testingE > lastEnergyVal) // ensure energy decrease
        {
            stepSize /= 2.0;
            if(stepSize == 0.0) {
                stopped = true;
                if(!mute) {
                    logFile << "testingE" << globalIterNum << " " << testingE << " > " << lastEnergyVal << std::endl;
                }
                break;
            }
            
            stepForward(resultV0, scaffoldV0, result, scaffold, stepSize);
            computeEnergyVal(result, scaffold, testingE);
        }
        if(!mute) {
            std::cout << stepSize << "(armijo) ";
        }

        while((!result.checkInversion()) ||
              ((scaffolding) && (!scaffold.airMesh.checkInversion())))
        {
            assert(0 && "element inversion after armijo shouldn't happen!");
            
            stepSize /= 2.0;
            if(stepSize == 0.0) {
                assert(0 && "line search failed!");
                stopped = true;
                break;
            }
            
            stepForward(resultV0, scaffoldV0, result, scaffold, stepSize);
            computeEnergyVal(result, scaffold, testingE);
        }
        
        lastEDec = lastEnergyVal - testingE;
        if(scaffolding) {
            lastEDec += (-lastEnergyVal_scaffold + energyVal_scaffold);
        }
        if(allowEDecRelTol && (lastEDec / lastEnergyVal < 1.0e-6 * stepSize) && (stepSize > 1.0e-3))
        { // avoid stopping in hard situations
            stopped = true;
        }
        lastEnergyVal = testingE;
        
        if(!mute) {
            std::cout << stepSize << std::endl;
            std::cout << "stepLen = " << (stepSize * searchDir).squaredNorm() << std::endl;
            std::cout << "E_cur_smooth = " << testingE - energyVal_scaffold << std::endl;

            if(!stopped) {
                writeEnergyValToFile(false);
            }
        }
        
        return stopped;
    }
    
    void Optimizer::stepForward(const Eigen::MatrixXd& dataV0, const Eigen::MatrixXd& scaffoldV0,
                                TriMesh& data, Scaffold& scaffoldData, double stepSize) const
    {
        assert(dataV0.rows() == data.V.rows());
        if(scaffolding) {
            assert(data.V.rows() + scaffoldData.airMesh.V.rows() - scaffoldData.bnd.size() == searchDir.size() / 2);
        }
        else {
            assert(data.V.rows() * 2 == searchDir.size());
        }
        assert(data.V.rows() == result.V.rows());
        
        for(int vI = 0; vI < data.V.rows(); vI++) {
            data.V(vI, 0) = dataV0(vI, 0) + stepSize * searchDir[vI * 2];
            data.V(vI, 1) = dataV0(vI, 1) + stepSize * searchDir[vI * 2 + 1];
        }
        if(scaffolding) {
            scaffoldData.stepForward(scaffoldV0, searchDir, stepSize);
        }
    }
    
    void Optimizer::updateTargetGRes(void)
    {
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
    
    void Optimizer::initStepSize(const TriMesh& data, double& stepSize) const
    {
        for(int eI = 0; eI < energyTerms.size(); eI++) {
            energyTerms[eI]->initStepSize(data, searchDir, stepSize);
        }
        
        if(scaffolding) {
            Eigen::VectorXd searchDir_scaffold;
            scaffold.wholeSearchDir2airMesh(searchDir, searchDir_scaffold);
            SymDirichletEnergy SD;
            SD.initStepSize(scaffold.airMesh, searchDir_scaffold, stepSize);
        }
    }
    
    void Optimizer::writeEnergyValToFile(bool flush)
    {
        double E_se;
        result.computeSeamSparsity(E_se, !fractureMode);
        E_se /= result.virtualRadius;
        
        if(fractureMode) {
            buffer_energyValPerIter << lastEnergyVal + (1.0 - energyParams[0]) * E_se;
        }
        else {
            buffer_energyValPerIter << lastEnergyVal;
        }
        
        for(int eI = 0; eI < energyTerms.size(); eI++) {
            buffer_energyValPerIter << " " << energyVal_ET[eI];
        }
        
        buffer_energyValPerIter << " " << E_se << " " << energyParams[0] << "\n";
        
        if(flush) {
            flushEnergyFileOutput();
        }
    }
    void Optimizer::writeGradL2NormToFile(bool flush)
    {
        buffer_gradientPerIter << gradient.squaredNorm();
        for(int eI = 0; eI < energyTerms.size(); eI++) {
            buffer_gradientPerIter << " " << gradient_ET[eI].squaredNorm();
        }
        buffer_gradientPerIter << "\n";
        
        if(flush) {
            flushGradFileOutput();
        }
    }
    void Optimizer::flushEnergyFileOutput(void)
    {
        file_energyValPerIter << buffer_energyValPerIter.str();
        file_energyValPerIter.flush();
        clearEnergyFileOutputBuffer();
    }
    void Optimizer::flushGradFileOutput(void)
    {
        file_gradientPerIter << buffer_gradientPerIter.str();
        file_gradientPerIter.flush();
        clearGradFileOutputBuffer();
    }
    void Optimizer::clearEnergyFileOutputBuffer(void)
    {
        buffer_energyValPerIter.str("");
        buffer_energyValPerIter.clear();
    }
    void Optimizer::clearGradFileOutputBuffer(void)
    {
        buffer_gradientPerIter.str("");
        buffer_gradientPerIter.clear();
    }
    
    void Optimizer::computeEnergyVal(const TriMesh& data, const Scaffold& scaffoldData, double& energyVal, bool excludeScaffold)
    {
        energyTerms[0]->computeEnergyVal(data, energyVal_ET[0]);
        energyVal = energyParams[0] * energyVal_ET[0];
        for(int eI = 1; eI < energyTerms.size(); eI++) {
            energyTerms[eI]->computeEnergyVal(data, energyVal_ET[eI]);
            energyVal += energyParams[eI] * energyVal_ET[eI];
        }
        
        if(scaffolding && (!excludeScaffold)) {
            SymDirichletEnergy SD;
            SD.computeEnergyVal(scaffoldData.airMesh, energyVal_scaffold, true);
            energyVal_scaffold *= w_scaf / scaffold.airMesh.F.rows();
            energyVal += energyVal_scaffold;
        }
        else {
            energyVal_scaffold = 0.0;
        }
    }
    void Optimizer::computeGradient(const TriMesh& data, const Scaffold& scaffoldData, Eigen::VectorXd& gradient, bool excludeScaffold)
    {
        energyTerms[0]->computeGradient(data, gradient_ET[0]);
        gradient = energyParams[0] * gradient_ET[0];
        for(int eI = 1; eI < energyTerms.size(); eI++) {
            energyTerms[eI]->computeGradient(data, gradient_ET[eI]);
            gradient += energyParams[eI] * gradient_ET[eI];
        }
        
        if(scaffolding) {
            SymDirichletEnergy SD;
            SD.computeGradient(scaffoldData.airMesh, gradient_scaffold, true);
            scaffoldData.augmentGradient(gradient, gradient_scaffold, (excludeScaffold ? 0.0 : (w_scaf / scaffold.airMesh.F.rows())));
        }
    }
    void Optimizer::computeHessian(const TriMesh& data, const Scaffold& scaffoldData)
    {
        if(!mute) { timer_step.start(0); }
        if(useDense) {
            energyTerms[0]->computeHessian(data, Hessian);
            Hessian *= energyParams[0];
            for(int eI = 1; eI < energyTerms.size(); eI++) {
                Eigen::MatrixXd HessianI;
                energyTerms[eI]->computeHessian(data, HessianI);
                Hessian += energyParams[eI] * HessianI;
            }
            
            if(scaffolding) {
                SymDirichletEnergy SD;
                Eigen::MatrixXd Hessian_scaf;
                SD.computeHessian(scaffoldData.airMesh, Hessian_scaf, true);
                scaffoldData.augmentProxyMatrix(Hessian, Hessian_scaf, w_scaf / scaffold.airMesh.F.rows());
            }
        }
        else {
            I_mtr.resize(0);
            J_mtr.resize(0);
            V_mtr.resize(0);
            for(int eI = 0; eI < energyTerms.size(); eI++) {
                Eigen::VectorXi I, J;
                Eigen::VectorXd V;
                energyTerms[eI]->computeHessian(data, &V, &I, &J);
                V *= energyParams[eI];
                I_mtr.conservativeResize(I_mtr.size() + I.size());
                I_mtr.bottomRows(I.size()) = I;
                J_mtr.conservativeResize(J_mtr.size() + J.size());
                J_mtr.bottomRows(J.size()) = J;
                V_mtr.conservativeResize(V_mtr.size() + V.size());
                V_mtr.bottomRows(V.size()) = V;
            }
            
            if(scaffolding) {
                SymDirichletEnergy SD;
                Eigen::VectorXi I, J;
                Eigen::VectorXd V;
                SD.computeHessian(scaffoldData.airMesh, &V, &I, &J, true);
                scaffoldData.augmentProxyMatrix(I_mtr, J_mtr, V_mtr, I, J, V, w_scaf / scaffold.airMesh.F.rows());
            }
        }
        if(!mute) { timer_step.stop(); }
    }
    
    double Optimizer::getLastEnergyVal(bool excludeScaffold) const
    {
        return ((excludeScaffold && scaffolding) ?
                (lastEnergyVal - energyVal_scaffold) :
                lastEnergyVal);
    }
}
