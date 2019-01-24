//
//  Optimizer.hpp
//  OptCuts
//
//  Created by Minchen Li on 8/31/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#ifndef Optimizer_hpp
#define Optimizer_hpp

#include "Types.hpp"
#include "Energy.hpp"
#include "Scaffold.hpp"

#include "LinSysSolver.hpp"

#include <fstream>

namespace OptCuts {
    
    // a class for solving an optimization problem
    class Optimizer {
        friend class TriMesh;
        
    protected: // referenced data
        const TriMesh& data0; // initial guess
        const std::vector<Energy*>& energyTerms; // E_0, E_1, E_2, ...
        const std::vector<double>& energyParams; // a_0, a_1, a_2, ...
        // E = \Sigma_i a_i E_i
        
    protected: // owned data
        bool useDense = false;
        int propagateFracture;
        bool fractureInitiated = false;
        bool allowEDecRelTol;
        bool mute;
        bool pardisoThreadAmt;
        bool needRefactorize;
        int globalIterNum;
        int topoIter;
        double relGL2Tol, energyParamSum;
        TriMesh result; // intermediate results of each iteration
        TriMesh data_findExtrema; // intermediate results for deciding the cuts in each topology step
        bool scaffolding; // whether to enable bijectivity parameterization
        double w_scaf;
        Scaffold scaffold; // air meshes to enforce bijectivity
        Eigen::VectorXi I_mtr, J_mtr; // triplet representation
        Eigen::VectorXd V_mtr;
        Eigen::MatrixXd Hessian; // when using dense representation
        // cholesky solver for solving the linear system for search directions
        LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* linSysSolver;
        Eigen::LDLT<Eigen::MatrixXd> denseSolver;
        Eigen::VectorXd gradient; // energy gradient computed in each iteration
        Eigen::VectorXd searchDir; // search direction comptued in each iteration
        double lastEnergyVal; // for output and line search
        double lastEDec;
        double targetGRes;
        std::vector<Eigen::VectorXd> gradient_ET;
        Eigen::VectorXd gradient_scaffold;
        std::vector<double> energyVal_ET;
        double energyVal_scaffold;
        
        Eigen::MatrixXd UV_bnds_scaffold;
        Eigen::MatrixXi E_scaffold;
        Eigen::VectorXi bnd_scaffold;
        std::vector<std::set<int>> vNeighbor_withScaf;
        std::set<int> fixedV_withScaf;
        
        std::ostringstream buffer_energyValPerIter;
        std::ostringstream buffer_gradientPerIter;
        std::ofstream file_energyValPerIter;
        std::ofstream file_gradientPerIter;
        
    public: // constructor and destructor
        Optimizer(const TriMesh& p_data0, const std::vector<Energy*>& p_energyTerms, const std::vector<double>& p_energyParams,
                  int p_propagateFracture = 1, bool p_mute = false, bool p_scaffolding = false,
                  const Eigen::MatrixXd& UV_bnds = Eigen::MatrixXd(),
                  const Eigen::MatrixXi& E = Eigen::MatrixXi(),
                  const Eigen::VectorXi& bnd = Eigen::VectorXi(),
                  bool p_useDense = false);
        ~Optimizer(void);
        
    public: // API
        // precompute preconditioning matrix and factorize for fast solve, prepare initial guess
        void precompute(void);
        
        // solve the optimization problem that minimizes E using a hill-climbing method,
        // the final result will be in result
        int solve(int maxIter = 100);
        
        void updatePrecondMtrAndFactorize(void);
        
        void updateEnergyData(bool updateEVal = true, bool updateGradient = true, bool updateHessian = true);
        bool createFracture(double stressThres, int propType,
                            bool allowPropagate = true, bool allowInSplit = false);
        bool createFracture(int opType, const std::vector<int>& path, const Eigen::MatrixXd& newVertPos, bool allowPropagate);
        void setConfig(const TriMesh& config, int iterNum, int p_topoIter);
        void setPropagateFracture(bool p_prop);
        void setScaffolding(bool p_scaffolding);
        void setUseDense(bool p_useDense = true);
        
        void computeLastEnergyVal(void);
        
        void getGradientVisual(Eigen::MatrixXd& arrowVec) const;
        TriMesh& getResult(void);
        const Scaffold& getScaffold(void) const;
        const TriMesh& getAirMesh(void) const;
        bool isScaffolding(void) const;
        const TriMesh& getData_findExtrema(void) const;
        int getIterNum(void) const;
        int getTopoIter(void) const;
        void setRelGL2Tol(double p_relTol);
        void setAllowEDecRelTol(bool p_allowEDecRelTol);
        
        void flushEnergyFileOutput(void);
        void flushGradFileOutput(void);
        void clearEnergyFileOutputBuffer(void);
        void clearGradFileOutputBuffer(void);
        
    protected: // helper functions
        // solve for new configuration in the next iteration
        //NOTE: must compute current gradient first
        bool solve_oneStep(void);
        
        bool lineSearch(void);

        void stepForward(const Eigen::MatrixXd& dataV0, const Eigen::MatrixXd& scaffoldV0,
                         TriMesh& data, Scaffold& scaffoldData, double stepSize) const;
        
        void updateTargetGRes(void);
        
        void computeEnergyVal(const TriMesh& data, const Scaffold& scaffoldData, double& energyVal, bool excludeScaffold = false);
        void computeGradient(const TriMesh& data, const Scaffold& scaffoldData, Eigen::VectorXd& gradient, bool excludeScaffold = false);
        void computeHessian(const TriMesh& data, const Scaffold& scaffoldData);
        
        void initStepSize(const TriMesh& data, double& stepSize) const;
        
        void writeEnergyValToFile(bool flush);
        void writeGradL2NormToFile(bool flush);
        
    public: // data access
        double getLastEnergyVal(bool excludeScaffold = false) const;
    };
    
}

#endif /* Optimizer_hpp */
