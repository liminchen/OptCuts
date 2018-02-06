//
//  Optimizer.hpp
//  FracCuts
//
//  Created by Minchen Li on 8/31/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#ifndef Optimizer_hpp
#define Optimizer_hpp

#include "Types.hpp"
#include "Energy.hpp"

#include "PardisoSolver.hpp"

#include <fstream>

namespace FracCuts {
    
    // a class for solving an optimization problem
    class Optimizer {
    protected: // referenced data
        const TriangleSoup& data0; // initial guess
        const std::vector<Energy*>& energyTerms; // E_0, E_1, E_2, ...
        const std::vector<double>& energyParams; // a_0, a_1, a_2, ...
        // E = \Sigma_i a_i E_i
        
    protected: // owned data
        bool propagateFracture;
        bool allowEDecRelTol;
        bool mute;
        bool pardisoThreadAmt;
        bool needRefactorize;
        int globalIterNum;
        int topoIter;
        double relGL2Tol, energyParamSum;
        TriangleSoup result; // intermediate results of each iteration
        // constant precondition matrix for solving the linear system for search directions
        Eigen::SparseMatrix<double> precondMtr;
        Eigen::VectorXi I_mtr, J_mtr; // triplet representation
        Eigen::VectorXd V_mtr;
        // cholesky solver for solving the linear system for search directions
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> cholSolver;
        PardisoSolver<Eigen::VectorXi, Eigen::VectorXd> pardisoSolver;
        Eigen::VectorXd gradient; // energy gradient computed in each iteration
        Eigen::VectorXd searchDir; // search direction comptued in each iteration
        double lastEnergyVal; // for output and line search
        double lastEDec;
        double targetGRes;
        std::vector<Eigen::VectorXd> gradient_ET;
        std::vector<double> energyVal_ET;
        
        std::ofstream file_energyValPerIter;
        std::ofstream file_gradientPerIter;
        
    public: // constructor and destructor
        Optimizer(const TriangleSoup& p_data0, const std::vector<Energy*>& p_energyTerms, const std::vector<double>& p_energyParams,
                  bool p_withTopologyStep = true, bool p_mute = false);
        ~Optimizer(void);
        
    public: // API
        // precompute preconditioning matrix and factorize for fast solve, prepare initial guess
        void precompute(void);
        
        // solve the optimization problem that minimizes E using a hill-climbing method,
        // the final result will be in result
        bool solve(int maxIter = 100);
        
        void updatePrecondMtrAndFactorize(void);
        
        bool createFracture(double stressThres, bool initiation,
                            bool allowPropagate = true, bool allowInSplit = false);
        void setConfig(const TriangleSoup& config);
        
        void computeLastEnergyVal(void);
        
        void getGradientVisual(Eigen::MatrixXd& arrowVec) const;
        const TriangleSoup& getResult(void) const;
        int getIterNum(void) const;
        int getTopoIter(void) const;
        void setRelGL2Tol(double p_relTol);
        void setAllowEDecRelTol(bool p_allowEDecRelTol);
        
    protected: // helper functions
        // solve for new configuration in the next iteration
        //NOTE: must compute current gradient first
        bool solve_oneStep(void);
        
        bool lineSearch(void);

        void stepForward(TriangleSoup& data, double stepSize) const;
        
        void updateTargetGRes(void);
        
        void computeEnergyVal(const TriangleSoup& data, double& energyVal);
        void computeGradient(const TriangleSoup& data, Eigen::VectorXd& gradient);
        void computePrecondMtr(const TriangleSoup& data, Eigen::SparseMatrix<double>& precondMtr);
        void computeHessian(const TriangleSoup& data, Eigen::SparseMatrix<double>& hessian) const;
        
        void initStepSize(const TriangleSoup& data, double& stepSize) const;
        
    public: // data access
        double getLastEnergyVal(void) const;
    };
    
}

#endif /* Optimizer_hpp */
