//
//  CHOLMODSolver.hpp
//  OptCuts
//
//  Created by Minchen Li on 6/22/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#ifndef CHOLMODSolver_hpp
#define CHOLMODSolver_hpp

#include "LinSysSolver.hpp"

#include "cholmod.h"

#include <Eigen/Eigen>

#include <vector>
#include <set>

namespace OptCuts {
    
    template <typename vectorTypeI, typename vectorTypeS>
    class CHOLMODSolver : public LinSysSolver<vectorTypeI, vectorTypeS>
    {
        typedef LinSysSolver<vectorTypeI, vectorTypeS> Base;
        
    protected:
        cholmod_common cm;
        cholmod_sparse *A;
        cholmod_factor *L;
        cholmod_dense *b;
        
    public:
        CHOLMODSolver(void);
        ~CHOLMODSolver(void);
        
        void set_type(int threadAmt, int _mtype, bool is_upper_half = false);
        
        void set_pattern(const std::vector<std::set<int>>& vNeighbor,
                         const std::set<int>& fixedVert);
        void set_pattern(const Eigen::SparseMatrix<double>& mtr); //NOTE: mtr must be SPD
        
        void update_a(const vectorTypeI &II,
                      const vectorTypeI &JJ,
                      const vectorTypeS &SS);
        void update_a(const Eigen::SparseMatrix<double>& mtr);
        
        void analyze_pattern(void);
        
        bool factorize(void);
        
        void solve(Eigen::VectorXd &rhs,
                   Eigen::VectorXd &result);
        
        virtual void multiply(const Eigen::VectorXd& x,
                              Eigen::VectorXd& Ax);
        
        void setZero(void);
        
        virtual void setCoeff(int rowI, int colI, double val);
        
        virtual void addCoeff(int rowI, int colI, double val);
    };
    
}

#endif /* CHOLMODSolver_hpp */
