//
//  EigenLibSolver.cpp
//  OptCuts
//
//  Created by Minchen Li on 6/30/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#include "EigenLibSolver.hpp"

namespace OptCuts {
    
    template <typename vectorTypeI, typename vectorTypeS>
    void EigenLibSolver<vectorTypeI, vectorTypeS>::set_type(int threadAmt, int _mtype, bool is_upper_half)
    {
        //TODO: support more matrix types, currently only SPD
        useDense = false;
        //TODO: move to base class and support for CHOLMOD
    }
    
    template <typename vectorTypeI, typename vectorTypeS>
    void EigenLibSolver<vectorTypeI, vectorTypeS>::set_pattern(const std::vector<std::set<int>>& vNeighbor,
                                                               const std::set<int>& fixedVert)
    {
        if(useDense) {
            Base::numRows = static_cast<int>(vNeighbor.size()) * 2;
            coefMtr_dense.resize(Base::numRows, Base::numRows);
        }
        else {
            Base::set_pattern(vNeighbor, fixedVert);
            
            //TODO: directly save into mtr
            coefMtr.resize(Base::numRows, Base::numRows);
            coefMtr.reserve(Base::ja.size());
            Base::ia.array() -= 1.0;
            Base::ja.array() -= 1.0;
            memcpy(coefMtr.innerIndexPtr(), Base::ja.data(), Base::ja.size() * sizeof(Base::ja[0]));
            memcpy(coefMtr.outerIndexPtr(), Base::ia.data(), Base::ia.size() * sizeof(Base::ia[0]));
        }
    }
    template <typename vectorTypeI, typename vectorTypeS>
    void EigenLibSolver<vectorTypeI, vectorTypeS>::set_pattern(const Eigen::SparseMatrix<double>& mtr) //NOTE: mtr must be SPD
    {
        if(useDense) {
            coefMtr_dense = Eigen::MatrixXd(mtr);
        }
        else {
            coefMtr = mtr;
        }
    }
    
    template <typename vectorTypeI, typename vectorTypeS>
    void EigenLibSolver<vectorTypeI, vectorTypeS>::update_a(const vectorTypeI &II,
                                                            const vectorTypeI &JJ,
                                                            const vectorTypeS &SS)
    {
        if(useDense) {
            coefMtr_dense.setZero();
            for(int i = 0; i < II.size(); i++) {
                coefMtr_dense(II[i], JJ[i]) += SS[i];
            }
        }
        else {
            Base::update_a(II, JJ, SS);
            
            //TODO: directly save into coefMtr
            memcpy(coefMtr.valuePtr(), Base::a.data(), Base::a.size() * sizeof(Base::a[0]));
        }
    }
    
    template <typename vectorTypeI, typename vectorTypeS>
    void EigenLibSolver<vectorTypeI, vectorTypeS>::analyze_pattern(void)
    {
        if(!useDense) {
            simplicialLDLT.analyzePattern(coefMtr);
            assert(simplicialLDLT.info() == Eigen::Success);
        }
    }
    
    template <typename vectorTypeI, typename vectorTypeS>
    bool EigenLibSolver<vectorTypeI, vectorTypeS>::factorize(void)
    {
        bool succeeded = false;
        if(useDense) {
            LDLT.compute(coefMtr_dense);
            succeeded = (LDLT.info() == Eigen::Success);
        }
        else {
            simplicialLDLT.factorize(coefMtr);
            succeeded = (simplicialLDLT.info() == Eigen::Success);
        }
        assert(succeeded);
        return succeeded;
    }
    
    template <typename vectorTypeI, typename vectorTypeS>
    void EigenLibSolver<vectorTypeI, vectorTypeS>::solve(Eigen::VectorXd &rhs,
                                                         Eigen::VectorXd &result)
    {
        if(useDense) {
            result = LDLT.solve(rhs);
            assert(LDLT.info() == Eigen::Success);
        }
        else {
            result = simplicialLDLT.solve(rhs);
            assert(simplicialLDLT.info() == Eigen::Success);
        }
    }
    
    template <typename vectorTypeI, typename vectorTypeS>
    double EigenLibSolver<vectorTypeI, vectorTypeS>::coeffMtr(int rowI, int colI) const
    {
        if(useDense) {
            return coefMtr_dense(rowI, colI);
        }
        else {
            return Base::coeffMtr(rowI, colI);
        }
    }
    
    template <typename vectorTypeI, typename vectorTypeS>
    void EigenLibSolver<vectorTypeI, vectorTypeS>::setZero(void)
    {
        //TODO: useDense
        //TODO: directly manipulate valuePtr without a
        Base::setZero();
        memcpy(coefMtr.valuePtr(), Base::a.data(), Base::a.size() * sizeof(Base::a[0]));
    }
    
    template <typename vectorTypeI, typename vectorTypeS>
    void EigenLibSolver<vectorTypeI, vectorTypeS>::setCoeff(int rowI, int colI, double val)
    {
        //TODO: useDense
        //TODO: directly manipulate valuePtr without a
        //TODO: faster O(1) indices!!
        
        if(rowI <= colI) {
            assert(rowI < Base::IJ2aI.size());
            const auto finder = Base::IJ2aI[rowI].find(colI);
            assert(finder != Base::IJ2aI[rowI].end());
            Base::a[finder->second] = val;
            coefMtr.valuePtr()[finder->second] = val;
        }
    }
    
    template <typename vectorTypeI, typename vectorTypeS>
    void EigenLibSolver<vectorTypeI, vectorTypeS>::addCoeff(int rowI, int colI, double val)
    {
        //TODO: useDense
        //TODO: directly manipulate valuePtr without a
        //TODO: faster O(1) indices!!
        
        if(rowI <= colI) {
            assert(rowI < Base::IJ2aI.size());
            const auto finder = Base::IJ2aI[rowI].find(colI);
            assert(finder != Base::IJ2aI[rowI].end());
            Base::a[finder->second] += val;
            coefMtr.valuePtr()[finder->second] += val;
        }
    }
    
    template class EigenLibSolver<Eigen::VectorXi, Eigen::VectorXd>;
    
}
