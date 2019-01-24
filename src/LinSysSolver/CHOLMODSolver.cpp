//
//  CHOLMODSolver.cpp
//  OptCuts
//
//  Created by Minchen Li on 6/22/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#include "CHOLMODSolver.hpp"

#include <iostream>

namespace OptCuts {
    
    template <typename vectorTypeI, typename vectorTypeS>
    CHOLMODSolver<vectorTypeI, vectorTypeS>::CHOLMODSolver(void)
    {
        cholmod_start(&cm);
        A = NULL;
        L = NULL;
        b = NULL;
    }
    
    template <typename vectorTypeI, typename vectorTypeS>
    CHOLMODSolver<vectorTypeI, vectorTypeS>::~CHOLMODSolver(void)
    {
        cholmod_free_sparse(&A, &cm);
        cholmod_free_factor(&L, &cm);
        cholmod_free_dense(&b, &cm);
        cholmod_finish(&cm);
    }
    
    template <typename vectorTypeI, typename vectorTypeS>
    void CHOLMODSolver<vectorTypeI, vectorTypeS>::set_type(int threadAmt,
                                                           int _mtype,
                                                           bool is_upper_half)
    {
        //TODO: support more matrix types, currently only SPD
    }
    
    template <typename vectorTypeI, typename vectorTypeS>
    void CHOLMODSolver<vectorTypeI, vectorTypeS>::set_pattern(const std::vector<std::set<int>>& vNeighbor,
                                                              const std::set<int>& fixedVert)
    {
        Base::set_pattern(vNeighbor, fixedVert);
        
        //TODO: directly save into A
        if(!A) {
            A = cholmod_allocate_sparse(Base::numRows, Base::numRows, Base::ja.size(),
                                        true, true, -1, CHOLMOD_REAL, &cm);
            // -1: upper right part will be ignored during computation
        }
        else if(A->nrow != Base::numRows) {
            cholmod_free_sparse(&A, &cm);
            A = cholmod_allocate_sparse(Base::numRows, Base::numRows, Base::ja.size(),
                                        true, true, -1, CHOLMOD_REAL, &cm);
            // -1: upper right part will be ignored during computation
        }
        Base::ia.array() -= 1; Base::ja.array() -= 1; // CHOLMOD's index starts from 0
        memcpy(A->i, Base::ja.data(), Base::ja.size() * sizeof(Base::ja[0]));
        memcpy(A->p, Base::ia.data(), Base::ia.size() * sizeof(Base::ia[0]));
    }
    template <typename vectorTypeI, typename vectorTypeS>
    void  CHOLMODSolver<vectorTypeI, vectorTypeS>::set_pattern(const Eigen::SparseMatrix<double>& mtr)
    {
        //TODO: extract, manage Base list
        Base::numRows = static_cast<int>(mtr.rows());
        if(!A) {
            A = cholmod_allocate_sparse(Base::numRows, Base::numRows, mtr.nonZeros(),
                                        true, true, -1, CHOLMOD_REAL, &cm);
            // -1: upper right part will be ignored during computation
        }
        else if(A->nrow != Base::numRows) {
            cholmod_free_sparse(&A, &cm);
            A = cholmod_allocate_sparse(Base::numRows, Base::numRows, mtr.nonZeros(),
                                        true, true, -1, CHOLMOD_REAL, &cm);
            // -1: upper right part will be ignored during computation
        }
        memcpy(A->i, mtr.innerIndexPtr(), mtr.nonZeros() * sizeof(mtr.innerIndexPtr()[0]));
        memcpy(A->p, mtr.outerIndexPtr(), (Base::numRows + 1) * sizeof(mtr.outerIndexPtr()[0]));
        memcpy(A->x, mtr.valuePtr(), mtr.nonZeros() * sizeof(mtr.valuePtr()[0]));
    }
    
    template <typename vectorTypeI, typename vectorTypeS>
    void CHOLMODSolver<vectorTypeI, vectorTypeS>::update_a(const vectorTypeI &II,
                                                           const vectorTypeI &JJ,
                                                           const vectorTypeS &SS)
    {
        Base::update_a(II, JJ, SS);
        
        //TODO: directly save into A
        memcpy(A->x, Base::a.data(), Base::a.size() * sizeof(Base::a[0]));
    }
    template <typename vectorTypeI, typename vectorTypeS>
    void CHOLMODSolver<vectorTypeI, vectorTypeS>::update_a(const Eigen::SparseMatrix<double>& mtr)
    {
        //TODO: extract, manage Base list
        assert(Base::numRows == static_cast<int>(mtr.rows()));
        assert(A);
        assert(A->x);
        memcpy(A->x, mtr.valuePtr(), mtr.nonZeros() * sizeof(mtr.valuePtr()[0]));
    }
    
    template <typename vectorTypeI, typename vectorTypeS>
    void CHOLMODSolver<vectorTypeI, vectorTypeS>::analyze_pattern(void)
    {
        cholmod_free_factor(&L, &cm);
        L = cholmod_analyze(A, &cm);
    }
    
    template <typename vectorTypeI, typename vectorTypeS>
    bool CHOLMODSolver<vectorTypeI, vectorTypeS>::factorize(void)
    {
        return !cholmod_factorize(A, L, &cm);
    }
    
    template <typename vectorTypeI, typename vectorTypeS>
    void CHOLMODSolver<vectorTypeI, vectorTypeS>::solve(Eigen::VectorXd &rhs,
                                                        Eigen::VectorXd &result)
    {
        //TODO: directly point to rhs?
        if(!b) {
            b = cholmod_allocate_dense(Base::numRows, 1, Base::numRows, CHOLMOD_REAL, &cm);
        }
        else if(b->nrow != Base::numRows) {
            cholmod_free_dense(&b, &cm);
            b = cholmod_allocate_dense(Base::numRows, 1, Base::numRows, CHOLMOD_REAL, &cm);
        }
        memcpy(b->x, rhs.data(), rhs.size() * sizeof(rhs[0]));
        cholmod_dense *x;
        x = cholmod_solve(CHOLMOD_A, L, b, &cm);
        result.resize(rhs.size());
        memcpy(result.data(), x->x, result.size() * sizeof(result[0]));
        cholmod_free_dense(&x, &cm);
    }
    
    template <typename vectorTypeI, typename vectorTypeS>
    void CHOLMODSolver<vectorTypeI, vectorTypeS>::multiply(const Eigen::VectorXd& x,
                                                           Eigen::VectorXd& Ax)
    {
        assert(x.size() == Base::numRows);
        
        //TODO: manually initialize cholmod_dense without reallocating memory
        cholmod_dense *x_cd = cholmod_allocate_dense(Base::numRows, 1, Base::numRows,
                                                     CHOLMOD_REAL, &cm);
        memcpy(x_cd->x, x.data(), Base::numRows * sizeof(x[0]));
        cholmod_dense *y_cd = cholmod_allocate_dense(Base::numRows, 1, Base::numRows,
                                                     CHOLMOD_REAL, &cm);
        double alpha[2] = {1.0, 1.0}, beta[2] = {0.0, 0.0};
        
        cholmod_sdmult(A, 0, alpha, beta, x_cd, y_cd, &cm);
        
        Ax.conservativeResize(Base::numRows);
        memcpy(Ax.data(), y_cd->x, Base::numRows * sizeof(Ax[0]));
        
        cholmod_free_dense(&x_cd, &cm);
        cholmod_free_dense(&y_cd, &cm);
    }
    
    template <typename vectorTypeI, typename vectorTypeS>
    void CHOLMODSolver<vectorTypeI, vectorTypeS>::setZero(void)
    {
        //TODO: directly manipulate valuePtr without a
        Base::setZero();
        memcpy(A->x, Base::a.data(), Base::a.size() * sizeof(Base::a[0]));
    }
    
    template <typename vectorTypeI, typename vectorTypeS>
    void CHOLMODSolver<vectorTypeI, vectorTypeS>::setCoeff(int rowI, int colI, double val)
    {
        //TODO: directly manipulate valuePtr without a
        //TODO: faster O(1) indices!!
        
        if(rowI <= colI) {
            assert(rowI < Base::IJ2aI.size());
            const auto finder = Base::IJ2aI[rowI].find(colI);
            assert(finder != Base::IJ2aI[rowI].end());
            Base::a[finder->second] = val;
            ((double*)A->x)[finder->second] = val;
        }
    }
    
    template <typename vectorTypeI, typename vectorTypeS>
    void CHOLMODSolver<vectorTypeI, vectorTypeS>::addCoeff(int rowI, int colI, double val)
    {
        //TODO: directly manipulate valuePtr without a
        //TODO: faster O(1) indices!!
        
        if(rowI <= colI) {
            assert(rowI < Base::IJ2aI.size());
            const auto finder = Base::IJ2aI[rowI].find(colI);
            assert(finder != Base::IJ2aI[rowI].end());
            Base::a[finder->second] += val;
            ((double*)A->x)[finder->second] += val;
        }
    }
    
    template class CHOLMODSolver<Eigen::VectorXi, Eigen::VectorXd>;
    
}


