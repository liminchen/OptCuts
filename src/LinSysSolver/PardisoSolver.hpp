//
//  PardisoSolver.h
//
//  Created by Olga Diamanti on 07/01/15.
//  Copyright (c) 2015 Olga Diamanti. All rights reserved.
//

#ifndef _PardisoSolver__
#define _PardisoSolver__

#include "LinSysSolver.hpp"

#include <vector>
#include <set>
#include <Eigen/Core>
#include <Eigen/Sparse>

namespace OptCuts {

    //extract II,JJ,SS (row,column and value vectors) from sparse matrix, Eigen version
    //Olga Diamanti's method for PARDISO
    void extract_ij_from_matrix(const Eigen::SparseMatrix<double> &A,
                                Eigen::VectorXi &II,
                                Eigen::VectorXi &JJ,
                                Eigen::VectorXd &SS);

    //extract II,JJ,SS (row,column and value vectors) from sparse matrix, std::vector version
    void extract_ij_from_matrix(const Eigen::SparseMatrix<double> &A,
                                std::vector<int> &II,
                                std::vector<int> &JJ,
                                std::vector<double> &SS);

    extern "C" {
        /* PARDISO prototype. */
        void pardisoinit (void   *, int    *,   int *, int *, double *, int *);
        void pardiso     (void   *, int    *,   int *, int *,    int *, int *,
                          double *, int    *,    int *, int *,   int *, int *,
                          int *, double *, double *, int *, double *);
        void pardiso_chkmatrix  (int *, int *, double *, int *, int *, int *);
        void pardiso_chkvec     (int *, int *, double *, int *);
        void pardiso_printstats (int *, int *, double *, int *, int *, int *,
                                 double *, int *);
    }

    template <typename vectorTypeI, typename vectorTypeS>
    class PardisoSolver : public LinSysSolver<vectorTypeI, vectorTypeS>
    {
        typedef LinSysSolver<vectorTypeI, vectorTypeS> Base;
        
    public:
        PardisoSolver() ;
        ~PardisoSolver();
        
        void set_type(int threadAmt, int _mtype, bool is_upper_half = false);
        
        void init(int threadAmt);
        
        void set_pattern(const vectorTypeI &II,
                         const vectorTypeI &JJ,
                         const vectorTypeS &SS);
        void set_pattern(const std::vector<std::set<int>>& vNeighbor,
                         const std::set<int>& fixedVert);
        void set_pattern(const Eigen::SparseMatrix<double>& mtr);
        
        void analyze_pattern();
        
        bool factorize();
        
        void solve(Eigen::VectorXd &rhs,
                   Eigen::VectorXd &result);
        
        void update_a(const vectorTypeS &SS);
        void update_a(const vectorTypeI &II,
                      const vectorTypeI &JJ,
                      const vectorTypeS &SS);
        
    protected:
        //vector that indicates which of the elements II,JJ input will be
        //kept and read into the matrix (for symmetric matrices, only those
        //elements II[i],JJ[i] for which II[i]<<JJ[i] will be kept)
        std::vector<int> lower_triangular_ind;
        
        std::vector<Eigen::VectorXi> iis;
        
        //pardiso stuff
        /*
         1: real and structurally symmetric, supernode pivoting
         2: real and symmetric positive definite
         -2: real and symmetric indefinite, diagonal or Bunch-Kaufman pivoting
         11: real and nonsymmetric, complete supernode pivoting
         */
        int mtype;       /* Matrix Type */
        
        // Remember if matrix is symmetric or not, to
        // decide whether to eliminate the non-upper-
        // diagonal entries from the input II,JJ,SS
        bool is_symmetric;
        bool is_upper_half;
        
        int nrhs = 1;     /* Number of right hand sides. */
        /* Internal solver memory pointer pt, */
        /* 32-bit: int pt[64]; 64-bit: long int pt[64] */
        /* or void *pt[64] should be OK on both architectures */
        void *pt[64];
        /* Pardiso control parameters. */
        int iparm[64];
        double   dparm[64];
        int maxfct, mnum, phase, error, msglvl, solver =0;
        /* Number of processors. */
        int      num_procs;
        /* Auxiliary variables. */
        char    *var;
        int i, k;
        double ddum;          /* Double dummy */
        int idum;         /* Integer dummy. */
    };
    
}


#endif /* defined(_PardisoSolver__) */
