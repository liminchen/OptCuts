//
//  Scaffold.hpp
//  OptCuts
//
//  Created by Minchen Li on 4/5/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#ifndef Scaffold_hpp
#define Scaffold_hpp

#include "TriMesh.hpp"

#include<Eigen/Eigen>

#include <cstdio>

namespace OptCuts {
    class Scaffold
    {
    public:
        TriMesh airMesh; // tessellation of voided regions
        Eigen::VectorXi bnd, localVI2Global; // map between airMesh indices to augmented system indices
        std::map<int, int> meshVI2AirMesh; // the inverse map of bnd
        int wholeMeshSize; // augmented system size
        
    public:
        Scaffold(void);
        Scaffold(const TriMesh& mesh, Eigen::MatrixXd UV_bnds = Eigen::MatrixXd(),
                Eigen::MatrixXi E = Eigen::MatrixXi(), const Eigen::VectorXi& p_bnd = Eigen::VectorXi());

        // augment mesh gradient with air mesh gradient with parameter w_scaf
        void augmentGradient(Eigen::VectorXd& gradient, const Eigen::VectorXd& gradient_scaf, double w_scaf) const;
        
        // augment mesh proxy matrix with air mesh proxy matrix with parameter w_scaf
        void augmentProxyMatrix(Eigen::VectorXi& I, Eigen::VectorXi& J, Eigen::VectorXd& V,
                                const Eigen::VectorXi& I_scaf, const Eigen::VectorXi& J_scaf, const Eigen::VectorXd& V_scaf,
                                double w_scaf) const;
        // when using dense representation:
        void augmentProxyMatrix(Eigen::MatrixXd& P,
                                const Eigen::MatrixXd& P_scaf,
                                double w_scaf) const;
        
        // extract air mesh searchDir from augmented searchDir
        void wholeSearchDir2airMesh(const Eigen::VectorXd& searchDir, Eigen::VectorXd& searchDir_airMesh) const;
        
        // stepForward air mesh using augmented searchDir
        void stepForward(const Eigen::MatrixXd& V0, const Eigen::VectorXd& searchDir, double stepSize);
        
        void mergeVNeighbor(const std::vector<std::set<int>>& vNeighbor_mesh, std::vector<std::set<int>>& vNeighbor) const;
        void mergeFixedV(const std::set<int>& fixedV_mesh, std::set<int>& fixedV) const;
        
        // for rendering purpose:
        void augmentUVwithAirMesh(Eigen::MatrixXd& UV, double scale) const;
        void augmentFwithAirMesh(Eigen::MatrixXi& F) const;
        void augmentFColorwithAirMesh(Eigen::MatrixXd& FColor) const;
        
        // get 1-ring airmesh loop for scaffolding optimization on local stencils
        void get1RingAirLoop(int vI,
                             Eigen::MatrixXd& UV, Eigen::MatrixXi& E, Eigen::VectorXi& bnd,
                             std::set<int>& loop_AMVI) const;
        
        bool getCornerAirLoop(const std::vector<int>& corner_mesh, const Eigen::RowVector2d& mergedPos,
                              Eigen::MatrixXd& UV, Eigen::MatrixXi& E, Eigen::VectorXi& bnd) const;
    };
}

#endif /* Scaffold_hpp */
