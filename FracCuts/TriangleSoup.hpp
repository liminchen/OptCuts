//
//  TriangleSoup.hpp
//  FracCuts
//
//  Created by Minchen Li on 8/30/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#ifndef TriangleSoup_hpp
#define TriangleSoup_hpp

#include <Eigen/Eigen>

namespace FracCuts{
    
    // duplicate the vertices and edges of a mesh to separate its triangles,
    // adjacent triangles in the original mesh will have a cohesive edge structure to
    // indicate the connectivity
    class TriangleSoup{
    public: // owned data
        Eigen::MatrixXd V; // duplicated vertex coordinates
        Eigen::MatrixXi F; // reordered triangle draw list (0, 1, 2, ...), indices based on V
        Eigen::MatrixXi E; // duplicated edges with 2 end vertex indices based on V
        Eigen::MatrixXi cohE; // cohesive edges with 2 end vertex indices for E based on V
        std::vector<Eigen::Matrix2d> restShape; // the rest shape of each element
        
    public: // constructor
        // default constructor that doesn't do anything
        TriangleSoup(void);
        
        // initialize from a triangle mesh, V will be constructed from UV_mesh in 2D,
        // V_mesh will be used to initialize restShape
        TriangleSoup(const Eigen::MatrixXd& V_mesh, const Eigen::MatrixXi& F_mesh,
                     const Eigen::MatrixXd& UV_mesh);
    };
    
}

#endif /* TriangleSoup_hpp */
