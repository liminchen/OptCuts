//
//  TriangleSoup.cpp
//  FracCuts
//
//  Created by Minchen Li on 8/30/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#include "TriangleSoup.hpp"

namespace FracCuts {
    
    TriangleSoup::TriangleSoup(void)
    {
        
    }
    
    TriangleSoup::TriangleSoup(const Eigen::MatrixXd& V_mesh, const Eigen::MatrixXi& F_mesh,
                               const Eigen::MatrixXd& UV_mesh)
    {
        // duplicate vertices and edges, use new face vertex indices,
        // construct fast-matching data structure for cohesive edge construction,
        // compute triangle matrix to save rest shapes
        V.resize(F_mesh.rows() * F_mesh.cols(), 3);
        F.resize(F_mesh.rows(), F_mesh.cols());
        E.resize(F_mesh.rows() * F_mesh.cols(), 2);
        std::map<std::pair<int, int>, int> edge2DupInd;
        restShape.resize(F_mesh.rows());
        for(int triI = 0; triI < F_mesh.rows(); triI++) {
            int vDupIndStart = triI * 3;
            
            V.row(vDupIndStart) = UV_mesh.row(F_mesh.row(triI)[0]);
            V.row(vDupIndStart + 1) = UV_mesh.row(F_mesh.row(triI)[1]);
            V.row(vDupIndStart + 2) = UV_mesh.row(F_mesh.row(triI)[2]);
            
            F(triI, 0) = vDupIndStart;
            F(triI, 1) = vDupIndStart + 1;
            F(triI, 2) = vDupIndStart + 2;
            
            E(vDupIndStart, 0) = F_mesh.row(triI)[0];
            E(vDupIndStart, 1) = F_mesh.row(triI)[1];
            E(vDupIndStart + 1, 0) = F_mesh.row(triI)[1];
            E(vDupIndStart + 1, 1) = F_mesh.row(triI)[2];
            E(vDupIndStart + 2, 0) = F_mesh.row(triI)[2];
            E(vDupIndStart + 2, 1) = F_mesh.row(triI)[0];
            
            edge2DupInd[std::pair<int, int>(F.row(triI)[0], F.row(triI)[1])] = vDupIndStart;
            edge2DupInd[std::pair<int, int>(F.row(triI)[1], F.row(triI)[2])] = vDupIndStart + 1;
            edge2DupInd[std::pair<int, int>(F.row(triI)[2], F.row(triI)[0])] = vDupIndStart + 2;
            
            const Eigen::Vector3d& v0 = V_mesh.row(F_mesh.row(triI)[0]);
            const Eigen::Vector3d& v1 = V_mesh.row(F_mesh.row(triI)[1]);
            const Eigen::Vector3d& v2 = V_mesh.row(F_mesh.row(triI)[2]);
            const Eigen::Vector3d e1 = v1 - v0, e2 = v2 - v0;
            const double e1Len = e1.norm(), e2Len = e2.norm();
            const double e2_projTo_e1 = e1.dot(e2) / e1Len;
            const double cosine_e1_e2 = e2_projTo_e1 / e2Len;
            restShape[triI] << e1Len, e2Len * sqrt(1 - cosine_e1_e2 * cosine_e1_e2),
            0.0, e2_projTo_e1;
        }
        
        // construct cohesive edge data structure
        cohE.resize(F_mesh.rows() * F_mesh.cols(), 2);
        for(int edgeI = 0; edgeI < E.rows(); edgeI++) {
            auto dupIndFinder = edge2DupInd.find(std::pair<int, int>(E(edgeI, 1), E(edgeI, 0)));
            if(dupIndFinder != edge2DupInd.end()) {
                cohE(edgeI, 0) = E(dupIndFinder->second, 1);
                cohE(edgeI, 1) = E(dupIndFinder->second, 0);
            }
            else {
                // boundary edge
                cohE(edgeI, 0) = cohE(edgeI, 1) = -1;
            }
        }
    }
    
}
