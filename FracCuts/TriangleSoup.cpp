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
//        // duplicate vertices and edges, use new face vertex indices,
//        // construct fast-matching data structure for cohesive edge construction,
//        // compute triangle matrix to save rest shapes
//        V_rest.resize(F_mesh.rows() * F_mesh.cols(), 3);
//        V.resize(F_mesh.rows() * F_mesh.cols(), 2);
//        F.resize(F_mesh.rows(), F_mesh.cols());
//        E.resize(F_mesh.rows() * F_mesh.cols(), 2);
//        std::map<std::pair<int, int>, int> edge2DupInd;
//        for(int triI = 0; triI < F_mesh.rows(); triI++) {
//            int vDupIndStart = triI * 3;
//            
//            V.row(vDupIndStart) = UV_mesh.row(F_mesh.row(triI)[0]);
//            V.row(vDupIndStart + 1) = UV_mesh.row(F_mesh.row(triI)[1]);
//            V.row(vDupIndStart + 2) = UV_mesh.row(F_mesh.row(triI)[2]);
//            V_rest.row(vDupIndStart) = V_mesh.row(F_mesh.row(triI)[0]);
//            V_rest.row(vDupIndStart + 1) = V_mesh.row(F_mesh.row(triI)[1]);
//            V_rest.row(vDupIndStart + 2) = V_mesh.row(F_mesh.row(triI)[2]);
//            
//            F(triI, 0) = vDupIndStart;
//            F(triI, 1) = vDupIndStart + 1;
//            F(triI, 2) = vDupIndStart + 2;
//            
//            E(vDupIndStart, 0) = F_mesh.row(triI)[0];
//            E(vDupIndStart, 1) = F_mesh.row(triI)[1];
//            E(vDupIndStart + 1, 0) = F_mesh.row(triI)[1];
//            E(vDupIndStart + 1, 1) = F_mesh.row(triI)[2];
//            E(vDupIndStart + 2, 0) = F_mesh.row(triI)[2];
//            E(vDupIndStart + 2, 1) = F_mesh.row(triI)[0];
//            
//            edge2DupInd[std::pair<int, int>(F.row(triI)[0], F.row(triI)[1])] = vDupIndStart;
//            edge2DupInd[std::pair<int, int>(F.row(triI)[1], F.row(triI)[2])] = vDupIndStart + 1;
//            edge2DupInd[std::pair<int, int>(F.row(triI)[2], F.row(triI)[0])] = vDupIndStart + 2;
//        }
//        
//        // construct cohesive edge data structure
//        cohE.resize(F_mesh.rows() * F_mesh.cols(), 2);
//        for(int edgeI = 0; edgeI < E.rows(); edgeI++) {
//            auto dupIndFinder = edge2DupInd.find(std::pair<int, int>(E(edgeI, 1), E(edgeI, 0)));
//            if(dupIndFinder != edge2DupInd.end()) {
//                cohE(edgeI, 0) = E(dupIndFinder->second, 1);
//                cohE(edgeI, 1) = E(dupIndFinder->second, 0);
//            }
//            else {
//                // boundary edge
//                cohE(edgeI, 0) = cohE(edgeI, 1) = -1;
//            }
//        }
        //!! for now just deal with regular mesh to implement [Smith and Schaefer 2015]
        V_rest = V_mesh;
        V = UV_mesh;
        F = F_mesh;
    }
    
    TriangleSoup::TriangleSoup(Primitive primitive, double size, double spacing)
    {
        switch(primitive)
        {
            case P_SQUARE: {
                assert(size >= spacing);
                int gridSize = static_cast<int>(size / spacing) + 1;
                spacing = size / (gridSize - 1);
                V_rest.resize(gridSize * gridSize, 3);
                V.resize(gridSize * gridSize, 2);
                for(int rowI = 0; rowI < gridSize; rowI++)
                {
                    for(int colI = 0; colI < gridSize; colI++)
                    {
                        V_rest.row(rowI * gridSize + colI) = Eigen::Vector3d(spacing * colI, spacing * rowI, 0.0);
                        V.row(rowI * gridSize + colI) = spacing * Eigen::Vector2d(colI, rowI);
                    }
                }
                
                F.resize((gridSize - 1) * (gridSize - 1) * 2, 3);
                for(int rowI = 0; rowI < gridSize - 1; rowI++)
                {
                    for(int colI = 0; colI < gridSize - 1; colI++)
                    {
                        int squareI = rowI * (gridSize - 1) + colI;
                        F.row(squareI * 2) = Eigen::Vector3i(
                            rowI * gridSize + colI, (rowI + 1) * gridSize + colI + 1, (rowI + 1) * gridSize + colI);
                        F.row(squareI * 2 + 1) = Eigen::Vector3i(
                            rowI * gridSize + colI, rowI * gridSize + colI + 1, (rowI + 1) * gridSize + colI + 1);
                    }
                }
                break;
            }
                
            default:
                assert(0 && "no such primitive to construct!");
                break;
        }
    }
    
}
