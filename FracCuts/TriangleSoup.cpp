//
//  TriangleSoup.cpp
//  FracCuts
//
//  Created by Minchen Li on 8/30/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#include "TriangleSoup.hpp"
#include "IglUtils.hpp"

#include <igl/cotmatrix.h>
#include <igl/avg_edge_length.h>
#include <igl/writeOBJ.h>

#include <fstream>

extern std::ofstream logFile;

namespace FracCuts {
    
    TriangleSoup::TriangleSoup(void)
    {
        
    }
    
    TriangleSoup::TriangleSoup(const Eigen::MatrixXd& V_mesh, const Eigen::MatrixXi& F_mesh,
                               const Eigen::MatrixXd& UV_mesh, bool separateTri)
    {
        if(separateTri)
        {
            // duplicate vertices and edges, use new face vertex indices,
            // construct cohesive edge pairs,
            // compute triangle matrix to save rest shapes
            V_rest.resize(F_mesh.rows() * F_mesh.cols(), 3);
            V.resize(F_mesh.rows() * F_mesh.cols(), 2);
            F.resize(F_mesh.rows(), F_mesh.cols());
            std::map<std::pair<int, int>, Eigen::Vector3i> edge2DupInd;
            int cohEAmt = 0;
            for(int triI = 0; triI < F_mesh.rows(); triI++)
            {
                int vDupIndStart = triI * 3;
                
                if(UV_mesh.rows() == V_mesh.rows()) {
                    // bijective map without seams, usually Tutte
                    V.row(vDupIndStart) = UV_mesh.row(F_mesh.row(triI)[0]);
                    V.row(vDupIndStart + 1) = UV_mesh.row(F_mesh.row(triI)[1]);
                    V.row(vDupIndStart + 2) = UV_mesh.row(F_mesh.row(triI)[2]);
                }
                
//                // perturb for testing separation energy
//                V.row(vDupIndStart + 1) = V.row(vDupIndStart) + 0.9 * (V.row(vDupIndStart + 1) - V.row(vDupIndStart));
//                V.row(vDupIndStart + 2) = V.row(vDupIndStart) + 0.9 * (V.row(vDupIndStart + 2) - V.row(vDupIndStart));
                
                V_rest.row(vDupIndStart) = V_mesh.row(F_mesh.row(triI)[0]);
                V_rest.row(vDupIndStart + 1) = V_mesh.row(F_mesh.row(triI)[1]);
                V_rest.row(vDupIndStart + 2) = V_mesh.row(F_mesh.row(triI)[2]);
                
                F(triI, 0) = vDupIndStart;
                F(triI, 1) = vDupIndStart + 1;
                F(triI, 2) = vDupIndStart + 2;
                
                for(int vI = 0; vI < 3; vI++)
                {
                    int vsI = F_mesh.row(triI)[vI], veI = F_mesh.row(triI)[(vI + 1) % 3];
                    auto cohEFinder = edge2DupInd.find(std::pair<int, int>(veI, vsI));
                    if(cohEFinder == edge2DupInd.end()) {
                        cohEAmt++;
                        edge2DupInd[std::pair<int, int>(vsI, veI)] = Eigen::Vector3i(cohEAmt, F(triI, vI), F(triI, (vI + 1) % 3));
                    }
                    else {
                        edge2DupInd[std::pair<int, int>(vsI, veI)] = Eigen::Vector3i(-cohEFinder->second[0], F(triI, vI), F(triI, (vI + 1) % 3));
                    }
                }
            }
            
            cohE.resize(cohEAmt, 4);
            cohE.setConstant(-1);
            for(const auto& cohPI : edge2DupInd) {
                if(cohPI.second[0] > 0) {
                    cohE.row(cohPI.second[0] - 1)[0] = cohPI.second[1];
                    cohE.row(cohPI.second[0] - 1)[1] = cohPI.second[2];
                }
                else {
                    cohE.row(-cohPI.second[0] - 1)[2] = cohPI.second[2];
                    cohE.row(-cohPI.second[0] - 1)[3] = cohPI.second[1];
                }
            }
//            std::cout << cohE << std::endl;
            
            if(UV_mesh.rows() == 0) {
                // no input UV
                initRigidUV();
            }
            else if(UV_mesh.rows() != V_mesh.rows()) {
                // input UV with seams
                assert(0 && "TODO: separate each triangle in UV space according to FUV!");
                //TODO: for verifying cohesive energy implementation, don't define cohesive edges pairs on seams?
            }
        }
        else {
            // deal with regular mesh
            assert((UV_mesh.rows() == V_mesh.rows()) &&
                   "TODO: load FUV, also support FUV in the energies!");
            V_rest = V_mesh;
            V = UV_mesh;
            F = F_mesh;
        }
        
        computeFeatures();
    }
    
    void initCylinder(double r1_x, double r1_y, double r2_x, double r2_y, double height, int circle_res, int height_resolution,
        Eigen::MatrixXd & V,
        Eigen::MatrixXi & F,
        Eigen::MatrixXd * uv_coords_per_face = NULL,
        Eigen::MatrixXi * uv_coords_face_ids = NULL)
    {
        int nvertices = circle_res * (height_resolution+1);
        int nfaces = 2*circle_res * height_resolution;
        
        V.resize(nvertices, 3);
        if(uv_coords_per_face) {
            uv_coords_per_face->resize(nvertices, 2);
        }
        F.resize(nfaces, 3);
        for (int j=0; j<height_resolution+1; j++) {
            for (int i=0; i<circle_res; i++)
            {
                double t = (double)j / (double)height_resolution;
                double h = height * t;
                double theta = i * 2*M_PI / circle_res;
                double r_x = r1_x * t + r2_x * (1-t);
                double r_y = r1_y * t + r2_y * (1-t);
                V.row(j*circle_res+i) = Eigen::Vector3d(r_x*cos(theta), height-h, r_y*sin(theta));
                if(uv_coords_per_face) {
                    uv_coords_per_face->row(j*circle_res+i) = Eigen::Vector2d(r_x*cos(theta), r_y*sin(theta));
                }
                
                if (j<height_resolution)
                {
                    int vl0 = j*circle_res+i;
                    int vl1 = j*circle_res+(i+1)%circle_res;
                    int vu0 = (j+1)*circle_res+i;
                    int vu1 = (j+1)*circle_res+(i+1)%circle_res;
                    F.row(2*(j*circle_res+i)+0) = Eigen::Vector3i(vl0, vl1, vu1);
                    F.row(2*(j*circle_res+i)+1) = Eigen::Vector3i(vu0, vl0, vu1);
                }
            }
        }
    }
    
    TriangleSoup::TriangleSoup(Primitive primitive, double size, double spacing, bool separateTri)
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
                
            case P_CYLINDER: {
                initCylinder(0.5, 0.5, 1.0, 1.0, 1.0, 20, 20, V_rest, F, &V);
                break;
            }
                
            default:
                assert(0 && "no such primitive to construct!");
                break;
        }
        
        if(separateTri) {
            *this = TriangleSoup(V_rest, F, V);
        }
        else {
            computeFeatures();
        }
    }
    
    void TriangleSoup::computeFeatures(void)
    {
        fixedVert.insert(0);
        
        boundaryEdge.resize(cohE.rows());
        edgeLen.resize(cohE.rows());
        for(int cohI = 0; cohI < cohE.rows(); cohI++)
        {
            if(cohE.row(cohI).minCoeff() >= 0) {
                boundaryEdge[cohI] = 0;
            }
            else {
                boundaryEdge[cohI] = 1;
            }
            edgeLen[cohI] = (V_rest.row(cohE(cohI, 0)) - V_rest.row(cohE(cohI, 1))).norm();
        }
        
        //        Eigen::SparseMatrix<double> M;
        //        massmatrix(data.V_rest, data.F, igl::MASSMATRIX_TYPE_DEFAULT, M);
        Eigen::SparseMatrix<double> L;
        igl::cotmatrix(V_rest, F, L);
        LaplacianMtr.resize(V.rows() * 2, V.rows() * 2);
        for (int k = 0; k < L.outerSize(); ++k)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it)
            {
                if((fixedVert.find(static_cast<int>(it.row())) == fixedVert.end()) &&
                   (fixedVert.find(static_cast<int>(it.col())) == fixedVert.end()))
                {
                    LaplacianMtr.insert(it.row() * 2, it.col() * 2) = -it.value();// * M.coeffRef(it.row(), it.row());
                    LaplacianMtr.insert(it.row() * 2 + 1, it.col() * 2 + 1) = -it.value();// * M.coeffRef(it.row(), it.row());
                }
            }
        }
        for(const auto fixedVI : fixedVert) {
            LaplacianMtr.insert(2 * fixedVI, 2 * fixedVI) = 1.0;
            LaplacianMtr.insert(2 * fixedVI + 1, 2 * fixedVI + 1) = 1.0;
        }
        LaplacianMtr.makeCompressed();
        
//        igl::cotmatrix_entries(V_rest, F, cotVals);
        
        triArea.resize(F.rows());
        triAreaSq.resize(F.rows());
        e0SqLen.resize(F.rows());
        e1SqLen.resize(F.rows());
        e0dote1.resize(F.rows());
        for(int triI = 0; triI < F.rows(); triI++) {
            const Eigen::Vector3i& triVInd = F.row(triI);
            
            const Eigen::Vector3d& P1 = V_rest.row(triVInd[0]);
            const Eigen::Vector3d& P2 = V_rest.row(triVInd[1]);
            const Eigen::Vector3d& P3 = V_rest.row(triVInd[2]);
            
            const Eigen::Vector3d P2m1 = P2 - P1;
            const Eigen::Vector3d P3m1 = P3 - P1;
            
            triArea[triI] = 0.5 * P2m1.cross(P3m1).norm();
            triAreaSq[triI] = triArea[triI] * triArea[triI];
            e0SqLen[triI] = P2m1.squaredNorm();
            e1SqLen[triI] = P3m1.squaredNorm();
            e0dote1[triI] = P2m1.dot(P3m1);
        }
        avgEdgeLen = igl::avg_edge_length(V_rest, F);
        
//        //!! for edge count minimization of separation energy
//        for(int cohI = 0; cohI < cohE.rows(); cohI++)
//        {
//            edgeLen[cohI] = avgEdgeLen;
//        }
        
        bbox.block(0, 0, 1, 3) = V_rest.row(0);
        bbox.block(1, 0, 1, 3) = V_rest.row(0);
        for(int vI = 1; vI < V_rest.rows(); vI++) {
            const Eigen::RowVector3d& v = V_rest.row(vI);
            for(int dimI = 0; dimI < 3; dimI++) {
                if(v[dimI] < bbox(0, dimI)) {
                    bbox(0, dimI) = v[dimI];
                }
                if(v[dimI] > bbox(1, dimI)) {
                    bbox(1, dimI) = v[dimI];
                }
            }
        }
    }
    
    void TriangleSoup::computeSeamScore(Eigen::VectorXd& seamScore) const
    {
        seamScore.resize(cohE.rows());
        for(int cohI = 0; cohI < cohE.rows(); cohI++)
        {
            if(boundaryEdge[cohI]) {
                seamScore[cohI] = -1.0;
            }
            else {
                seamScore[cohI] = std::max((V.row(cohE(cohI, 0)) - V.row(cohE(cohI, 2))).norm(),
                    (V.row(cohE(cohI, 1)) - V.row(cohE(cohI, 3))).norm()) / avgEdgeLen;
            }
        }
    }
    
    void TriangleSoup::initRigidUV(void)
    {
        V.resize(V_rest.rows(), 2);
        for(int triI = 0; triI < F.rows(); triI++)
        {
            const Eigen::Vector3i& triVInd = F.row(triI);
            
            const Eigen::Vector3d x_3D[3] = {
                V_rest.row(triVInd[0]),
                V_rest.row(triVInd[1]),
                V_rest.row(triVInd[2])
            };
            Eigen::Vector2d x[3];
            IglUtils::mapTriangleTo2D(x_3D, x);
            
            V.row(triVInd[0]) = x[0];
            V.row(triVInd[1]) = x[1];
            V.row(triVInd[2]) = x[2];
        }
    }
    
    void TriangleSoup::save(const std::string& filePath, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                            const Eigen::MatrixXd UV, const Eigen::MatrixXi& FUV) const
    {
        std::ofstream out;
        out.open(filePath);
        assert(out.is_open());
        
        for(int vI = 0; vI < V.rows(); vI++) {
            const Eigen::RowVector3d& v = V.row(vI);
            out << "v " << v[0] << " " << v[1] << " " << v[2] << std::endl;
        }
        
        for(int vI = 0; vI < UV.rows(); vI++) {
            const Eigen::RowVector2d& uv = UV.row(vI);
            out << "vt " << uv[0] << " " << uv[1] << std::endl;
        }
        
        if(FUV.rows() == F.rows()) {
            for(int triI = 0; triI < F.rows(); triI++) {
                const Eigen::RowVector3i& tri = F.row(triI);
                const Eigen::RowVector3i& tri_UV = FUV.row(triI);
                out << "f " << tri[0] + 1 << "/" << tri_UV[0] + 1 <<
                " " << tri[1] + 1 << "/" << tri_UV[1] + 1 <<
                " " << tri[2] + 1 << "/" << tri_UV[2] + 1 << std::endl;
            }
        }
        else {
            for(int triI = 0; triI < F.rows(); triI++) {
                const Eigen::RowVector3i& tri = F.row(triI);
                out << "f " << tri[0] + 1 << "/" << tri[0] + 1 <<
                " " << tri[1] + 1 << "/" << tri[1] + 1 <<
                " " << tri[2] + 1 << "/" << tri[2] + 1 << std::endl;
            }
        }
        
        out.close();
    }
    
    void TriangleSoup::save(const std::string& filePath) const
    {
        save(filePath, V_rest, F, V);
    }
    
    void TriangleSoup::saveAsMesh(const std::string& filePath, bool scaleUV) const
    {
        const double thres = 1.0e-2;
        std::vector<int> dupVI2GroupI(V.rows());
        std::vector<std::set<int>> meshVGroup(V.rows());
        std::vector<int> dupVI2GroupI_3D(V_rest.rows());
        std::vector<std::set<int>> meshVGroup_3D(V_rest.rows());
        for(int dupI = 0; dupI < V.rows(); dupI++) {
            dupVI2GroupI[dupI] = dupI;
            meshVGroup[dupI].insert(dupI);
            dupVI2GroupI_3D[dupI] = dupI;
            meshVGroup_3D[dupI].insert(dupI);
        }
        for(int cohI = 0; cohI < cohE.rows(); cohI++) {
            if(boundaryEdge[cohI]) {
                continue;
            }
            
            for(int pI = 0; pI < 2; pI++) {
                int groupI0_3D = dupVI2GroupI_3D[cohE(cohI, 0 + pI)];
                int groupI2_3D = dupVI2GroupI_3D[cohE(cohI, 2 + pI)];
                if(groupI0_3D != groupI2_3D) {
                    for(const auto& vI : meshVGroup_3D[groupI2_3D]) {
                        dupVI2GroupI_3D[vI] = groupI0_3D;
                        meshVGroup_3D[groupI0_3D].insert(vI);
                    }
                    meshVGroup_3D[groupI2_3D].clear();
                }
                
                if((V.row(cohE(cohI, 0 + pI)) - V.row(cohE(cohI, 2 + pI))).norm() / avgEdgeLen > thres) {
                    continue;
                }
                
                int groupI0 = dupVI2GroupI[cohE(cohI, 0 + pI)];
                int groupI2 = dupVI2GroupI[cohE(cohI, 2 + pI)];
                if(groupI0 != groupI2) {
                    for(const auto& vI : meshVGroup[groupI2]) {
                        dupVI2GroupI[vI] = groupI0;
                        meshVGroup[groupI0].insert(vI);
                    }
                    meshVGroup[groupI2].clear();
                }
            }
        }
        
        int meshVAmt = 0;
        for(int gI = 0; gI < meshVGroup.size(); gI++) {
            if(!meshVGroup[gI].empty()) {
                meshVAmt++;
            }
        }
        std::vector<int> groupI2meshVI(meshVGroup.size(), -1);
        Eigen::MatrixXd UV_mesh;
        UV_mesh.resize(meshVAmt, 2);
        
        int meshVAmt_3D = 0;
        for(int gI = 0; gI < meshVGroup_3D.size(); gI++) {
            if(!meshVGroup_3D[gI].empty()) {
                meshVAmt_3D++;
            }
        }
        std::vector<int> groupI2meshVI_3D(meshVGroup_3D.size(), -1);
        Eigen::MatrixXd V_mesh;
        V_mesh.resize(meshVAmt_3D, 3);
        
        int nextVI = 0;
        for(int gI = 0; gI < meshVGroup.size(); gI++) {
            if(meshVGroup[gI].empty()) {
                continue;
            }
            
            groupI2meshVI[gI] = nextVI;
            UV_mesh.row(nextVI) = Eigen::RowVector2d::Zero();
            for(const auto dupVI : meshVGroup[gI]) {
                UV_mesh.row(nextVI) += V.row(dupVI);
            }
            UV_mesh.row(nextVI) /= meshVGroup[gI].size();
            nextVI++;
        }
        int nextVI_3D = 0;
        for(int gI = 0; gI < meshVGroup_3D.size(); gI++) {
            if(meshVGroup_3D[gI].empty()) {
                continue;
            }
            
            groupI2meshVI_3D[gI] = nextVI_3D;
            V_mesh.row(nextVI_3D) = Eigen::RowVector3d::Zero();
            for(const auto dupVI : meshVGroup_3D[gI]) {
                V_mesh.row(nextVI_3D) += V_rest.row(dupVI);
            }
            V_mesh.row(nextVI_3D) /= meshVGroup_3D[gI].size();
            nextVI_3D++;
        }
        
        Eigen::MatrixXi F_mesh, FUV_mesh;
        F_mesh.resize(F.rows(), 3);
        FUV_mesh.resize(F.rows(), 3);
        for(int triI = 0; triI < F.rows(); triI++) {
            for(int vI = 0; vI < 3; vI++) {
                int groupI = dupVI2GroupI[F(triI, vI)];
                assert(groupI >= 0);
                int meshVI = groupI2meshVI[groupI];
                assert(meshVI >= 0);
                FUV_mesh(triI, vI) = meshVI;
                
                int groupI_3D = dupVI2GroupI_3D[F(triI, vI)];
                assert(groupI_3D >= 0);
                int meshVI_3D = groupI2meshVI_3D[groupI_3D];
                assert(meshVI_3D >= 0);
                F_mesh(triI, vI) = meshVI_3D;
            }
        }
        
        if(scaleUV) {
            const Eigen::VectorXd& u = UV_mesh.col(0);
            const Eigen::VectorXd& v = UV_mesh.col(1);
            const double uMin = u.minCoeff();
            const double vMin = v.minCoeff();
            const double uScale = u.maxCoeff() - uMin;
            const double vScale = v.maxCoeff() - vMin;
            const double scale = std::max(uScale, vScale);
            for(int uvI = 0; uvI < UV_mesh.rows(); uvI++) {
                UV_mesh(uvI, 0) = (UV_mesh(uvI, 0) - uMin) / scale;
                UV_mesh(uvI, 1) = (UV_mesh(uvI, 1) - vMin) / scale;
            }
        }
        
        save(filePath, V_mesh, F_mesh, UV_mesh, FUV_mesh);
    }
    
}
