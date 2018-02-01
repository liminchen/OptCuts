//
//  TriangleSoup.cpp
//  FracCuts
//
//  Created by Minchen Li on 8/30/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#include "TriangleSoup.hpp"
#include "IglUtils.hpp"
#include "SymStretchEnergy.hpp"
#include "Optimizer.hpp"

#include <igl/cotmatrix.h>
#include <igl/avg_edge_length.h>
#include <igl/writeOBJ.h>
#include <igl/list_to_matrix.h>

#include <tbb/tbb.h>

#include <fstream>

extern std::ofstream logFile;

namespace FracCuts {
    
    TriangleSoup::TriangleSoup(void)
    {
        initSeamLen = 0.0;
    }
    
    TriangleSoup::TriangleSoup(const Eigen::MatrixXd& V_mesh, const Eigen::MatrixXi& F_mesh,
                               const Eigen::MatrixXd& UV_mesh, const Eigen::MatrixXi& FUV_mesh,
                               bool separateTri, double p_initSeamLen)
    {
        initSeamLen = p_initSeamLen;
        
        bool multiComp = false; //TODO: detect whether the mesh is multi-component
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
//                V.row(vDupIndStart + 1) = V.row(vDupIndStart) + 0.5 * (V.row(vDupIndStart + 1) - V.row(vDupIndStart));
//                V.row(vDupIndStart + 2) = V.row(vDupIndStart) + 0.5 * (V.row(vDupIndStart + 2) - V.row(vDupIndStart));
                
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
            }
        }
        else {
            // deal with mesh
            if(UV_mesh.rows() == V_mesh.rows()) {
                // same vertex and uv index
                V_rest = V_mesh;
                V = UV_mesh;
                F = F_mesh;
            }
            else if(UV_mesh.rows() != 0) {
                // different vertex and uv index, split 3D surface according to UV and merge back while saving into files
                assert(F_mesh.rows() == FUV_mesh.rows());
                // UV map contains seams
                // Split triangles along the seams on the surface (construct cohesive edges there)
                // to construct a bijective map
                std::set<std::pair<int, int>> HE_UV;
                std::map<std::pair<int, int>, std::pair<int, int>> HE;
                for(int triI = 0; triI < FUV_mesh.rows(); triI++) {
                    const Eigen::RowVector3i& triVInd_UV = FUV_mesh.row(triI);
                    HE_UV.insert(std::pair<int, int>(triVInd_UV[0], triVInd_UV[1]));
                    HE_UV.insert(std::pair<int, int>(triVInd_UV[1], triVInd_UV[2]));
                    HE_UV.insert(std::pair<int, int>(triVInd_UV[2], triVInd_UV[0]));
                    const Eigen::RowVector3i& triVInd = F_mesh.row(triI);
                    HE[std::pair<int, int>(triVInd[0], triVInd[1])] = std::pair<int, int>(triI, 0);
                    HE[std::pair<int, int>(triVInd[1], triVInd[2])] = std::pair<int, int>(triI, 1);
                    HE[std::pair<int, int>(triVInd[2], triVInd[0])] = std::pair<int, int>(triI, 2);
                }
                std::vector<std::vector<int>> cohEdges;
                for(int triI = 0; triI < FUV_mesh.rows(); triI++) {
                    const Eigen::RowVector3i& triVInd_UV = FUV_mesh.row(triI);
                    const Eigen::RowVector3i& triVInd = F_mesh.row(triI);
                    for(int eI = 0; eI < 3; eI++) {
                        int vI = eI, vI_post = (eI + 1) % 3;
                        if(HE_UV.find(std::pair<int, int>(triVInd_UV[vI_post], triVInd_UV[vI])) == HE_UV.end()) {
                            // boundary edge in UV space
                            const auto finder = HE.find(std::pair<int, int>(triVInd[vI_post], triVInd[vI]));
                            if(finder != HE.end()) {
                                // non-boundary edge on the surface
                                // construct cohesive edge pair
                                cohEdges.resize(cohEdges.size() + 1);
                                cohEdges.back().emplace_back(triVInd_UV[vI]);
                                cohEdges.back().emplace_back(triVInd_UV[vI_post]);
                                cohEdges.back().emplace_back(FUV_mesh(finder->second.first, (finder->second.second + 1) % 3));
                                cohEdges.back().emplace_back(FUV_mesh(finder->second.first, finder->second.second));
                                HE.erase(std::pair<int, int>(triVInd[vI], triVInd[vI_post])); // prevent from inserting again
                            }
                        }
                    }
                }
                bool makeCoh = true;
                if(makeCoh) {
                    igl::list_to_matrix(cohEdges, cohE);
                }
                
                V_rest.resize(UV_mesh.rows(), 3);
                V = UV_mesh;
                F = FUV_mesh;
                std::vector<bool> updated(UV_mesh.rows(), false);
                for(int triI = 0; triI < F_mesh.rows(); triI++) {
                    const Eigen::RowVector3i& triVInd = F_mesh.row(triI);
                    const Eigen::RowVector3i& triVInd_UV = FUV_mesh.row(triI);
                    for(int vI = 0; vI < 3; vI++) {
                        if(!updated[triVInd_UV[vI]]) {
                            V_rest.row(triVInd_UV[vI]) = V_mesh.row(triVInd[vI]);
                            updated[triVInd_UV[vI]] = true;
                        }
                    }
                }
                
                if(!makeCoh) {
                    for(const auto& cohI : cohEdges) {
                        initSeamLen += 2.0 * (V_rest.row(cohI[0]) - V_rest.row(cohI[1])).norm();
                    }
                }
                else {
                    initSeams = cohE;
                }
            }
            else {
                assert(V_mesh.rows() > 0);
                assert(F_mesh.rows() > 0);
                V_rest = V_mesh;
                F = F_mesh;
                V = Eigen::MatrixXd::Zero(V_rest.rows(), 2);
                std::cout << "No UV provided, initialized to all 0" << std::endl;
            }
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
        initSeamLen = 0.0;
    }
    
    void TriangleSoup::computeLaplacianMtr(void)
    {
        Eigen::SparseMatrix<double> L;
        igl::cotmatrix(V_rest, F, L);
        LaplacianMtr.resize(L.rows(), L.cols());
        LaplacianMtr.setZero();
        LaplacianMtr.reserve(L.nonZeros());
        for (int k = 0; k < L.outerSize(); ++k)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it)
            {
                if((fixedVert.find(static_cast<int>(it.row())) == fixedVert.end()) &&
                   (fixedVert.find(static_cast<int>(it.col())) == fixedVert.end()))
                {
                    LaplacianMtr.insert(it.row(), it.col()) = -it.value();
                }
            }
        }
        for(const auto fixedVI : fixedVert) {
            LaplacianMtr.insert(fixedVI, fixedVI) = 1.0;
        }
        LaplacianMtr.makeCompressed();
//        //        Eigen::SparseMatrix<double> M;
//        //        massmatrix(data.V_rest, data.F, igl::MASSMATRIX_TYPE_DEFAULT, M);
//                    LaplacianMtr.insert(it.row() * 2, it.col() * 2) = -it.value();// * M.coeffRef(it.row(), it.row());
//                    LaplacianMtr.insert(it.row() * 2 + 1, it.col() * 2 + 1) = -it.value();// * M.coeffRef(it.row(), it.row());
    }
    
    void TriangleSoup::computeFeatures(bool multiComp, bool resetFixedV)
    {
        //TODO: if the mesh is multi-component, then fix more vertices
        if(resetFixedV) {
            fixedVert.clear();
            fixedVert.insert(0);
        }
        
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
        
//        igl::cotmatrix_entries(V_rest, F, cotVals);
        
        triArea.resize(F.rows());
        surfaceArea = 0.0;
        triAreaSq.resize(F.rows());
        e0SqLen.resize(F.rows());
        e1SqLen.resize(F.rows());
        e0dote1.resize(F.rows());
        e0SqLen_div_dbAreaSq.resize(F.rows());
        e1SqLen_div_dbAreaSq.resize(F.rows());
        e0dote1_div_dbAreaSq.resize(F.rows());
        std::vector<Eigen::RowVector3d> vertNormals(V_rest.rows(), Eigen::Vector3d::Zero());
        for(int triI = 0; triI < F.rows(); triI++) {
            const Eigen::Vector3i& triVInd = F.row(triI);
            
            const Eigen::Vector3d& P1 = V_rest.row(triVInd[0]);
            const Eigen::Vector3d& P2 = V_rest.row(triVInd[1]);
            const Eigen::Vector3d& P3 = V_rest.row(triVInd[2]);
            
            const Eigen::Vector3d P2m1 = P2 - P1;
            const Eigen::Vector3d P3m1 = P3 - P1;
            const Eigen::RowVector3d normalVec = P2m1.cross(P3m1);
            
            triArea[triI] = 0.5 * normalVec.norm();
            surfaceArea += triArea[triI];
            triAreaSq[triI] = triArea[triI] * triArea[triI];
            e0SqLen[triI] = P2m1.squaredNorm();
            e1SqLen[triI] = P3m1.squaredNorm();
            e0dote1[triI] = P2m1.dot(P3m1);
            
            e0SqLen_div_dbAreaSq[triI] = e0SqLen[triI] / 2. / triAreaSq[triI];
            e1SqLen_div_dbAreaSq[triI] = e1SqLen[triI] / 2. / triAreaSq[triI];
            e0dote1_div_dbAreaSq[triI] = e0dote1[triI] / 2. / triAreaSq[triI];
            
            vertNormals[triVInd[0]] += normalVec;
            vertNormals[triVInd[1]] += normalVec;
            vertNormals[triVInd[2]] += normalVec;
        }
        avgEdgeLen = igl::avg_edge_length(V_rest, F);
        virtualPerimeter = avgEdgeLen * std::sqrt(F.rows());
        for(auto& vNI : vertNormals) {
            vNI.normalize();
        }
        
        computeLaplacianMtr();
        
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
        
        edge2Tri.clear();
        vNeighbor.resize(0);
        vNeighbor.resize(V_rest.rows());
        for(int triI = 0; triI < F.rows(); triI++) {
            const Eigen::RowVector3i& triVInd = F.row(triI);
            for(int vI = 0; vI < 3; vI++) {
                int vI_post = (vI + 1) % 3;
                edge2Tri[std::pair<int, int>(triVInd[vI], triVInd[vI_post])] = triI;
                vNeighbor[triVInd[vI]].insert(triVInd[vI_post]);
                vNeighbor[triVInd[vI_post]].insert(triVInd[vI]);
            }
        }
        cohEIndex.clear();
        for(int cohI = 0; cohI < cohE.rows(); cohI++) {
            const Eigen::RowVector4i& cohEI = cohE.row(cohI);
            if(cohEI.minCoeff() >= 0) {
                cohEIndex[std::pair<int, int>(cohEI[0], cohEI[1])] = cohI;
                cohEIndex[std::pair<int, int>(cohEI[3], cohEI[2])] = -cohI - 1;
            }
        }
        
        validSplit.resize(V_rest.rows());
        for(int vI = 0; vI < V_rest.rows(); vI++) {
            validSplit[vI].clear();
            if(isBoundaryVert(edge2Tri, vNeighbor, vI)) {
                continue;
            }
            
            std::vector<int> nbVs(vNeighbor[vI].begin(), vNeighbor[vI].end());
            std::vector<Eigen::RowVector3d> projectedEdge(nbVs.size());
            for(int nbI = 0; nbI < nbVs.size(); nbI++) {
                const Eigen::RowVector3d edge = V_rest.row(nbVs[nbI]) - V_rest.row(vI);
                projectedEdge[nbI] = (edge - edge.dot(vertNormals[vI]) * vertNormals[vI]).normalized();
            }
            for(int nbI = 0; nbI + 1 < nbVs.size(); nbI++) {
                for(int nbJ = nbI + 1; nbJ < nbVs.size(); nbJ++) {
                    if(projectedEdge[nbI].dot(projectedEdge[nbJ]) <= 0.0) {
                        validSplit[vI].insert(std::pair<int, int>(nbVs[nbI], nbVs[nbJ]));
                        validSplit[vI].insert(std::pair<int, int>(nbVs[nbJ], nbVs[nbI]));
                    }
                }
            }
        }
    }
    
    void TriangleSoup::updateFeatures(void)
    {
        const int nCE = static_cast<int>(boundaryEdge.size());
        boundaryEdge.conservativeResize(cohE.rows());
        edgeLen.conservativeResize(cohE.rows());
        for(int cohI = nCE; cohI < cohE.rows(); cohI++)
        {
            if(cohE.row(cohI).minCoeff() >= 0) {
                boundaryEdge[cohI] = 0;
            }
            else {
                boundaryEdge[cohI] = 1;
            }
            edgeLen[cohI] = (V_rest.row(cohE(cohI, 0)) - V_rest.row(cohE(cohI, 1))).norm();
        }
        
        computeLaplacianMtr();
    }
    
    void TriangleSoup::resetFixedVert(const std::set<int>& p_fixedVert)
    {
        for(const auto& vI : p_fixedVert) {
            assert(vI < V.rows());
        }
        
        fixedVert = p_fixedVert;
        computeLaplacianMtr();
    }
    
    bool TriangleSoup::separateTriangle(const Eigen::VectorXd& measure, double thres)
    {
        assert(measure.size() == F.rows());
        
        // separate triangles when necessary
        bool changed = false;
        for(int triI = 0; triI < F.rows(); triI++) {
            if(measure[triI] <= thres) {
                continue;
            }
            
            const Eigen::RowVector3i triVInd = F.row(triI);
            Eigen::Vector3i needSeparate = Eigen::Vector3i::Zero();
            std::map<std::pair<int, int>, int>::iterator edgeFinder[3];
            for(int eI = 0; eI < 3; eI++) {
                if(edge2Tri.find(std::pair<int, int>(triVInd[(eI + 1) % 3], triVInd[eI])) != edge2Tri.end()) {
                    needSeparate[eI] = 1;
                    edgeFinder[eI] = edge2Tri.find(std::pair<int, int>(triVInd[eI], triVInd[(eI + 1) % 3]));
                }
            }
            if(needSeparate.sum() == 0) {
                continue;
            }
            
            changed = true;
            if(needSeparate.sum() == 1) {
                // duplicate the edge
                for(int eI = 0; eI < 3; eI++) {
                    if(needSeparate[eI]) {
                        const int vI = triVInd[eI], vI_post = triVInd[(eI + 1) % 3];
                        const int nV = static_cast<int>(V_rest.rows());
                        V_rest.conservativeResize(nV + 2, 3);
                        V_rest.row(nV) = V_rest.row(vI);
                        V_rest.row(nV + 1) = V_rest.row(vI_post);
                        V.conservativeResize(nV + 2, 2);
                        V.row(nV) = V.row(vI);
                        V.row(nV + 1) = V.row(vI_post);
                        
                        F(triI, eI) = nV;
                        F(triI, (eI + 1) % 3) = nV + 1;
                        
                        const int nCE = static_cast<int>(cohE.rows());
                        cohE.conservativeResize(nCE + 1, 4);
                        cohE.row(nCE) << nV, nV + 1, vI, vI_post;
                        cohEIndex[std::pair<int, int>(nV, nV + 1)] = nCE;
                        cohEIndex[std::pair<int, int>(vI_post, vI)] = -nCE - 1;
                        
                        edge2Tri.erase(edgeFinder[eI]);
                        edge2Tri[std::pair<int, int>(nV, nV + 1)] = triI;
                        
                        const int vI_pre = triVInd[(eI + 2) % 3];
                        auto finder0 = cohEIndex.find(std::pair<int, int>(vI_post, vI_pre));
                        if(finder0 != cohEIndex.end()) {
                            if(finder0->second >= 0) {
                                cohE(finder0->second, 0) = nV + 1;
                            }
                            else {
                                cohE(-finder0->second - 1, 3) = nV + 1;
                            }
                            cohEIndex[std::pair<int, int>(nV + 1, vI_pre)] = finder0->second;
                            cohEIndex.erase(finder0);
                        }
                        auto finder1 = cohEIndex.find(std::pair<int, int>(vI_pre, vI));
                        if(finder1 != cohEIndex.end()) {
                            if(finder1->second >= 0) {
                                cohE(finder1->second, 1) = nV;
                            }
                            else {
                                cohE(-finder1->second - 1, 2) = nV;
                            }
                            cohEIndex[std::pair<int, int>(vI_pre, nV)] = finder1->second;
                            cohEIndex.erase(finder1);
                        }
                        
                        break;
                    }
                }
            }
            else if(needSeparate.sum() > 1) {
                std::vector<std::vector<int>> tri_toSep;
                std::vector<std::pair<int, int>> boundaryEdge;
                std::vector<int> vI_toSplit, vI_toSplit_post;
                std::vector<bool> needSplit;
                for(int eI = 0; eI < 3; eI++) {
                    if(needSeparate[eI] && needSeparate[(eI + 2) % 3]) {
                        vI_toSplit.emplace_back(triVInd[eI]);
                        vI_toSplit_post.emplace_back(triVInd[(eI + 1) % 3]);
                        tri_toSep.resize(tri_toSep.size() + 1);
                        boundaryEdge.resize(boundaryEdge.size() + 1);
                        needSplit.push_back(isBoundaryVert(edge2Tri, vI_toSplit.back(), vI_toSplit_post.back(),
                                                              tri_toSep.back(), boundaryEdge.back()));
                    }
                }
                
                // duplicate all vertices
                const int vI0 = triVInd[0], vI1 = triVInd[1], vI2 = triVInd[2];
                const int nV = static_cast<int>(V_rest.rows());
                V_rest.conservativeResize(nV + 3, 3);
                V_rest.row(nV) = V_rest.row(vI0);
                V_rest.row(nV + 1) = V_rest.row(vI1);
                V_rest.row(nV + 2) = V_rest.row(vI2);
                V.conservativeResize(nV + 3, 2);
                V.row(nV) = V.row(vI0);
                V.row(nV + 1) = V.row(vI1);
                V.row(nV + 2) = V.row(vI2);
                
                F.row(triI) << nV, nV + 1, nV + 2;
                
                // construct cohesive edges:
                for(int eI = 0; eI < 3; eI++) {
                    if(needSeparate[eI]) {
                        const int nCE = static_cast<int>(cohE.rows());
                        cohE.conservativeResize(nCE + 1, 4);
                        const int vI = eI, vI_post = (eI + 1) % 3;
                        cohE.row(nCE) << nV + vI, nV + vI_post, triVInd[vI], triVInd[vI_post];
                        cohEIndex[std::pair<int, int>(nV + vI, nV + vI_post)] = nCE;
                        cohEIndex[std::pair<int, int>(triVInd[vI_post], triVInd[vI])] = -nCE - 1;
                        
                        edge2Tri.erase(edgeFinder[eI]);
                        edge2Tri[std::pair<int, int>(nV + vI, nV + vI_post)] = triI;
                    }
                    else {
                        int vI = eI, vI_post = (eI + 1) % 3;
                        auto finder = cohEIndex.find(std::pair<int, int>(triVInd[vI], triVInd[vI_post]));
                        if(finder != cohEIndex.end()) {
                            if(finder->second >= 0) {
                                cohE(finder->second, 0) = nV + vI;
                                cohE(finder->second, 1) = nV + vI_post;
                            }
                            else {
                                cohE(-finder->second - 1, 3) = nV + vI;
                                cohE(-finder->second - 1, 2) = nV + vI_post;
                            }
                            cohEIndex[std::pair<int, int>(nV + vI, nV + vI_post)] = finder->second;
                            cohEIndex.erase(finder);
                        }
                    }
                }
                
                for(int sI = 0; sI < needSplit.size(); sI++) {
                    if(!needSplit[sI]) {
                        continue;
                    }
                    
                    assert(!tri_toSep.empty());
                    const int nV = static_cast<int>(V_rest.rows());
                    V_rest.conservativeResize(nV + 1, 3);
                    V_rest.row(nV) = V_rest.row(vI_toSplit[sI]);
                    V.conservativeResize(nV + 1, 2);
                    V.row(nV) = V.row(vI_toSplit[sI]);
                    for(const auto triToSepI : tri_toSep[sI]) {
                        int i = 0;
                        for(; i < 3; i++) {
                            if(F(triToSepI, i) == vI_toSplit[sI]) {
                                F(triToSepI, i) = nV;
                                int vI_post = F(triToSepI, (i + 1) % 3);
                                int vI_pre = F(triToSepI, (i + 2) % 3);
                                edge2Tri[std::pair<int, int>(nV, vI_post)] = triToSepI;
                                edge2Tri[std::pair<int, int>(vI_pre, nV)] = triToSepI;
                                edge2Tri.erase(std::pair<int, int>(vI_toSplit[sI], vI_post));
                                edge2Tri.erase(std::pair<int, int>(vI_pre, vI_toSplit[sI]));
                                break;
                            }
                        }
                        assert(i < 3);
                    }
                    auto finder = cohEIndex.find(std::pair<int, int>(vI_toSplit_post[sI], vI_toSplit[sI]));
                    assert(finder != cohEIndex.end());
                    if(finder->second >= 0) {
                        cohE(finder->second, 1) = nV;
                    }
                    else {
                        cohE(-finder->second - 1, 2) = nV;
                    }
                    cohEIndex[std::pair<int, int>(vI_toSplit_post[sI], nV)] = finder->second;
                    cohEIndex.erase(finder);
                    
                    finder = cohEIndex.find(boundaryEdge[sI]);
                    if(finder != cohEIndex.end()) {
                        if(finder->second >= 0) {
                            cohE(finder->second, 0) = nV;
                        }
                        else {
                            cohE(-finder->second - 1, 3) = nV;
                        }
                        cohEIndex[std::pair<int, int>(nV, boundaryEdge[sI].second)] = finder->second;
                        cohEIndex.erase(finder);
                    }
                }
            }
        }
        
        if(changed) {
            updateFeatures();
        }
        
        return changed;
    }
    
    bool TriangleSoup::splitVertex(const Eigen::VectorXd& measure, double thres)
    {
        assert(measure.rows() == V.rows());
        
        bool modified = false;
        for(int vI = 0; vI < measure.size(); vI++) {
            if(measure[vI] > thres) {
                if(isBoundaryVert(edge2Tri, vNeighbor, vI)) {
                    // right now only on boundary vertices
                    int vI_interior = -1;
                    for(const auto& vI_neighbor : vNeighbor[vI]) {
                        if((edge2Tri.find(std::pair<int, int>(vI, vI_neighbor)) != edge2Tri.end()) &&
                           (edge2Tri.find(std::pair<int, int>(vI_neighbor, vI)) != edge2Tri.end()))
                        {
                            vI_interior = vI_neighbor;
                            break;
                        }
                    }
                    if(vI_interior >= 0) {
//                        splitEdgeOnBoundary(std::pair<int, int>(vI, vI_interior), edge2Tri, vNeighbor, cohEIndex);
                        modified = true;
                    }
                }
            }
        }
        
        if(modified) {
            updateFeatures();
        }
        
        return modified;
    }
    
    void TriangleSoup::resetSubOptInfo(void)
    {
        curFracTail = -1;
    }
    
    bool TriangleSoup::splitEdge(double lambda_t, double thres, bool propagate, bool splitInterior)
    {
        const double filterExp_b = 0.8;
        const double filterExp_in = 0.8;
        
        std::vector<int> bestCandVerts;
        int bestCandAmt_b = 0; // number of boundary vertices to query
        if(!propagate) {
            Eigen::VectorXd divGradPerVert;
            SymStretchEnergy SD;
            SD.computeDivGradPerVert(*this, divGradPerVert);
            
            std::map<double, int> sortedCandVerts_b, sortedCandVerts_in;
            for(int vI = 0; vI < V_rest.rows(); vI++) {
                if(vNeighbor[vI].size() <= 2) {
                    // this vertex is impossible to be splitted further
                    continue;
                }
                
                if(isBoundaryVert(edge2Tri, vNeighbor, vI)) {
                    sortedCandVerts_b[-divGradPerVert[vI]] = vI;
                }
                else if(splitInterior) {
                    sortedCandVerts_in[-divGradPerVert[vI]] = vI;
                }
            }
            
            {
                bestCandAmt_b = static_cast<int>(std::pow(sortedCandVerts_b.size(), filterExp_b));
                if(bestCandAmt_b < 2) {
                    bestCandAmt_b = 2;
                }
                bestCandVerts.reserve(bestCandAmt_b);
                for(const auto& candI : sortedCandVerts_b) {
                    bestCandVerts.emplace_back(candI.second);
                    if(bestCandVerts.size() >= bestCandAmt_b) {
                        break;
                    }
                }
            }
            if(splitInterior)
            {
                assert(!sortedCandVerts_in.empty());
                int bestCandAmt_in = static_cast<int>(std::pow(sortedCandVerts_in.size(), filterExp_in));
                if(bestCandAmt_in < 2) {
                    bestCandAmt_in = 2;
                }
                bestCandVerts.reserve(bestCandVerts.size() + bestCandAmt_in);
                int counter = 0;
                for(const auto& candI : sortedCandVerts_in) {
                    bestCandVerts.emplace_back(candI.second);
                    if(++counter >= bestCandAmt_in) {
                        break;
                    }
                }
            }
            
            resetSubOptInfo();
        }
        else {
            // see whether fracture could be propagated from each fracture tail
            if(fracTail.empty()) {
                return false;
            }
            else {
                splitInterior = false;
                bestCandVerts.insert(bestCandVerts.end(), fracTail.begin(), fracTail.end());
                bestCandAmt_b = static_cast<int>(bestCandVerts.size());
            }
        }
        
        if(bestCandVerts.empty() || (bestCandAmt_b > bestCandVerts.size())) {
            std::cout << bestCandAmt_b << " " << bestCandVerts.size() << std::endl;
            assert(0);
        }
        
        // evaluate local energy decrease
        std::cout << "evaluate vertex splits, " << bestCandVerts.size() << " candidate verts" << std::endl;
        // run in parallel:
        std::vector<double> EwDecs(bestCandVerts.size());
        std::vector<std::vector<int>> paths(bestCandVerts.size());
        std::vector<Eigen::MatrixXd> newVertPoses(bestCandVerts.size());
        // query boundary splits
        tbb::parallel_for(0, bestCandAmt_b, 1, [&](int candI) {
            EwDecs[candI] = computeLocalEwDec(bestCandVerts[candI], lambda_t, paths[candI], newVertPoses[candI]);
        });
        if(splitInterior) {
            assert(!propagate);
            assert(bestCandAmt_b < bestCandVerts.size());
            // query interior splits
            tbb::parallel_for(bestCandAmt_b, (int)bestCandVerts.size(), 1, [&](int candI) {
                EwDecs[candI] = 0.5 * computeLocalEwDec(bestCandVerts[candI], lambda_t, paths[candI], newVertPoses[candI]);
            });
        }
        int candI_max = 0;
        for(int candI = 1; candI < bestCandVerts.size(); candI++) {
            if(EwDecs[candI] > EwDecs[candI_max]) {
                candI_max = candI;
            }
        }
        std::cout << "E_dec threshold = " << thres << std::endl;
        if(EwDecs[candI_max] > thres) {
            if(candI_max < bestCandAmt_b) {
                // boundary split
                std::cout << "boundary split E_dec = " << EwDecs[candI_max] << std::endl;
                splitEdgeOnBoundary(std::pair<int, int>(paths[candI_max][0], paths[candI_max][1]),
                                    newVertPoses[candI_max], edge2Tri, vNeighbor, cohEIndex);
                //TODO: process fractail here!
                updateFeatures();
            }
            else {
                assert(splitInterior && (!propagate));
                // interior split
                std::cout << "interior split E_dec = " << EwDecs[candI_max] << std::endl;
                cutPath(paths[candI_max], true, 1, newVertPoses[candI_max]);
                //                fracTail.insert(path_in[0]);
                //                fracTail.insert(path_in[2]);
            }
            return true;
        }
        else {
            std::cout << "max E_dec = " << EwDecs[candI_max] << " < thres" << std::endl;
            return false;
        }
    }
    
    bool TriangleSoup::mergeEdge(void)
    {
        //TODO: check for element inversion only locally right now
        //TODO: share the precomputation with split
        
        // compute stress tensor and do SVD
        std::vector<Eigen::JacobiSVD<Eigen::Matrix2d>> dg(F.rows());
        for(int triI = 0; triI < F.rows(); triI++) {
            const Eigen::RowVector3i& triVInd = F.row(triI);
            
            const Eigen::Vector3d x_3D[3] = {
                V_rest.row(triVInd[0]),
                V_rest.row(triVInd[1]),
                V_rest.row(triVInd[2])
            };
            
            const Eigen::Vector2d u[3] = {
                V.row(triVInd[0]),
                V.row(triVInd[1]),
                V.row(triVInd[2])
            };

            Eigen::Matrix2d stressTensor;
            SymStretchEnergy::computeStressTensor(x_3D, u, stressTensor);
            dg[triI].compute(stressTensor, Eigen::ComputeFullU | Eigen::ComputeFullV);
        }
        
        int minI = -1, minForkVI = 0;
        double minStretch = 0.25; // the initial value is the stretch strength threshold
        for(int cohI = 0; cohI < cohE.rows(); cohI++) {
            int forkVI = 0;
            if(cohE(cohI, 0) == cohE(cohI, 2))
            {
                forkVI = 0;
            }
            else if(cohE(cohI, 1) == cohE(cohI, 3))
            {
                forkVI = 1;
            }
            else {
                // only consider "zipper bottom" edge pairs
                continue;
            }
            
            const double dist = (V.row(cohE(cohI, 1 - forkVI)) - V.row(cohE(cohI, 3 - forkVI))).norm() / avgEdgeLen;
            if(dist < 0.25) { // when they are close enough
                const Eigen::Vector2d mergedPos = (V.row(cohE(cohI, 1 - forkVI)) + V.row(cohE(cohI, 3 - forkVI))) / 2.0;
                
                const Eigen::RowVector2d backup0 = V.row(cohE(cohI, 1 - forkVI));
                const Eigen::RowVector2d backup1 = V.row(cohE(cohI, 3 - forkVI));
                V.row(cohE(cohI, 1 - forkVI)) = mergedPos;
                V.row(cohE(cohI, 3 - forkVI)) = mergedPos;
                bool noInversion = checkInversion();
                V.row(cohE(cohI, 1 - forkVI)) = backup0;
                V.row(cohE(cohI, 3 - forkVI)) = backup1;
                if(!noInversion) {
                    continue;
                }
                
                const Eigen::Vector2d mergedEdgeDir = (V.row(cohE(cohI, 0 + forkVI)).transpose() - mergedPos).normalized();
                Eigen::JacobiSVD<Eigen::Matrix2d> svd_stressTensor[2];
                for(int eI = 0; eI < 2; eI++) {
                    auto eIFinder = edge2Tri.find(std::pair<int, int>(cohE(cohI, eI * 3), cohE(cohI, 1 + eI)));
                    assert(eIFinder != edge2Tri.end());
                    Eigen::Vector3d x3D[3];
                    Eigen::Vector2d uv[3];
                    for(int vI = 0; vI < 3; vI++) {
                        if(F(eIFinder->second, vI) == cohE(cohI, eI * 2 + 1 - forkVI)) {
                            uv[vI] = mergedPos;
                        }
                        else {
                            uv[vI] = V.row(F(eIFinder->second, vI));
                        }
                        x3D[vI] = V_rest.row(F(eIFinder->second, vI));
                    }
                    Eigen::Matrix2d stressTensor;
                    SymStretchEnergy::computeStressTensor(x3D, uv, stressTensor);
                    svd_stressTensor[eI].compute(stressTensor, Eigen::ComputeFullU | Eigen::ComputeFullV);
                }
                const Eigen::Vector2d& stretchDir0 = svd_stressTensor[0].matrixU().block(0, 0, 2, 1);
                const Eigen::Vector2d& stretchDir1 = svd_stressTensor[1].matrixU().block(0, 0, 2, 1);
                const double cosine0 = std::abs(mergedEdgeDir.dot(stretchDir0));
                const double cosine1 = std::abs(mergedEdgeDir.dot(stretchDir1));
                const double stretchStrength = svd_stressTensor[0].singularValues()[0] * (1.0 - cosine0) +
                    svd_stressTensor[1].singularValues()[0] * (1.0 - cosine1);
                
                if(stretchStrength < minStretch) {
                    minStretch = stretchStrength;
                    minI = cohI;
                    minForkVI = forkVI;
                }
            }
        }
        
        if(minI >= 0) {
            if(minForkVI == 0) {
                mergeBoundaryEdges(std::pair<int, int>(cohE(minI, 3), cohE(minI, 2)),
                               std::pair<int, int>(cohE(minI, 0), cohE(minI, 1)),
                               edge2Tri, vNeighbor, cohEIndex);
            }
            else if(minForkVI == 1) {
                mergeBoundaryEdges(std::pair<int, int>(cohE(minI, 0), cohE(minI, 1)),
                                   std::pair<int, int>(cohE(minI, 3), cohE(minI, 2)),
                                   edge2Tri, vNeighbor, cohEIndex);
            }
            else {
                assert(0 && "Invalid fork vertex index!");
            }
            std::cout << "edge merged" << std::endl;
            
            if(!checkInversion()) {
                
            }

            computeFeatures(); //TODO: only update locally
            return true;
        }
        else {
            return false;
        }
    }
    
    void TriangleSoup::onePointCut(int vI)
    {
        assert((vI >= 0) && (vI < V_rest.rows()));
        std::vector<int> path(vNeighbor[vI].begin(), vNeighbor[vI].end());
        assert(path.size() >= 3);
        path[1] = vI;
        path.resize(3);
        
        bool makeCoh = true;
        if(!makeCoh) {
            for(int pI = 0; pI + 1 < path.size(); pI++) {
                initSeamLen += 2.0 * (V_rest.row(path[pI]) - V_rest.row(path[pI + 1])).norm();
            }
        }
        
        cutPath(path, makeCoh);
        
        if(makeCoh) {
            initSeams = cohE;
        }
    }
    
    void TriangleSoup::highCurvOnePointCut(void)
    {
        std::vector<double> gaussianCurv(V.rows(), 2.0 * M_PI);
        for(int triI = 0; triI < F.rows(); triI++) {
            const Eigen::RowVector3i& triVInd = F.row(triI);
            const Eigen::RowVector3d v[3] = {
                V_rest.row(triVInd[0]),
                V_rest.row(triVInd[1]),
                V_rest.row(triVInd[2])
            };
            for(int vI = 0; vI < 3; vI++) {
                int vI_post = (vI + 1) % 3;
                int vI_pre = (vI + 2) % 3;
                const Eigen::RowVector3d e0 = v[vI_pre] - v[vI];
                const Eigen::RowVector3d e1 = v[vI_post] - v[vI];
                gaussianCurv[triVInd[vI]] -= std::acos(std::max(-1.0, std::min(1.0, e0.dot(e1) / e0.norm() / e1.norm())));
            }
        }
        
        double maxGC = 0.0;
        int vI_maxGC = -1;
        for(int vI = 0; vI < gaussianCurv.size(); vI++) {
            if(gaussianCurv[vI] > maxGC) {
                maxGC = gaussianCurv[vI];
                vI_maxGC = vI;
            }
        }
        onePointCut(vI_maxGC);
    }
    
    // A utility function to find the vertex with minimum distance value, from
    // the set of vertices not yet included in shortest path tree
    int minDistance(const std::vector<double>& dist, const std::vector<bool>& sptSet)
    {
        // Initialize min value
        double min = __DBL_MAX__;
        int min_index = -1;
        
        for (int v = 0; v < dist.size(); v++) {
            if ((!sptSet[v]) && (dist[v] <= min)) {
                min = dist[v], min_index = v;
            }
        }
        
        return min_index;
    }
    
    // Funtion that implements Dijkstra's single source shortest path algorithm
    // for a graph represented using adjacency matrix representation
    void dijkstra(const std::vector<std::map<int, double>>& graph, int src, std::vector<int>& path)
    {
        int nV = static_cast<int>(graph.size());
        
        std::vector<double> dist(nV, __DBL_MAX__);     // The output array.  dist[i] will hold the shortest
        // distance from src to i
        
        std::vector<bool> sptSet(nV, false); // sptSet[i] will true if vertex i is included in shortest
        // path tree or shortest distance from src to i is finalized
        
        std::vector<int> parent(nV, -1);
        
        // Distance of source vertex from itself is always 0
        dist[src] = 0.0;
        
        // Find shortest path for all vertices
        for (int count = 0; count + 1 < nV; count++)
        {
            // Pick the minimum distance vertex from the set of vertices not
            // yet processed. u is always equal to src in first iteration.
            int u = minDistance(dist, sptSet);
            
            // Mark the picked vertex as processed
            sptSet[u] = true;
            
            for(const auto v : graph[u]) {
                // Update dist[v] only if is not in sptSet, there is an edge from
                // u to v, and total weight of path from src to  v through u is
                // smaller than current value of dist[v]
                if ((!sptSet[v.first]) && (dist[u] != __DBL_MAX__)
                    && (dist[u] + v.second < dist[v.first]))
                {
                    dist[v.first] = dist[u] + v.second;
                    parent[v.first] = u;
                }
            }
        }
        
        double maxDist = 0.0;
        int vI_maxDist = -1;
        for(int vI = 0; vI < nV; vI++) {
            if(dist[vI] > maxDist) {
                maxDist = dist[vI];
                vI_maxDist = vI;
            }
        }
        assert(vI_maxDist >= 0);
        path.resize(0);
        while(vI_maxDist >= 0) {
            path.emplace_back(vI_maxDist);
            vI_maxDist = parent[vI_maxDist];
        }
        std::reverse(path.begin(), path.end());
    }
    
    void TriangleSoup::farthestPointCut(void)
    {
        assert(vNeighbor.size() == V_rest.rows());
        
        std::vector<std::map<int, double>> graph(vNeighbor.size());
        for(int vI = 0; vI < vNeighbor.size(); vI++) {
            for(const auto nbI : vNeighbor[vI]) {
                if(nbI > vI) {
                    graph[nbI][vI] = graph[vI][nbI] = (V_rest.row(vI) - V_rest.row(nbI)).norm();
                }
            }
        }
        
        std::vector<int> path;
        dijkstra(graph, 0, path);
        dijkstra(graph, path.back(), path);
        
        bool makeCoh = true;
        if(!makeCoh) {
            for(int pI = 0; pI + 1 < path.size(); pI++) {
                initSeamLen += 2.0 * (V_rest.row(path[pI]) - V_rest.row(path[pI + 1])).norm();
            }
        }
        
        cutPath(path, makeCoh);
//        save("/Users/mincli/Downloads/meshes/test_triSoup.obj");
//        saveAsMesh("/Users/mincli/Downloads/meshes/test_mesh.obj");
        
        if(makeCoh) {
            initSeams = cohE;
        }
    }
    
    void TriangleSoup::cutPath(const std::vector<int>& path, bool makeCoh, int changePos, const Eigen::MatrixXd& newVertPos)
    {
        assert(path.size() >= 3);
        if(changePos) {
            assert((changePos == 1) && "right now only support change 1");
            assert(newVertPos.cols() == 2);
            assert(changePos * 2 == newVertPos.rows());
        }
        
        if(isBoundaryVert(edge2Tri, vNeighbor, path[0])) {
            assert(0 && "forbiddened for preventing redundant cut");
        }
        else if(isBoundaryVert(edge2Tri, vNeighbor, path[2])) {
            assert(0 && "forbiddened for preventing redundant cut");
        }
        else {
            std::vector<int> tri_left;
            int vI = path[1];
            int vI_new = path[0];
            while(1) {
                auto finder = edge2Tri.find(std::pair<int, int>(vI, vI_new));
                assert(finder != edge2Tri.end());
                tri_left.emplace_back(finder->second);
                const Eigen::RowVector3i& triVInd = F.row(finder->second);
                for(int i = 0; i < 3; i++) {
                    if((triVInd[i] != vI) && (triVInd[i] != vI_new)) {
                        vI_new = triVInd[i];
                        break;
                    }
                }
                
                if(vI_new == path[2]) {
                    break;
                }
                if(vI_new == path[0]) {
                    assert(0 && "not a valid path!");
                }
            }
            
            int nV = static_cast<int>(V_rest.rows());
            V_rest.conservativeResize(nV + 1, 3);
            V_rest.row(nV) = V_rest.row(path[1]);
            V.conservativeResize(nV + 1, 2);
            if(changePos) {
                V.row(nV) = newVertPos.block(0, 0, 1, 2);
                V.row(path[1]) = newVertPos.block(1, 0, 1, 2);
            }
            else {
                V.row(nV) = V.row(path[1]);
            }
            for(const auto triI : tri_left) {
                for(int vI = 0; vI < 3; vI++) {
                    if(F(triI, vI) == path[1]) {
                        F(triI, vI) = nV;
                        break;
                    }
                }
            }
            if(makeCoh) {
                int nCoh = static_cast<int>(cohE.rows());
                cohE.conservativeResize(nCoh + 2, 4);
                cohE.row(nCoh) << nV, path[0], path[1], path[0];
                cohE.row(nCoh + 1) << path[2], nV, path[2], path[1];
            }
            
            computeFeatures(); //TODO: only update locally
            
            for(int vI = 2; vI + 1 < path.size(); vI++) {
                //TODO: enable change pos!
                int vInd_s = path[vI];
                int vInd_e = path[vI + 1];
                assert(edge2Tri.find(std::pair<int, int>(vInd_s, vInd_e)) != edge2Tri.end());
                assert(edge2Tri.find(std::pair<int, int>(vInd_e, vInd_s)) != edge2Tri.end());
                Eigen::Matrix2d newVertPos;
                newVertPos << V.row(vInd_s), V.row(vInd_s);
                splitEdgeOnBoundary(std::pair<int, int>(vInd_s, vInd_e), newVertPos, edge2Tri, vNeighbor, cohEIndex); //!!! make coh?
                updateFeatures();
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
    void TriangleSoup::computeBoundaryLen(double& boundaryLen) const
    {
        boundaryLen = 0.0;
        for(const auto& e : edge2Tri) {
            if(edge2Tri.find(std::pair<int, int>(e.first.second, e.first.first)) == edge2Tri.end()) {
                boundaryLen += (V_rest.row(e.first.second) - V_rest.row(e.first.first)).norm();
            }
        }
    }
    void TriangleSoup::computeSeamSparsity(double& sparsity, bool triSoup) const
    {
        const double thres = 1.0e-2;
        sparsity = 0.0;
        for(int cohI = 0; cohI < cohE.rows(); cohI++)
        {
            if(!boundaryEdge[cohI]) {
                if((!triSoup) ||
                   ((V.row(cohE(cohI, 0)) - V.row(cohE(cohI, 2))).norm() / avgEdgeLen > thres) ||
                   ((V.row(cohE(cohI, 1)) - V.row(cohE(cohI, 3))).norm() / avgEdgeLen > thres))
                {
                    sparsity += 2.0 * edgeLen[cohI];
                }
            }
        }
        sparsity += initSeamLen;
    }
    void TriangleSoup::computeStandardStretch(double& stretch_l2, double& stretch_inf, double& stretch_shear) const
    {
        stretch_l2 = 0.0;
        stretch_inf = 0.0;
        stretch_shear = 0.0;
        for(int triI = 0; triI < F.rows(); triI++)
        {
            const Eigen::Vector3i& triVInd = F.row(triI);
            const Eigen::Vector3d x_3D[3] = {
                V_rest.row(triVInd[0]),
                V_rest.row(triVInd[1]),
                V_rest.row(triVInd[2])
            };
            const Eigen::Vector2d uv[3] = {
                V.row(triVInd[0]),
                V.row(triVInd[1]),
                V.row(triVInd[2])
            };
            Eigen::Matrix2d dg;
            IglUtils::computeDeformationGradient(x_3D, uv, dg);
            
            const double a = Eigen::RowVector2d(dg.block(0, 0, 1, 2)).squaredNorm();
            const double b = Eigen::RowVector2d(dg.block(0, 0, 1, 2)).dot(Eigen::RowVector2d(dg.block(1, 0, 1, 2)));
            const double c = Eigen::RowVector2d(dg.block(1, 0, 1, 2)).squaredNorm();
            const double t0 = a + c;
            const double t1 = std::sqrt((a - c) * (a - c) + 4. * b * b);
            const double tau = std::sqrt((t0 + t1) / 2.);
            
            stretch_l2 += t0 / 2.0 * triArea[triI];
            
            if(tau > stretch_inf) {
                stretch_inf = tau;
            }
            
            stretch_shear += b * b / a / c * triArea[triI];
        }
        stretch_l2 /= surfaceArea;
        stretch_l2 = std::sqrt(stretch_l2);
        stretch_shear /= surfaceArea;
        stretch_shear = std::sqrt(stretch_shear);
    }
    void TriangleSoup::outputStandardStretch(std::ofstream& file) const
    {
        double stretch_l2, stretch_inf, stretch_shear;
        computeStandardStretch(stretch_l2, stretch_inf, stretch_shear);
        file << stretch_l2 << " " << stretch_inf << " " << stretch_shear << std::endl;
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
    
    bool TriangleSoup::checkInversion(void) const
    {
        const double eps = 0.0;//1.0e-20 * avgEdgeLen * avgEdgeLen;
        for(int triI = 0; triI < F.rows(); triI++)
        {
            const Eigen::Vector3i& triVInd = F.row(triI);
            
            const Eigen::Vector2d e_u[2] = {
                V.row(triVInd[1]) - V.row(triVInd[0]),
                V.row(triVInd[2]) - V.row(triVInd[0])
            };
            
            const double dbArea = e_u[0][0] * e_u[1][1] - e_u[0][1] * e_u[1][0];
            if(dbArea < eps)
            {
                std::cout << "***Element inversion detected: " << dbArea << " < " << eps << std::endl;
                logFile << "***Element inversion detected: " << dbArea << " < " << eps << std::endl;
                return false;
            }
        }
        
        return true;
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
    
    bool TriangleSoup::findBoundaryEdge(int vI, const std::pair<int, int>& startEdge,
                          const std::map<std::pair<int, int>, int>& edge2Tri,
                          std::pair<int, int>& boundaryEdge)
    {
        auto finder = edge2Tri.find(startEdge);
        assert(finder != edge2Tri.end());
        bool proceed = (startEdge.first != vI);
        const int vI_neighbor = (proceed ? startEdge.first : startEdge.second);
        int vI_new = vI_neighbor;
        while(1) {
            const Eigen::RowVector3i& triVInd = F.row(finder->second);
            for(int i = 0; i < 3; i++) {
                if((triVInd[i] != vI) && (triVInd[i] != vI_new)) {
                    vI_new = triVInd[i];
                    break;
                }
            }
            
            if(vI_new == vI_neighbor) {
                return false;
            }
            
            finder = edge2Tri.find(proceed ? std::pair<int, int>(vI_new, vI) : std::pair<int, int>(vI, vI_new));
            if(finder == edge2Tri.end()) {
                boundaryEdge.first = (proceed ? vI : vI_new);
                boundaryEdge.second = (proceed ? vI_new : vI);
                return true;
            }
        }
    }
    
    bool TriangleSoup::isBoundaryVert(const std::map<std::pair<int, int>, int>& edge2Tri, int vI, int vI_neighbor,
                                      std::vector<int>& tri_toSep, std::pair<int, int>& boundaryEdge, bool toBound) const
    {
//        const auto inputEdgeTri = edge2Tri.find(toBound ? std::pair<int, int>(vI, vI_neighbor) :
//                                                std::pair<int, int>(vI_neighbor, vI));
//        assert(inputEdgeTri != edge2Tri.end());
        
        tri_toSep.resize(0);
        auto finder = edge2Tri.find(toBound ? std::pair<int, int>(vI_neighbor, vI):
                                    std::pair<int, int>(vI, vI_neighbor));
        if(finder == edge2Tri.end()) {
            boundaryEdge.first = (toBound ? vI : vI_neighbor);
            boundaryEdge.second = (toBound ? vI_neighbor : vI);
            return true;
        }
        
        int vI_new = vI_neighbor;
        do {
            tri_toSep.emplace_back(finder->second);
            const Eigen::RowVector3i& triVInd = F.row(finder->second);
            for(int i = 0; i < 3; i++) {
                if((triVInd[i] != vI) && (triVInd[i] != vI_new)) {
                    vI_new = triVInd[i];
                    break;
                }
            }
            
            if(vI_new == vI_neighbor) {
                return false;
            }
            
            finder = edge2Tri.find(toBound ? std::pair<int, int>(vI_new, vI) :
                                   std::pair<int, int>(vI, vI_new));
            if(finder == edge2Tri.end()) {
                boundaryEdge.first = (toBound ? vI : vI_new);
                boundaryEdge.second = (toBound ? vI_new : vI);
                return true;
            }
        } while(1);
    }
    
    bool TriangleSoup::isBoundaryVert(const std::map<std::pair<int, int>, int>& edge2Tri,
                                      const std::vector<std::set<int>>& vNeighbor, int vI) const
    {
        assert(vNeighbor.size() == V.rows());
        assert(vI < vNeighbor.size());
        
        for(const auto vI_neighbor : vNeighbor[vI]) {
            if((edge2Tri.find(std::pair<int, int>(vI, vI_neighbor)) == edge2Tri.end()) ||
                (edge2Tri.find(std::pair<int, int>(vI_neighbor, vI)) == edge2Tri.end()))
            {
                return true;
            }
        }
        
        return false;
    }
    
    double TriangleSoup::computeLocalEwDec(int vI, double lambda_t, std::vector<int>& path_max, Eigen::MatrixXd& newVertPos_max) const
    {
        std::vector<int> umbrella;
        std::pair<int, int> boundaryEdge;
        if(isBoundaryVert(edge2Tri, vI, *(vNeighbor[vI].begin()), umbrella, boundaryEdge, false)) {
            // boundary split
            double maxEwDec = -__DBL_MAX__;
            path_max.resize(2);
            for(const auto& nbVI : vNeighbor[vI]) {
                const std::pair<int, int> edge(vI, nbVI);
                if((edge2Tri.find(edge) != edge2Tri.end()) &&
                   (edge2Tri.find(std::pair<int, int>(nbVI, vI)) != edge2Tri.end()))
                {
                    // interior edge
                    Eigen::MatrixXd newVertPosI;
                    const double seInc = (V_rest.row(vI) - V_rest.row(nbVI)).norm() * 2.0 / virtualPerimeter;
                    const double SDDec = computeLocalEDec(edge, edge2Tri, vNeighbor, cohEIndex, newVertPosI);
                    const double curEwDec = (1.0 - lambda_t) * SDDec - lambda_t * seInc;
                    if(curEwDec > maxEwDec) {
                        maxEwDec = curEwDec;
                        path_max[0] = vI;
                        path_max[1] = nbVI;
                        newVertPos_max = newVertPosI;
                    }
                }
            }
            return maxEwDec;
        }
        else {
            // interior split
            for(const auto& nbVI : vNeighbor[vI]) {
                if(isBoundaryVert(edge2Tri, vNeighbor, nbVI)) {
                    return -__DBL_MAX__; // don't split vertices connected to boundary here
                }
            }
            
//            std::cout << "umbrella obtained, size " << umbrella.size() << std::endl;
            if(umbrella.size() > 10) {
                std::cout << "large degree vert, " << umbrella.size() << " incident tris" << std::endl;
                logFile << "large degree vert, " << umbrella.size() << " incident tris" << std::endl;
            }
            
            path_max.resize(3);
            double EwDec_max = -__DBL_MAX__;
            std::set<int> freeVert;
            freeVert.insert(vI);
            std::map<int, Eigen::RowVector2d> newVertPosMap;
            std::vector<int> path(3);
            path[1] = vI;
            for(int startI = 0; startI + 1 < umbrella.size(); startI++) {
                for(int i = 0; i < 3; i++) {
                    if(F(umbrella[startI], i) == vI) {
                        path[0] = F(umbrella[startI], (i + 1) % 3);
                        break;
                    }
                }
                
                for(int endI = startI + 1; endI < umbrella.size(); endI++) {
                    for(int i = 0; i < 3; i++) {
                        if(F(umbrella[endI], i) == vI) {
                            path[2] = F(umbrella[endI], (i + 1) % 3);
                            break;
                        }
                    }
                    
//                    // don't make sharp turn splits
//                    if(validSplit[vI].find(std::pair<int, int>(path[0], path[2])) == validSplit[vI].end()) {
//                        continue;
//                    }
                    
                    double SDDec = 0.0;
                    Eigen::Matrix2d newVertPos;
                    
                    std::vector<int> triangles(umbrella.begin() + startI, umbrella.begin() + endI);
                    SDDec += computeLocalEDec(triangles, freeVert, newVertPosMap);
                    newVertPos.block(0, 0, 1, 2) = newVertPosMap[vI];
                    
                    triangles.resize(0);
                    triangles.insert(triangles.end(), umbrella.begin(), umbrella.begin() + startI);
                    triangles.insert(triangles.end(), umbrella.begin() + endI, umbrella.end());
                    SDDec += computeLocalEDec(triangles, freeVert, newVertPosMap);
                    newVertPos.block(1, 0, 1, 2) = newVertPosMap[vI];
                    
                    const double seInc = 2.0 * ((V_rest.row(path[0]) - V_rest.row(path[1])).norm() +
                                                (V_rest.row(path[1]) - V_rest.row(path[2])).norm()) / virtualPerimeter;
                    const double EwDec = (1.0 - lambda_t) * SDDec - lambda_t * seInc;
                    if(EwDec > EwDec_max) {
                        EwDec_max = EwDec;
                        newVertPos_max = newVertPos;
                        path_max = path;
                    }
                }
            }
            return EwDec_max;
        }
    }
    
    double TriangleSoup::computeLocalEDec(const std::vector<int>& triangles, const std::set<int>& freeVert,
                                          std::map<int, Eigen::RowVector2d>& newVertPos, int maxIter) const
    {
        assert(triangles.size() && freeVert.size());
        
        // construct local mesh
        Eigen::MatrixXi localF;
        localF.resize(triangles.size(), 3);
        Eigen::MatrixXd localV_rest, localV;
        std::set<int> fixedVert;
        std::map<int, int> globalVI2local;
        int localTriI = 0;
        for(const auto triI : triangles) {
            for(int vI = 0; vI < 3; vI++) {
                int globalVI = F(triI, vI);
                auto localVIFinder = globalVI2local.find(globalVI);
                if(localVIFinder == globalVI2local.end()) {
                    int localVI = static_cast<int>(localV_rest.rows());
                    if(freeVert.find(globalVI) == freeVert.end()) {
                        fixedVert.insert(localVI);
                    }
                    localV_rest.conservativeResize(localVI + 1, 3);
                    localV_rest.row(localVI) = V_rest.row(globalVI);
                    localV.conservativeResize(localVI + 1, 2);
                    localV.row(localVI) = V.row(globalVI);
                    localF(localTriI, vI) = localVI;
                    globalVI2local[globalVI] = localVI;
                }
                else {
                    localF(localTriI, vI) = localVIFinder->second;
                }
            }
            localTriI++;
        }
        TriangleSoup localMesh(localV_rest, localF, localV, Eigen::MatrixXi(), false);
        localMesh.resetFixedVert(fixedVert);
        
        // conduct optimization on local mesh
        std::vector<FracCuts::Energy*> energyTerms(1, new SymStretchEnergy());
        std::vector<double> energyParams(1, 1.0);
        Optimizer optimizer(localMesh, energyTerms, energyParams, false, true);
        optimizer.precompute();
        optimizer.setRelGL2Tol(1.0e-4);
        double initE = optimizer.getLastEnergyVal();
        optimizer.solve(maxIter); //do not output, the other part
        //            std::cout << "local opt " << optimizer.getIterNum() << " iters" << std::endl;
        const double eDec = (initE - optimizer.getLastEnergyVal()) * localMesh.surfaceArea / surfaceArea; //!!! this should be written in a more general way, cause this way it only works for E_SD
        
        // get new vertex positions
        newVertPos.clear();
        for(const auto& vI_free : freeVert) {
            newVertPos[vI_free] = optimizer.getResult().V.row(globalVI2local[vI_free]);
        }
        
        delete energyTerms[0];
        return eDec;
    }
    
    double TriangleSoup::computeLocalEDec(const std::pair<int, int>& edge,
        const std::map<std::pair<int, int>, int>& edge2Tri, const std::vector<std::set<int>>& vNeighbor,
        const std::map<std::pair<int, int>, int>& cohEIndex, Eigen::MatrixXd& newVertPos) const
    {
        assert(vNeighbor.size() == V.rows());
        auto edgeTriIndFinder = edge2Tri.find(edge);
        auto edgeTriIndFinder_dual = edge2Tri.find(std::pair<int, int>(edge.second, edge.first));
        assert(edgeTriIndFinder != edge2Tri.end());
        assert(edgeTriIndFinder_dual != edge2Tri.end());
        
        int vI_boundary = edge.first, vI_interior = edge.second;
        bool cutThrough = false;
        if(isBoundaryVert(edge2Tri, vNeighbor, edge.first)) {
            if(isBoundaryVert(edge2Tri, vNeighbor, edge.second)) {
                cutThrough = true;
            }
        }
        else {
            assert(isBoundaryVert(edge2Tri, vNeighbor, edge.second) && "Input edge must attach mesh boundary!");
            
            vI_boundary = edge.second;
            vI_interior = edge.first;
        }
        
        if(cutThrough) {
            newVertPos.resize(4, 2);
        }
        else {
            newVertPos.resize(2, 2);
        }
        
        std::vector<FracCuts::Energy*> energyTerms(1, new SymStretchEnergy());
        std::vector<double> energyParams(1, 1.0);
        double eDec = 0.0;
        for(int toBound = 0; toBound < 2; toBound++) {
            std::set<int> freeVertGID;
            freeVertGID.insert(vI_boundary);
            if(cutThrough) {
                freeVertGID.insert(vI_interior);
            }
            
            std::vector<int> tri_toSep;
            std::pair<int, int> boundaryEdge;
            isBoundaryVert(edge2Tri, vI_boundary, vI_interior, tri_toSep, boundaryEdge, toBound);
            assert(!tri_toSep.empty());
            if(cutThrough) {
                std::vector<int> tri_interior;
                std::pair<int, int> boundaryEdge_interior;
                isBoundaryVert(edge2Tri, vI_interior, vI_boundary, tri_interior, boundaryEdge_interior, !toBound);
                for(const auto& triI : tri_interior) {
                    bool newTri = true;
                    for(const auto& triI_b : tri_toSep) {
                        if(triI_b == triI) {
                            newTri = false;
                            break;
                        }
                    }
                    if(newTri) {
                        tri_toSep.emplace_back(triI);
                    }
                }
            }
            
            std::map<int, Eigen::RowVector2d> newVertPosMap;
            eDec += computeLocalEDec(tri_toSep, freeVertGID, newVertPosMap);
            newVertPos.block(toBound, 0, 1, 2) = newVertPosMap[vI_boundary];
            if(cutThrough) {
                newVertPos.row(2 + toBound) = newVertPosMap[vI_interior];
            }
        }
        delete energyTerms[0];

        return eDec;
    }
    
    void TriangleSoup::splitEdgeOnBoundary(const std::pair<int, int>& edge, const Eigen::MatrixXd& newVertPos,
        std::map<std::pair<int, int>, int>& edge2Tri, std::vector<std::set<int>>& vNeighbor,
        std::map<std::pair<int, int>, int>& cohEIndex, bool changeVertPos)
    {
        assert(vNeighbor.size() == V.rows());
        auto edgeTriIndFinder = edge2Tri.find(edge);
        auto edgeTriIndFinder_dual = edge2Tri.find(std::pair<int, int>(edge.second, edge.first));
        assert(edgeTriIndFinder != edge2Tri.end());
        assert(edgeTriIndFinder_dual != edge2Tri.end());
        
        bool duplicateBoth = false;
        int vI_boundary = edge.first, vI_interior = edge.second;
        if(isBoundaryVert(edge2Tri, vNeighbor, edge.first)) {
            if(isBoundaryVert(edge2Tri, vNeighbor, edge.second)) {
                assert(newVertPos.rows() == 4);
                duplicateBoth = true;
            }
        }
        else {
            assert(isBoundaryVert(edge2Tri, vNeighbor, edge.second) && "Input edge must attach mesh boundary!");
            
            vI_boundary = edge.second;
            vI_interior = edge.first;
        }
        
        fracTail.erase(vI_boundary);
        if(!duplicateBoth) {
            fracTail.insert(vI_interior);
            curFracTail = vI_interior;
        }
        else {
            curFracTail = -1;
        }
        
        // duplicate vI_boundary
        std::vector<int> tri_toSep[2];
        std::pair<int, int> boundaryEdge[2];
        for(int toBound = 0; toBound < 2; toBound++) {
            isBoundaryVert(edge2Tri, vI_boundary, vI_interior, tri_toSep[1], boundaryEdge[1], toBound);
            assert(!tri_toSep[1].empty());
        }
        if(duplicateBoth) {
            isBoundaryVert(edge2Tri, vI_interior, vI_boundary, tri_toSep[0], boundaryEdge[0], true);
            assert(!tri_toSep[0].empty());
        }
        
        int nV = static_cast<int>(V_rest.rows());
        V_rest.conservativeResize(nV + 1, 3);
        V_rest.row(nV) = V_rest.row(vI_boundary);
        V.conservativeResize(nV + 1, 2);
        if(changeVertPos) {
            V.row(nV) = newVertPos.block(1, 0, 1, 2);
            V.row(vI_boundary) = newVertPos.block(0, 0, 1, 2);
        }
        else {
            V.row(nV) = V.row(vI_boundary);
        }
        
        for(const auto triI : tri_toSep[1]) {
            for(int vI = 0; vI < 3; vI++) {
                if(F(triI, vI) == vI_boundary) {
                    // update triangle vertInd, edge2Tri and vNeighbor
                    int vI_post = F(triI, (vI + 1) % 3);
                    int vI_pre = F(triI, (vI + 2) % 3);
                    
                    F(triI, vI) = nV;
                    
                    edge2Tri.erase(std::pair<int, int>(vI_boundary, vI_post));
                    edge2Tri[std::pair<int, int>(nV, vI_post)] = triI;
                    edge2Tri.erase(std::pair<int, int>(vI_pre, vI_boundary));
                    edge2Tri[std::pair<int, int>(vI_pre, nV)] = triI;
                    
                    vNeighbor[vI_pre].erase(vI_boundary);
                    vNeighbor[vI_pre].insert(nV);
                    vNeighbor[vI_post].erase(vI_boundary);
                    vNeighbor[vI_post].insert(nV);
                    vNeighbor[vI_boundary].erase(vI_pre);
                    vNeighbor[vI_boundary].erase(vI_post);
                    vNeighbor.resize(nV + 1);
                    vNeighbor[nV].insert(vI_pre);
                    vNeighbor[nV].insert(vI_post);
                    
                    break;
                }
            }
        }
        vNeighbor[vI_boundary].insert(vI_interior);
        vNeighbor[vI_interior].insert(vI_boundary);
        
        // add cohesive edge pair and update cohEIndex
        const int nCE = static_cast<int>(cohE.rows());
        cohE.conservativeResize(nCE + 1, 4);
        cohE.row(nCE) << vI_interior, nV, vI_interior, vI_boundary; //!! is it a problem?
        cohEIndex[std::pair<int, int>(vI_interior, nV)] = nCE;
        cohEIndex[std::pair<int, int>(vI_boundary, vI_interior)] = -nCE - 1;
        auto CEIfinder = cohEIndex.find(boundaryEdge[1]);
        if(CEIfinder != cohEIndex.end()) {
            if(CEIfinder->second >= 0) {
                cohE(CEIfinder->second, 0) = nV;
            }
            else {
                cohE(-CEIfinder->second - 1, 3) = nV;
            }
            cohEIndex[std::pair<int, int>(nV, boundaryEdge[1].second)] = CEIfinder->second;
            cohEIndex.erase(CEIfinder);
        }
        
        if(duplicateBoth) {
            int nV = static_cast<int>(V_rest.rows());
//            subOptimizerInfo[1].first.insert(nV);
            V_rest.conservativeResize(nV + 1, 3);
            V_rest.row(nV) = V_rest.row(vI_interior);
            V.conservativeResize(nV + 1, 2);
            if(changeVertPos) {
                V.row(nV) = newVertPos.block(2, 0, 1, 2);
                V.row(vI_interior) = newVertPos.block(3, 0, 1, 2);
            }
            else {
                V.row(nV) = V.row(vI_interior);
            }
            
            for(const auto triI : tri_toSep[0]) {
                for(int vI = 0; vI < 3; vI++) {
                    if(F(triI, vI) == vI_interior) {
                        // update triangle vertInd, edge2Tri and vNeighbor
                        int vI_post = F(triI, (vI + 1) % 3);
                        int vI_pre = F(triI, (vI + 2) % 3);
                        
                        F(triI, vI) = nV;
                        
                        edge2Tri.erase(std::pair<int, int>(vI_interior, vI_post));
                        edge2Tri[std::pair<int, int>(nV, vI_post)] = triI;
                        edge2Tri.erase(std::pair<int, int>(vI_pre, vI_interior));
                        edge2Tri[std::pair<int, int>(vI_pre, nV)] = triI;

                        vNeighbor[vI_pre].erase(vI_interior);
                        vNeighbor[vI_pre].insert(nV);
                        vNeighbor[vI_post].erase(vI_interior);
                        vNeighbor[vI_post].insert(nV);
                        vNeighbor[vI_interior].erase(vI_pre);
                        vNeighbor[vI_interior].erase(vI_post);
                        vNeighbor.resize(nV + 1);
                        vNeighbor[nV].insert(vI_pre);
                        vNeighbor[nV].insert(vI_post);
                        
                        break;
                    }
                }
            }
            
            // update cohesive edge pair and update cohEIndex
            cohE(nCE, 2) = nV;
            cohEIndex.erase(std::pair<int, int>(vI_boundary, vI_interior));
            cohEIndex[std::pair<int, int>(vI_boundary, nV)] = -nCE - 1;
            auto CEIfinder = cohEIndex.find(boundaryEdge[0]);
            if(CEIfinder != cohEIndex.end()) {
                if(CEIfinder->second >= 0) {
                    cohE(CEIfinder->second, 0) = nV;
                }
                else {
                    cohE(-CEIfinder->second - 1, 3) = nV;
                }
                cohEIndex[std::pair<int, int>(nV, boundaryEdge[0].second)] = CEIfinder->second;
                cohEIndex.erase(CEIfinder);
            }
        }
    }
    
    void TriangleSoup::mergeBoundaryEdges(const std::pair<int, int>& edge0, const std::pair<int, int>& edge1,
        std::map<std::pair<int, int>, int>& edge2Tri, std::vector<std::set<int>>& vNeighbor,
        std::map<std::pair<int, int>, int>& cohEIndex)
    {
        std::cout << edge0.first << " " << edge0.second << std::endl;
        std::cout << edge1.first << " " << edge1.second << std::endl;
        assert(edge0.second == edge1.first);
        assert(edge2Tri.find(std::pair<int, int>(edge0.second, edge0.first)) == edge2Tri.end());
        assert(edge2Tri.find(std::pair<int, int>(edge1.second, edge1.first)) == edge2Tri.end());
        assert(vNeighbor.size() == V.rows());
        
        V.row(edge0.first) = (V.row(edge0.first) + V.row(edge1.second)) / 2.0;
        int vBackI = static_cast<int>(V.rows()) - 1;
        if(edge1.second < vBackI) {
            V_rest.row(edge1.second) = V_rest.row(vBackI);
            V.row(edge1.second) = V.row(vBackI);
        }
        else {
            assert(edge1.second == vBackI);
        }
        V_rest.conservativeResize(vBackI, 3);
        V.conservativeResize(vBackI, 2);
        
        for(const auto& nbI : vNeighbor[edge1.second]) {
            std::pair<int, int> edgeToFind[2] = {
                std::pair<int, int>(edge1.second, nbI),
                std::pair<int, int>(nbI, edge1.second)
            };
            for(int eI = 0; eI < 2; eI++) {
                auto edgeTri = edge2Tri.find(edgeToFind[eI]);
                if(edgeTri != edge2Tri.end()) {
                    for(int vI = 0; vI < 3; vI++) {
                        if(F(edgeTri->second, vI) == edge1.second) {
                            F(edgeTri->second, vI) = edge0.first;
                            break;
                        }
                    }
                }
            }
        }

        if(edge1.second < vBackI) {
            for(int triI = 0; triI < F.rows(); triI++) {
                for(int vI = 0; vI < 3; vI++) {
                    if(F(triI, vI) == vBackI) {
                        F(triI, vI) = edge1.second;
                    }
                }
            }
//            // not valid because vNeighbor is not updated
//            for(const auto& nbI : vNeighbor[vBackI]) {
//                std::pair<int, int> edgeToFind[2] = {
//                    std::pair<int, int>(vBackI, nbI),
//                    std::pair<int, int>(nbI, vBackI)
//                };
//                for(int eI = 0; eI < 2; eI++) {
//                    auto edgeTri = edge2Tri.find(edgeToFind[eI]);
//                    if(edgeTri != edge2Tri.end()) {
//                        for(int vI = 0; vI < 3; vI++) {
//                            if(F(edgeTri->second, vI) == vBackI) {
//                                F(edgeTri->second, vI) = edge1.second;
//                                break;
//                            }
//                        }
//                    }
//                }
//            }
        }
        
        auto cohEFinder = cohEIndex.find(edge0);
        assert(cohEFinder != cohEIndex.end());
        int cohEBackI = static_cast<int>(cohE.rows()) - 1;
        if(cohEFinder->second >= 0) {
            if(cohEFinder->second < cohEBackI) {
                cohE.row(cohEFinder->second) = cohE.row(cohEBackI);
            }
            else {
                assert(cohEFinder->second == cohEBackI);
            }
        }
        else {
            if(-cohEFinder->second - 1 < cohEBackI) {
                cohE.row(-cohEFinder->second - 1) = cohE.row(cohEBackI);
            }
            else {
                assert(-cohEFinder->second - 1 == cohEBackI);
            }
        }
        cohE.conservativeResize(cohEBackI, 4);
        
        for(int cohI = 0; cohI < cohE.rows(); cohI++) {
            for(int pI = 0; pI < 4; pI++) {
                if(cohE(cohI, pI) == edge1.second) {
                    cohE(cohI, pI) = edge0.first;
                }
            }
        }
        if(edge1.second < vBackI) {
            for(int cohI = 0; cohI < cohE.rows(); cohI++) {
                for(int pI = 0; pI < 4; pI++) {
                    if(cohE(cohI, pI) == vBackI) {
                        cohE(cohI, pI) = edge1.second;
                    }
                }
            }
        }
        
        //TODO: locally update edge2Tri, vNeighbor, cohEIndex
    }
    
}
