//
//  Scaffold.cpp
//  OptCuts
//
//  Created by Minchen Li on 4/5/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#include "Scaffold.hpp"
#include "IglUtils.hpp"
#include "SymDirichletEnergy.hpp"
#include "Timer.hpp"

#include <igl/triangle/triangulate.h>
#include <igl/boundary_loop.h>
#include <igl/avg_edge_length.h>
#include <igl/components.h>

#include <tbb/tbb.h>

extern Timer timer;

namespace OptCuts {
    Scaffold::Scaffold(void)
    {
    }
    
    Scaffold::Scaffold(const TriMesh& mesh, Eigen::MatrixXd UV_bnds, Eigen::MatrixXi E, const Eigen::VectorXi& p_bnd)
    {
        assert(E.rows() == UV_bnds.rows());

        Eigen::MatrixXd H;
        bool fixAMBoundary = false;
        bool countTime = false;
        double edgeLen_eps = mesh.avgEdgeLen * 0.5; //NOTE: different from what's used in [Jiang et al. 2017]
        if(E.rows() == 0) {
            countTime = true;
            timer.start(2);
            std::vector<std::vector<int>> bnd_all;
            igl::boundary_loop(mesh.F, bnd_all);
            assert(bnd_all.size());
            
            int curBndVAmt = 0;
            for(int bndI = 0; bndI < bnd_all.size(); bndI++) {
                UV_bnds.conservativeResize(curBndVAmt + bnd_all[bndI].size(), 2);
                E.conservativeResize(curBndVAmt + bnd_all[bndI].size(), 2);
                bnd.conservativeResize(curBndVAmt + bnd_all[bndI].size());
                for(int bndVI = 0; bndVI + 1 < bnd_all[bndI].size(); bndVI++) {
                    E.row(curBndVAmt + bndVI) << curBndVAmt + bndVI, curBndVAmt + bndVI + 1;
                    UV_bnds.row(curBndVAmt + bndVI) = mesh.V.row(bnd_all[bndI][bndVI]);
                    bnd[curBndVAmt + bndVI] = bnd_all[bndI][bndVI];
                    meshVI2AirMesh[bnd[curBndVAmt + bndVI]] = curBndVAmt + bndVI;
                }
                E.row(curBndVAmt + bnd_all[bndI].size() - 1) << curBndVAmt + bnd_all[bndI].size() - 1, curBndVAmt;
                UV_bnds.row(curBndVAmt + bnd_all[bndI].size() - 1) = mesh.V.row(bnd_all[bndI].back());
                bnd[curBndVAmt + bnd_all[bndI].size() - 1] = bnd_all[bndI].back();
                meshVI2AirMesh[bnd[curBndVAmt + bnd_all[bndI].size() - 1]] = curBndVAmt + bnd_all[bndI].size() - 1;
                curBndVAmt = E.rows();
            }
            
            const double scaleFactor = 3.0; // >=2.0 is recommanded
            const int bandAmt = 2; // >= 2 is recommanded
            const double segLen = mesh.avgEdgeLen * std::pow(scaleFactor, bandAmt);
            const double margin = (mesh.avgEdgeLen - segLen) / (1.0 - scaleFactor); //TODO: use boundary edge lengths of current UV map?
            double minX = mesh.V.col(0).minCoeff() - margin;
            double maxX = mesh.V.col(0).maxCoeff() + margin;
            double minY = mesh.V.col(1).minCoeff() - margin;
            double maxY = mesh.V.col(1).maxCoeff() + margin;
            // segment the bounding box:
            int segAmtX = static_cast<int>((maxX - minX) / segLen);
            int segAmtY = static_cast<int>((maxY - minY) / segLen);
            E.conservativeResize(E.rows() + (segAmtX + segAmtY) * 2, 2);
            for(int segI = bnd.size(); segI + 1 < E.rows(); segI++) {
                E.row(segI) << segI, segI + 1;
            }
            E.row(E.rows() - 1) << E.rows() - 1, bnd.size();
            double stepX = (maxX - minX) / segAmtX;
            double stepY = (maxY - minY) / segAmtY;
            UV_bnds.conservativeResize(UV_bnds.rows() + (segAmtX + segAmtY) * 2, 2);
            for(int segI = 0; segI < segAmtX; segI++) {
                UV_bnds.row(bnd.size() + segI) << minX + segI * stepX, minY;
            }
            for(int segI = 0; segI < segAmtY; segI++) {
                UV_bnds.row(bnd.size() + segAmtX + segI) << maxX, minY + segI * stepY;
            }
            for(int segI = 0; segI < segAmtX; segI++) {
                UV_bnds.row(bnd.size() + segAmtX + segAmtY + segI) << maxX - segI * stepX, maxY;
            }
            for(int segI = 0; segI < segAmtY; segI++) {
                UV_bnds.row(bnd.size() + 2 * segAmtX + segAmtY + segI) << minX, maxY - segI * stepY;
            }
            
            // compute connected component of mesh
            Eigen::VectorXi compI_V;
            igl::components(mesh.F, compI_V);
            // mark holes
            std::set<int> processedComp;
            for(int vI = 0; vI < compI_V.size(); vI++) {
                if(processedComp.find(compI_V[vI]) == processedComp.end()) {
                    H.conservativeResize(H.rows() + 1, 2);
                    
                    std::vector<int> incTris;
                    std::pair<int, int> bEdge;
                    if(mesh.isBoundaryVert(vI, *mesh.vNeighbor[vI].begin(), incTris, bEdge, false)) {
                        // push vI a little bit inside mesh and then add into H
                        // get all incident triangles
                        std::vector<int> temp;
                        mesh.isBoundaryVert(vI, *mesh.vNeighbor[vI].begin(), temp, bEdge, true);
                        incTris.insert(incTris.end(), temp.begin(), temp.end());
                        // construct local mesh
                        Eigen::MatrixXi localF;
                        localF.resize(incTris.size(), 3);
                        Eigen::MatrixXd localV_rest, localV;
                        std::map<int, int> globalVI2local;
                        int localTriI = 0;
                        for(const auto triI : incTris) {
                            for(int vI = 0; vI < 3; vI++) {
                                int globalVI = mesh.F(triI, vI);
                                auto localVIFinder = globalVI2local.find(globalVI);
                                if(localVIFinder == globalVI2local.end()) {
                                    int localVI = static_cast<int>(localV_rest.rows());
                                    localV_rest.conservativeResize(localVI + 1, 3);
                                    localV_rest.row(localVI) = mesh.V_rest.row(globalVI);
                                    localV.conservativeResize(localVI + 1, 2);
                                    localV.row(localVI) = mesh.V.row(globalVI);
                                    localF(localTriI, vI) = localVI;
                                    globalVI2local[globalVI] = localVI;
                                }
                                else {
                                    localF(localTriI, vI) = localVIFinder->second;
                                }
                            }
                            localTriI++;
                        }
                        TriMesh localMesh(localV_rest, localF, localV, Eigen::MatrixXi(), false);
                        // compute inward normal
                        Eigen::RowVector2d sepDir_oneV;
                        mesh.compute2DInwardNormal(vI, sepDir_oneV);
                        Eigen::VectorXd sepDir = Eigen::VectorXd::Zero(localMesh.V.rows() * 2);
                        sepDir.block(globalVI2local[vI] * 2, 0, 2, 1) = sepDir_oneV.transpose();
                        double stepSize_sep = 1.0;
                        SymDirichletEnergy SD;
                        SD.initStepSize(localMesh, sepDir, stepSize_sep);
                        H.bottomRows(1) = mesh.V.row(vI) + 0.5 * stepSize_sep * sepDir_oneV;
                    }
                    else {
                        H.bottomRows(1) = mesh.V.row(vI);
                    }
                    
                    processedComp.insert(compI_V[vI]);
                }
            }
        }
        else {
            assert(p_bnd.rows() > 0);
            
            edgeLen_eps *= 0.1;
            fixAMBoundary = true;
            bnd = p_bnd;
            for(int bndI = 0; bndI < bnd.size(); bndI++) {
                UV_bnds.row(bndI) = mesh.V.row(bnd[bndI]);
                meshVI2AirMesh[bnd[bndI]] = bndI;
            }
            
            //[NOTE] this option is for optimization on local stencil
            // where there shouldn't be holes on the one-ring,
            // so no processing for H
        }
        
        igl::triangle::triangulate(UV_bnds, E, H, "qYQ", airMesh.V, airMesh.F);
        // "Y" for no Steiner points on mesh boundary
        // "q" for high quality mesh generation
        // "Q" for quiet mode (no output)
        
        airMesh.V_rest.resize(airMesh.V.rows(), 3);
        airMesh.V_rest << airMesh.V, Eigen::VectorXd::Zero(airMesh.V.rows());
        airMesh.areaThres_AM = std::sqrt(3.0) / 4.0 * edgeLen_eps * edgeLen_eps; // for preventing degenerate air mesh triangles
        airMesh.computeFeatures();
        
        localVI2Global = bnd;
        localVI2Global.conservativeResize(airMesh.V.rows());
        for(int vI = bnd.size(); vI < airMesh.V.rows(); vI++) {
            localVI2Global[vI] = mesh.V.rows() + vI - bnd.size();
        }
        wholeMeshSize = mesh.V.rows() + airMesh.V.rows() - bnd.size();
        
        // mark fixed boundary vertices if any in air mesh
        for(const auto& meshFixedVI : mesh.fixedVert) {
            const auto finder = meshVI2AirMesh.find(meshFixedVI);
            if(finder != meshVI2AirMesh.end()) {
                airMesh.fixedVert.insert(finder->second);
            }
        }
        
        if(fixAMBoundary) {
            // fix air mesh boundary vertices
            for(int vI = bnd.size(); vI < UV_bnds.rows(); vI++) {
                airMesh.fixedVert.insert(vI);
            }
        }
        
        if(countTime) {
            timer.stop();
        }
        
        // add bijectivity to optimization on local stencil: interior split?
        // filter out operations that will cause overlap initially
        // fix bounding box?
    }
    
    void Scaffold::augmentGradient(Eigen::VectorXd& gradient, const Eigen::VectorXd& gradient_scaf, double w_scaf) const
    {
        assert(gradient.size() / 2 + airMesh.V.rows() - bnd.size() == wholeMeshSize);
        assert(w_scaf >= 0.0);
        
        int systemSize0 = gradient.size();
        gradient.conservativeResize(wholeMeshSize * 2);
        if(w_scaf == 0.0) {
            // for line search
            gradient.bottomRows(gradient.size() - systemSize0) = Eigen::VectorXd::Zero(gradient.size() - systemSize0);
        }
        else {
            for(int vI = 0; vI < bnd.size(); vI++) {
                gradient[bnd[vI] * 2] += w_scaf * gradient_scaf[vI * 2];
                gradient[bnd[vI] * 2 + 1] += w_scaf * gradient_scaf[vI * 2 + 1];
            }
            gradient.bottomRows(gradient.size() - systemSize0) =
                w_scaf * gradient_scaf.bottomRows(gradient_scaf.size() - bnd.size() * 2);
        }
    }
    
    void Scaffold::augmentProxyMatrix(Eigen::VectorXi& I, Eigen::VectorXi& J, Eigen::VectorXd& V,
                                      const Eigen::VectorXi& I_scaf, const Eigen::VectorXi& J_scaf,
                                      const Eigen::VectorXd& V_scaf, double w_scaf) const
    {
        assert(w_scaf > 0.0);
        
        int tupleSize0 = I.size();
        
        V.conservativeResize(tupleSize0 + V_scaf.size());
        V.bottomRows(V_scaf.size()) = w_scaf * V_scaf;
        
        I.conservativeResize(tupleSize0 + I_scaf.size());
        J.conservativeResize(tupleSize0 + J_scaf.size());
        for(int tupleI = 0; tupleI < I_scaf.size(); tupleI++) {
            I[tupleI + tupleSize0] = localVI2Global[I_scaf[tupleI] / 2] * 2 + I_scaf[tupleI] % 2;
            J[tupleI + tupleSize0] = localVI2Global[J_scaf[tupleI] / 2] * 2 + J_scaf[tupleI] % 2;
        }
    }
    void Scaffold::augmentProxyMatrix(Eigen::MatrixXd& P,
                                      const Eigen::MatrixXd& P_scaf,
                                      double w_scaf) const
    {
        assert(w_scaf > 0.0);
        
        int P_oldRows = static_cast<int>(P.rows());
        P.conservativeResize(wholeMeshSize * 2, wholeMeshSize * 2);
        P.rightCols(P.rows() - P_oldRows).setZero();
        P.block(P_oldRows, 0, P.rows() - P_oldRows, P_oldRows).setZero();
        for(int localVI = 0; localVI < localVI2Global.size(); localVI++) {
            int _2globalVI = localVI2Global[localVI] * 2;
            for(int localVJ = 0; localVJ < localVI2Global.size(); localVJ++) {
                int _2globalVJ = localVI2Global[localVJ] * 2;
                P.block(_2globalVI, _2globalVJ, 2, 2) +=
                    w_scaf * P_scaf.block(localVI * 2, localVJ * 2, 2, 2);
            }
        }
    }
    
    void Scaffold::wholeSearchDir2airMesh(const Eigen::VectorXd& searchDir, Eigen::VectorXd& searchDir_airMesh) const
    {
        assert(searchDir.size() / 2 == wholeMeshSize);
        
        searchDir_airMesh.resize(airMesh.V.rows() * 2);
        for(int vI = 0; vI < bnd.size(); vI++) {
            searchDir_airMesh[vI * 2] = searchDir[bnd[vI] * 2];
            searchDir_airMesh[vI * 2 + 1] = searchDir[bnd[vI] * 2 + 1];
        }
        int airSystemSize = (airMesh.V.rows() - bnd.size()) * 2;
        searchDir_airMesh.bottomRows(airSystemSize) = searchDir.bottomRows(airSystemSize);
    }
    
    void Scaffold::stepForward(const Eigen::MatrixXd& V0, const Eigen::VectorXd& searchDir, double stepSize)
    {
        assert(searchDir.size() / 2 == wholeMeshSize);
        assert(V0.rows() == airMesh.V.rows());

        Eigen::VectorXd searchDir_airMesh;
        wholeSearchDir2airMesh(searchDir, searchDir_airMesh);
        for(int vI = 0; vI < airMesh.V.rows(); vI++) {
            airMesh.V(vI, 0) = V0(vI, 0) + stepSize * searchDir_airMesh[vI * 2];
            airMesh.V(vI, 1) = V0(vI, 1) + stepSize * searchDir_airMesh[vI * 2 + 1];
        }
    }
    
    void Scaffold::mergeVNeighbor(const std::vector<std::set<int>>& vNeighbor_mesh, std::vector<std::set<int>>& vNeighbor) const
    {
        vNeighbor = vNeighbor_mesh;
        vNeighbor.resize(wholeMeshSize);
        tbb::parallel_for(0, (int)airMesh.vNeighbor.size(), 1, [&](int scafVI) {
            auto& neighbors = vNeighbor[localVI2Global[scafVI]];
            for(const auto& nb_scafVI : airMesh.vNeighbor[scafVI]) {
                neighbors.insert(localVI2Global[nb_scafVI]);
            }
        });
    }
    
    void Scaffold::mergeFixedV(const std::set<int>& fixedV_mesh, std::set<int>& fixedV) const
    {
        fixedV = fixedV_mesh;
        for(const auto& fixedV_scafVI : airMesh.fixedVert) {
            fixedV.insert(localVI2Global[fixedV_scafVI]);
        }
    }
    
    void Scaffold::augmentUVwithAirMesh(Eigen::MatrixXd& UV, double scale) const
    {
        int meshSize = UV.rows();
        UV.conservativeResize(wholeMeshSize, 2);
        UV.bottomRows(wholeMeshSize - meshSize) = scale * airMesh.V.bottomRows(airMesh.V.rows() - bnd.size());
    }
    
    void Scaffold::augmentFwithAirMesh(Eigen::MatrixXi& F) const
    {
        int meshFAmt = F.rows();
        F.conservativeResize(meshFAmt + airMesh.F.rows(), 3);
        for(int fI = meshFAmt; fI < F.rows(); fI++) {
            for(int vI = 0; vI < 3; vI++) {
                F(fI, vI) = localVI2Global[airMesh.F(fI - meshFAmt, vI)];
            }
        }
    }
    
    void Scaffold::augmentFColorwithAirMesh(Eigen::MatrixXd& FColor) const
    {
        FColor.conservativeResize(FColor.rows() + airMesh.F.rows(), 3);
        FColor.bottomRows(airMesh.F.rows()) = Eigen::MatrixXd::Ones(airMesh.F.rows(), 3);
    }
    
    void Scaffold::get1RingAirLoop(int vI,
                                   Eigen::MatrixXd& UV, Eigen::MatrixXi& E, Eigen::VectorXi& bnd,
                                   std::set<int>& loop_AMVI) const
    {
        const auto finder = meshVI2AirMesh.find(vI);
        assert(finder != meshVI2AirMesh.end());
        assert(airMesh.isBoundaryVert(finder->second));
        
        std::vector<int> umbrella0, umbrella;
        std::pair<int, int> boundaryEdge;
        int nbV0I = *airMesh.vNeighbor[finder->second].begin();
        airMesh.isBoundaryVert(finder->second, nbV0I, umbrella0, boundaryEdge, false);
        airMesh.isBoundaryVert(finder->second, nbV0I, umbrella, boundaryEdge, true);
        if(!umbrella.empty()) {
            std::reverse(umbrella.begin(), umbrella.end());
        }
        if(!umbrella0.empty()) {
            umbrella.insert(umbrella.end(), umbrella0.begin(), umbrella0.end());
        }
        assert(!umbrella.empty());
        
        loop_AMVI.clear();
        UV.resize(umbrella.size() + 2, 2);
        E.resize(umbrella.size() + 2, 2);
        bnd.resize(3);
        for(int tI = 0; tI < umbrella.size(); tI++) {
            int triI = umbrella[tI];
            for(int vI = 0; vI < 3; vI++) {
                if(airMesh.F(triI, vI) == finder->second) {
                    if(tI == 0) {
                        UV.row(1) = airMesh.V.row(finder->second);
                        loop_AMVI.insert(finder->second);
                        E.row(0) << 0, 1;
                        bnd[1] = this->bnd[finder->second];
                        bnd[2] = this->bnd[airMesh.F(triI, (vI + 1) % 3)];
                    }
                    UV.row(tI + 2) = airMesh.V.row(airMesh.F(triI, (vI + 1) % 3));
                    loop_AMVI.insert(airMesh.F(triI, (vI + 1) % 3));
                    E.row(tI + 1) << tI + 1, tI + 2;
                    if(tI + 1 == umbrella.size()) {
                        UV.row(0) = airMesh.V.row(airMesh.F(triI, (vI + 2) % 3));
                        loop_AMVI.insert(airMesh.F(triI, (vI + 2) % 3));
                        E.row(tI + 2) << tI + 2, 0;
                        bnd[0] = this->bnd[airMesh.F(triI, (vI + 2) % 3)];
                    }
                    break;
                }
            }
        }
    }
    
    bool Scaffold::getCornerAirLoop(const std::vector<int>& corner_mesh, const Eigen::RowVector2d& mergedPos,
                                    Eigen::MatrixXd& UV, Eigen::MatrixXi& E, Eigen::VectorXi& bnd) const
    {
        assert(corner_mesh.size() == 3);
        
        // convert mesh vertex index to air mesh vertex index
        std::vector<int> corner(3);
        for(int i = 0; i < 3; i++) {
            const auto finder = meshVI2AirMesh.find(corner_mesh[i]);
            assert(finder != meshVI2AirMesh.end());
            corner[i] = finder->second;
        }
        
        // get incident triangles
        std::set<int> incTris;
        std::pair<int, int> boundaryEdge;
        for(int vI = 0; vI < 3; vI++) {
            std::vector<int> incTris_temp[2];
            airMesh.isBoundaryVert(corner[vI], *airMesh.vNeighbor[corner[vI]].begin(), incTris_temp[0], boundaryEdge, false);
            incTris.insert(incTris_temp[0].begin(), incTris_temp[0].end());
            
            airMesh.isBoundaryVert(corner[vI], *airMesh.vNeighbor[corner[vI]].begin(), incTris_temp[1], boundaryEdge, true);
            incTris.insert(incTris_temp[1].begin(), incTris_temp[1].end());
            
            assert(incTris_temp[0].size() + incTris_temp[1].size() > 0);
        }
        Eigen::MatrixXi F_inc;
        F_inc.resize(incTris.size(), 3);
        int fI = 0;
        for(const auto& triI : incTris) {
            F_inc.row(fI) = airMesh.F.row(triI);
            fI++;
        }
        
        // compute outer loop and ensure no vertex duplication
        std::vector<int> loop;
        igl::boundary_loop(F_inc, loop);
        std::set<int> testDuplication(loop.begin(), loop.end());
        if(testDuplication.size() != loop.size()) {
            assert(0 && "vertex duplication found in loop!");
        }
        
        // eliminate merged vertices from loop
        int startI = -1;
        for(int vI = 0; vI < loop.size(); vI++) {
            if(loop[vI] == corner[0]) {
                if((loop[(vI - 1 + loop.size()) % loop.size()] == corner[1]) &&
                   (loop[(vI - 2 + loop.size()) % loop.size()] == corner[2]))
                {
                    startI = vI;
                    break;
                }
                else {
                    assert(0 && "corner not found on air mesh boundary loop");
                }
            }
        }
        assert(startI >= 0);
        int delI = (startI - 2 + loop.size()) % loop.size();
        if(delI + 1 == loop.size()) {
            loop.erase(loop.begin() + delI);
            loop.erase(loop.begin());
            startI = 0;
        }
        else {
            loop.erase(loop.begin() + delI);
            loop.erase(loop.begin() + delI);
        }
        if(startI < 3) {
            startI = 0;
        }
        else {
            startI -= 2;
        }
        
        // sort loop
        std::vector<int> loop_merged;
        int beginI = (startI - 1 + loop.size()) % loop.size();
        loop_merged.insert(loop_merged.end(), loop.begin() + beginI, loop.end());
        loop_merged.insert(loop_merged.end(), loop.begin(), loop.begin() + beginI);
        
        // construct air mesh data
        bnd.resize(3);
        for(int i = 0; i < 3; i ++) {
            assert(loop_merged[i] < this->bnd.size());
            bnd[i] = this->bnd[loop_merged[i]];
        }
        
        E.resize(loop_merged.size(), 2);
        UV.resize(loop_merged.size(), 2);
        for(int i = 0; i + 1 < loop_merged.size(); i++) {
            UV.row(i) = airMesh.V.row(loop_merged[i]);
            E.row(i) << i, i + 1;
        }
        E.bottomRows(1) << loop_merged.size() - 1, 0;
        UV.bottomRows(1) = airMesh.V.row(loop_merged.back());
        UV.row(1) = mergedPos;
        
        // check loop boundary intersection
        for(int eJ = 2; eJ + 1 < E.rows(); eJ++) {
            if(IglUtils::Test2DSegmentSegment(UV.row(E(0, 0)), UV.row(E(0, 1)),
                                              UV.row(E(eJ, 0)), UV.row(E(eJ, 1))))
            {
                return false;
            }
        }
        for(int eI = 1; eI + 2 < E.rows(); eI++) {
            for(int eJ = eI + 2; eJ < E.rows(); eJ++) {
                if(IglUtils::Test2DSegmentSegment(UV.row(E(eI, 0)), UV.row(E(eI, 1)),
                                                  UV.row(E(eJ, 0)), UV.row(E(eJ, 1))))
                {
                    return false;
                }
            }
        }
        
        // ensure the loop is not totally inverted
        double rotAngle = 0.0;
        for(int eI = 0; eI + 1 < E.rows(); eI++) {
            rotAngle += IglUtils::computeRotAngle(UV.row(E(eI, 1)) - UV.row(E(eI, 0)),
                                                  UV.row(E(eI + 1, 1)) - UV.row(E(eI + 1, 0)));
        }
        rotAngle += IglUtils::computeRotAngle(UV.row(E(E.rows() - 1, 1)) - UV.row(E(E.rows() - 1, 0)),
                                              UV.row(E(0, 1)) - UV.row(E(0, 0)));
        if(rotAngle > 0.0) {
            return true;
        }
        else {
            return false;
        }
    }
    
}
