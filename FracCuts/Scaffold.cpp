//
//  Scaffold.cpp
//  FracCuts
//
//  Created by Minchen Li on 4/5/18.
//  Copyright © 2018 Minchen Li. All rights reserved.
//

#include "Scaffold.hpp"

#include <igl/triangle/triangulate.h>
#include <igl/boundary_loop.h>
#include <igl/avg_edge_length.h>
//#include <igl/components.h>

namespace FracCuts {
    Scaffold::Scaffold(void)
    {
    }
    
    Scaffold::Scaffold(const TriangleSoup& mesh, Eigen::MatrixXd UV_bnds, Eigen::MatrixXi E, const Eigen::VectorXi& p_bnd)
    {
        assert(E.rows() == UV_bnds.rows());

        Eigen::MatrixXd H;
        bool fixAMBoundary = false;
        if(E.rows() == 0) {
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
            
            double avgELen = mesh.avgEdgeLen;
            double minX = mesh.V.col(0).minCoeff() - avgELen;
            double maxX = mesh.V.col(0).maxCoeff() + avgELen;
            double minY = mesh.V.col(1).minCoeff() - avgELen;
            double maxY = mesh.V.col(1).maxCoeff() + avgELen;
            // segment the bounding box:
            int segAmtX = static_cast<int>((maxX - minX) / avgELen) / 2;
            int segAmtY = static_cast<int>((maxY - minY) / avgELen) / 2;
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
            
            //TODO: handle multi-component UV map
    //        igl::components(mesh.F, );
            H.resize(1,2);
            for(int vI = 0; vI < mesh.V.rows(); vI++) {
                if(!mesh.isBoundaryVert(vI)) {
                    H.row(0) = mesh.V.row(vI);
                    break;
                }
            }
        }
        else {
            assert(p_bnd.rows() > 0);
            
            fixAMBoundary = true;
            bnd = p_bnd;
            for(int bndI = 0; bndI < bnd.size(); bndI++) {
                UV_bnds.row(bndI) = mesh.V.row(bnd[bndI]);
                meshVI2AirMesh[bnd[bndI]] = bndI;
            }
        }
        
        igl::triangle::triangulate(UV_bnds, E, H, "qYYQ", airMesh.V, airMesh.F); // "Y" for no Steiner points
        airMesh.V_rest.resize(airMesh.V.rows(), 3);
        airMesh.V_rest << airMesh.V, Eigen::VectorXd::Zero(airMesh.V.rows());
        const double edgeLen_eps = mesh.avgEdgeLen * 0.5; // for preventing degenerate air mesh triangles
        airMesh.areaThres_AM = std::sqrt(3.0) / 4.0 * edgeLen_eps * edgeLen_eps; //NOTE: different from what's used in [Jiang et al. 2017]
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
        
        // tune edgeLen_eps and bounding box margin
        // line search only on E_UV? E_w condition only on E_UV? consider energy changes due to remeshing during optimization!
        // add bijectivity to optimization on local stencil! filter out those want to overlap along edge! need to be careful with local query E
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
                            const Eigen::VectorXi& I_scaf, const Eigen::VectorXi& J_scaf, const Eigen::VectorXd& V_scaf,
                            double w_scaf) const
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
    
    void Scaffold::stepForward(const TriangleSoup& airMesh0, const Eigen::VectorXd& searchDir, double stepSize)
    {
        assert(searchDir.size() / 2 == wholeMeshSize);
        assert(airMesh0.V.rows() == airMesh.V.rows());

        Eigen::VectorXd searchDir_airMesh;
        wholeSearchDir2airMesh(searchDir, searchDir_airMesh);
        for(int vI = 0; vI < airMesh.V.rows(); vI++) {
            airMesh.V(vI, 0) = airMesh0.V(vI, 0) + stepSize * searchDir_airMesh[vI * 2];
            airMesh.V(vI, 1) = airMesh0.V(vI, 1) + stepSize * searchDir_airMesh[vI * 2 + 1];
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
    
    void Scaffold::get1RingAirLoop(int vI, const TriangleSoup& mesh,
                                   Eigen::MatrixXd& UV, Eigen::MatrixXi& E, Eigen::VectorXi& bnd,
                                   std::set<int>& loop_meshVI) const
    {
        assert(mesh.V.rows() + airMesh.V.rows() - this->bnd.size() == wholeMeshSize);
        
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
        
        loop_meshVI.clear();
        UV.resize(umbrella.size() + 2, 2);
        E.resize(umbrella.size() + 2, 2);
        bnd.resize(3);
        for(int tI = 0; tI < umbrella.size(); tI++) {
            int triI = umbrella[tI];
            for(int vI = 0; vI < 3; vI++) {
                if(airMesh.F(triI, vI) == finder->second) {
                    if(tI == 0) {
                        UV.row(1) = airMesh.V.row(finder->second);
                        loop_meshVI.insert(finder->second);
                        E.row(0) << 0, 1;
                        bnd[1] = this->bnd[finder->second];
                        bnd[2] = this->bnd[airMesh.F(triI, (vI + 1) % 3)];
                    }
                    UV.row(tI + 2) = airMesh.V.row(airMesh.F(triI, (vI + 1) % 3));
                    loop_meshVI.insert(airMesh.F(triI, (vI + 1) % 3));
                    E.row(tI + 1) << tI + 1, tI + 2;
                    if(tI + 1 == umbrella.size()) {
                        UV.row(0) = airMesh.V.row(airMesh.F(triI, (vI + 2) % 3));
                        loop_meshVI.insert(airMesh.F(triI, (vI + 2) % 3));
                        E.row(tI + 2) << tI + 2, 0;
                        bnd[0] = this->bnd[airMesh.F(triI, (vI + 2) % 3)];
                    }
                    break;
                }
            }
        }
    }
    
}
