//
//  IglUtils.cpp
//  FracCuts
//
//  Created by Minchen Li on 8/30/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#include "IglUtils.hpp"

#include <set>

namespace FracCuts {
    void IglUtils::computeGraphLaplacian(const Eigen::MatrixXi& F, Eigen::SparseMatrix<double>& graphL) {
        // compute vertex adjacency
        int vertAmt = F.maxCoeff() + 1;
        std::vector<std::set<int>> adjacentVertices(vertAmt);
        for(int rowI = 0; rowI < F.rows(); rowI++) {
            adjacentVertices[F(rowI, 0)].insert(F(rowI, 1));
            adjacentVertices[F(rowI, 1)].insert(F(rowI, 0));
            adjacentVertices[F(rowI, 1)].insert(F(rowI, 2));
            adjacentVertices[F(rowI, 2)].insert(F(rowI, 1));
            adjacentVertices[F(rowI, 2)].insert(F(rowI, 0));
            adjacentVertices[F(rowI, 0)].insert(F(rowI, 2));
        }
        
        graphL.resize(vertAmt, vertAmt);
        graphL.setZero();
        graphL.reserve(vertAmt * 7);
        for(int rowI = 0; rowI < vertAmt; rowI++) {
            graphL.insert(rowI, rowI) = -static_cast<double>(adjacentVertices[rowI].size());
            for(const auto& neighborI : adjacentVertices[rowI]) {
                graphL.insert(rowI, neighborI) = 1;
            }
        }
    }
    
    void IglUtils::computeUniformLaplacian(const Eigen::MatrixXi& F, Eigen::SparseMatrix<double>& graphL) {
        int vertAmt = F.maxCoeff() + 1;
        graphL.resize(vertAmt, vertAmt);
        graphL.setZero();
        graphL.reserve(vertAmt * 7);
        for(int rowI = 0; rowI < F.rows(); rowI++) {
            graphL.coeffRef(F(rowI, 0), F(rowI, 1))++;
            graphL.coeffRef(F(rowI, 1), F(rowI, 0))++;
            graphL.coeffRef(F(rowI, 1), F(rowI, 2))++;
            graphL.coeffRef(F(rowI, 2), F(rowI, 1))++;
            graphL.coeffRef(F(rowI, 2), F(rowI, 0))++;
            graphL.coeffRef(F(rowI, 0), F(rowI, 2))++;
            
            graphL.coeffRef(F(rowI, 0), F(rowI, 0)) -= 2;
            graphL.coeffRef(F(rowI, 1), F(rowI, 1)) -= 2;
            graphL.coeffRef(F(rowI, 2), F(rowI, 2)) -= 2;
        }
    }
    
    double getHETan(const std::map<std::pair<int, int>, double>& HETan, int v0, int v1) {
        auto finder = HETan.find(std::pair<int, int>(v0, v1));
        if(finder == HETan.end()) {
            return 0.0;
        }
        else {
            return finder->second;
        }
    }
    
    void IglUtils::computeMVCMtr(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::SparseMatrix<double>& MVCMtr)
    {
        std::map<std::pair<int, int>, double> HETan;
        std::map<std::pair<int, int>, int> thirdPoint;
        std::vector<std::set<int>> vvNeighbor(V.rows());
        for (int triI = 0; triI < F.rows(); triI++)
        {
            int v0I = F(triI, 0);
            int v1I = F(triI, 1);
            int v2I = F(triI, 2);
            
            Eigen::Vector3d e01 = V.row(v1I) - V.row(v0I);
            Eigen::Vector3d e12 = V.row(v2I) - V.row(v1I);
            Eigen::Vector3d e20 = V.row(v0I) - V.row(v2I);
            double dot0102 = -e01.dot(e20);
            double dot1210 = -e12.dot(e01);
            double dot2021 = -e20.dot(e12);
            double cos0102 = dot0102/(e01.norm()*e20.norm());
            double cos1210 = dot1210/(e01.norm()*e12.norm());
            double cos2021 = dot2021/(e12.norm()*e20.norm());
            
            HETan[std::pair<int, int>(v0I, v1I)] = sqrt(1.0 - cos0102*cos0102) / (1.0 + cos0102);
            HETan[std::pair<int, int>(v1I, v2I)] = sqrt(1.0 - cos1210*cos1210) / (1.0 + cos1210);
            HETan[std::pair<int, int>(v2I, v0I)] = sqrt(1.0 - cos2021*cos2021) / (1.0 + cos2021);
            
            thirdPoint[std::pair<int, int>(v0I, v1I)] = v2I;
            thirdPoint[std::pair<int, int>(v1I, v2I)] = v0I;
            thirdPoint[std::pair<int, int>(v2I, v0I)] = v1I;
            
            vvNeighbor[v0I].insert(v1I);
            vvNeighbor[v0I].insert(v2I);
            vvNeighbor[v1I].insert(v0I);
            vvNeighbor[v1I].insert(v2I);
            vvNeighbor[v2I].insert(v0I);
            vvNeighbor[v2I].insert(v1I);
        }
        
        MVCMtr.resize(V.rows(), V.rows());
        MVCMtr.setZero();
        MVCMtr.reserve(V.rows() * 7);
        for(int rowI = 0; rowI < V.rows(); rowI++)
        {
            for(const auto& nbVI : vvNeighbor[rowI]) {
                double weight = getHETan(HETan, rowI, nbVI);
                auto finder = thirdPoint.find(std::pair<int, int>(nbVI, rowI));
                if(finder != thirdPoint.end()) {
                    weight += getHETan(HETan, rowI, finder->second);
                }
                weight /= (V.row(rowI) - V.row(nbVI)).norm();
                
                MVCMtr.coeffRef(rowI, rowI) -= weight;
                MVCMtr.insert(rowI, nbVI) = weight;
                
//                // symmetrized version
//                MVCMtr.coeffRef(rowI, rowI) -= weight;
//                MVCMtr.coeffRef(rowI, nbVI) += weight;
//                MVCMtr.coeffRef(nbVI, nbVI) -= weight;
//                MVCMtr.coeffRef(nbVI, rowI) += weight;
            }
        }
//        writeSparseMatrixToFile("/Users/mincli/Desktop/meshes/mtr", MVCMtr);
    }
    
    void IglUtils::fixedBoundaryParam_MVC(Eigen::SparseMatrix<double> A, const Eigen::VectorXi& bnd,
                                       const Eigen::MatrixXd& bnd_uv, Eigen::MatrixXd& UV_Tutte)
    {
        assert(bnd.size() == bnd_uv.rows());
        assert(bnd.maxCoeff() < A.rows());
        assert(A.rows() == A.cols());
        
        int vN = static_cast<int>(A.rows());
        A.conservativeResize(vN + bnd.size(), vN + bnd.size());
        A.reserve(A.nonZeros() + bnd.size() * 2);
        for(int pcI = 0; pcI < bnd.size(); pcI++) {
            A.insert(vN + pcI, bnd[pcI]) = 1.0;
            A.insert(bnd[pcI], vN + pcI) = 1.0;
        }
        
        Eigen::SparseLU<Eigen::SparseMatrix<double>> spLUSolver;
        spLUSolver.compute(A);
        if(spLUSolver.info() == Eigen::Success) {
            UV_Tutte.resize(A.rows(), 2);
            Eigen::VectorXd rhs;
            rhs.resize(A.rows());
            
            for(int dimI = 0; dimI < 2; dimI++) {
                rhs << Eigen::VectorXd::Zero(vN), bnd_uv.col(dimI);
                UV_Tutte.col(dimI) = spLUSolver.solve(rhs);
                if(spLUSolver.info() != Eigen::Success) {
                    assert("LU back solve failed!");
                }
            }
            
            UV_Tutte.conservativeResize(vN, 2);
        }
        else {
            assert("LU decomposition on MVC matrix (with Langrange Multiplier) failed!");
        }
    }
    
    void IglUtils::mapTriangleTo2D(const Eigen::Vector3d v[3], Eigen::Vector2d u[3])
    {
        const Eigen::Vector3d e[2] = {
            v[1] - v[0], v[2] - v[0]
        };
        u[0] << 0.0, 0.0;
        u[1] << e[0].norm(), 0.0;
        u[2] << e[0].dot(e[1]) / u[1][0], e[0].cross(e[1]).norm() / u[1][0];
    }
    
    void IglUtils::computeDeformationGradient(const Eigen::Vector3d v[3], const Eigen::Vector2d u[3], Eigen::Matrix2d& F)
    {
        Eigen::Vector2d x[3];
        IglUtils::mapTriangleTo2D(v, x);

        const Eigen::Vector2d u01 = u[1] - u[0];
        const Eigen::Vector2d u02 = u[2] - u[0];
        const double u01Len = u01.norm();
    
        Eigen::Matrix2d U; U << u01Len, u01.dot(u02) / u01Len, 0.0, (u01[0] * u02[1] - u01[1] * u02[0]) / u01Len;
        Eigen::Matrix2d V; V << x[1], x[2];
        F = V * U.inverse();
    }
    
    void IglUtils::map_vertices_to_circle(
        const Eigen::MatrixXd& V,
        const Eigen::VectorXi& bnd,
        Eigen::MatrixXd& UV)
    {
        // Get sorted list of boundary vertices
        std::vector<int> interior,map_ij;
        map_ij.resize(V.rows());
        
        std::vector<bool> isOnBnd(V.rows(),false);
        for (int i = 0; i < bnd.size(); i++)
        {
            isOnBnd[bnd[i]] = true;
            map_ij[bnd[i]] = i;
        }
        
        for (int i = 0; i < (int)isOnBnd.size(); i++)
        {
            if (!isOnBnd[i])
            {
                map_ij[i] = static_cast<int>(interior.size());
                interior.push_back(i);
            }
        }
        
        // Map boundary to circle
        std::vector<double> len(bnd.size());
        len[0] = 0.;
        
        for (int i = 1; i < bnd.size(); i++)
        {
            len[i] = len[i-1] + (V.row(bnd[i-1]) - V.row(bnd[i])).norm();
        }
        double total_len = len[len.size()-1] + (V.row(bnd[0]) - V.row(bnd[bnd.size()-1])).norm();
        
        UV.resize(bnd.size(),2);
        const double radius = total_len / 2.0 / M_PI;
        for (int i = 0; i < bnd.size(); i++)
        {
            double frac = len[i] * 2. * M_PI / total_len;
            UV.row(map_ij[bnd[i]]) << radius * cos(frac), radius * sin(frac);
        }
        
    }
    
    void splitRGB(char32_t color, double rgb[3]) {
        rgb[0] = static_cast<int>((0xff0000 & color) >> 16) / 255.0;
        rgb[1] = static_cast<int>((0x00ff00 & color) >> 8) / 255.0;
        rgb[2] = static_cast<int>(0x0000ff & color) / 255.0;
    }
    void getColor(double scalar, double rgb[3], double center, double halfScale)
    {
        static char32_t colorMap[2][100] = {
            // red to blue
            {0xdd0000,0xdc0005,0xdc0009,0xdb000e,0xda0012,0xd90015,0xd80018,0xd8001b,0xd7001e,0xd60021,0xd50023,0xd50026,0xd40028,0xd3002a,0xd2002d,0xd1002f,0xd00031,0xd00033,0xcf0036,0xce0038,0xcd003a,0xcc003c,0xcb003e,0xca0040,0xca0042,0xc90044,0xc80046,0xc70048,0xc6004a,0xc5004c,0xc4004e,0xc30050,0xc20052,0xc10054,0xc00056,0xbf0058,0xbe005a,0xbd005c,0xbc005e,0xbb0060,0xba0062,0xb80064,0xb70066,0xb60068,0xb5006a,0xb4006c,0xb3006e,0xb10070,0xb00072,0xaf0074,0xae0076,0xac0078,0xab007b,0xaa007d,0xa8007f,0xa70081,0xa50083,0xa40085,0xa20087,0xa10089,0x9f008b,0x9e008d,0x9c008f,0x9b0091,0x990093,0x970095,0x950097,0x940099,0x92009b,0x90009d,0x8e009f,0x8c00a1,0x8a00a4,0x8800a6,0x8600a8,0x8300aa,0x8100ac,0x7f00ae,0x7c00b0,0x7a00b2,0x7700b4,0x7400b6,0x7100b9,0x6f00bb,0x6b00bd,0x6800bf,0x6500c1,0x6100c3,0x5d00c5,0x5900c7,0x5500ca,0x5100cc,0x4c00ce,0x4600d0,0x4000d2,0x3900d4,0x3100d6,0x2800d9,0x1a00db,0x0000dd},
            
            // red to green
            {0xdd0000,0xdc0e00,0xdb1800,0xda1f00,0xda2500,0xd92a00,0xd82e00,0xd73200,0xd63600,0xd53a00,0xd43d00,0xd34000,0xd34300,0xd24600,0xd14900,0xd04b00,0xcf4e00,0xce5000,0xcd5300,0xcc5500,0xcb5700,0xca5900,0xc95c00,0xc85e00,0xc76000,0xc66200,0xc56400,0xc46600,0xc36800,0xc26a00,0xc16b00,0xbf6d00,0xbe6f00,0xbd7100,0xbc7300,0xbb7400,0xba7600,0xb97800,0xb77900,0xb67b00,0xb57d00,0xb47e00,0xb28000,0xb18100,0xb08300,0xaf8500,0xad8600,0xac8800,0xab8900,0xa98b00,0xa88c00,0xa68e00,0xa58f00,0xa39100,0xa29200,0xa09300,0x9f9500,0x9d9600,0x9c9800,0x9a9900,0x999a00,0x979c00,0x959d00,0x939f00,0x92a000,0x90a100,0x8ea300,0x8ca400,0x8aa500,0x88a700,0x86a800,0x84a900,0x82ab00,0x80ac00,0x7ead00,0x7cae00,0x79b000,0x77b100,0x74b200,0x72b400,0x6fb500,0x6db600,0x6ab700,0x67b900,0x64ba00,0x61bb00,0x5ebc00,0x5abd00,0x56bf00,0x53c000,0x4fc100,0x4ac200,0x45c400,0x40c500,0x3bc600,0x34c700,0x2dc800,0x24ca00,0x17cb00,0x00cc00}
        };
        
        scalar -= center;
        scalar /= halfScale;
        
        if(scalar >= 1.0) {
            splitRGB(colorMap[0][99], rgb);
        }
        else if(scalar >= 0.0){
            splitRGB(colorMap[0][static_cast<int>(scalar * 100.0)], rgb);
        }
        else if(scalar > -1.0) {
            splitRGB(colorMap[1][static_cast<int>(scalar * -100.0)], rgb);
        }
        else { // scalar <= -1.0
            splitRGB(colorMap[1][99], rgb);
        }
    }
    void IglUtils::mapScalarToColor_bin(const Eigen::VectorXd& scalar, Eigen::MatrixXd& color, double thres)
    {
        assert(thres > 0.0);
        color.resize(scalar.size(), 3);
        for(int elemI = 0; elemI < scalar.size(); elemI++)
        {
            if(scalar[elemI] < 0.0) {
                // boundary edge
                color.row(elemI) = Eigen::RowVector3d(0.0, 0.2, 0.8);
            }
            else {
                const double s = ((scalar[elemI] > thres) ? 1.0 : 0.0);
                color.row(elemI) = Eigen::RowVector3d(1.0 - s, 1.0 - s, 1.0 - s);
            }
        }
    }
    void IglUtils::mapScalarToColor(const Eigen::VectorXd& scalar, Eigen::MatrixXd& color, double lowerBound, double upperBound)
    {
        color.resize(scalar.size(), 3);
        for(int elemI = 0; elemI < scalar.size(); elemI++) {
            double rgb[3];
            getColor(scalar[elemI], rgb, (upperBound + lowerBound) / 2.0, (upperBound - lowerBound) / 2.0);
            color.row(elemI) << rgb[0], rgb[1], rgb[2];
        }
    }
    
    void IglUtils::addBlockToMatrix(Eigen::SparseMatrix<double>& mtr, const Eigen::MatrixXd& block,
                                 const Eigen::VectorXi& index, int dim)
    {
        assert(block.rows() == block.cols());
        assert(index.size() * dim == block.rows());
        assert(mtr.rows() == mtr.cols());
        assert(index.maxCoeff() * dim + dim - 1 < mtr.rows());
        
        for(int indI = 0; indI < index.size(); indI++) {
            if(index[indI] < 0) {
                continue;
            }
            int startIndI = index[indI] * dim;
            int startIndI_block = indI * dim;
            
            for(int indJ = 0; indJ < index.size(); indJ++) {
                if(index[indJ] < 0) {
                    continue;
                }
                int startIndJ = index[indJ] * dim;
                int startIndJ_block = indJ * dim;
                
                for(int dimI = 0; dimI < dim; dimI++) {
                    for(int dimJ = 0; dimJ < dim; dimJ++) {
                        mtr.coeffRef(startIndI + dimI, startIndJ + dimJ) += block(startIndI_block + dimI, startIndJ_block + dimJ);
                    }
                }
            }
        }
    }
    
    void IglUtils::addDiagonalToMatrix(const Eigen::VectorXd& diagonal, const Eigen::VectorXi& index, int dim,
                                    Eigen::VectorXd* V, Eigen::VectorXi* I, Eigen::VectorXi* J)
    {
        assert(index.size() * dim == diagonal.size());
        
        assert(V);
        int tripletInd = static_cast<int>(V->size());
        const int entryAmt = static_cast<int>(diagonal.size());
        V->conservativeResize(tripletInd + entryAmt);
        if(I) {
            assert(J);
            assert(I->size() == tripletInd);
            assert(J->size() == tripletInd);
            I->conservativeResize(tripletInd + entryAmt);
            J->conservativeResize(tripletInd + entryAmt);
        }
        
        for(int indI = 0; indI < index.size(); indI++) {
            if(index[indI] < 0) {
                assert(0 && "currently doesn't support fixed vertices here!");
                continue;
            }
            int startIndI = index[indI] * dim;
            int startIndI_diagonal = indI * dim;
            
            for(int dimI = 0; dimI < dim; dimI++) {
                (*V)[tripletInd] = diagonal(startIndI_diagonal + dimI);
                if(I) {
                    (*I)[tripletInd] = (*J)[tripletInd] = startIndI + dimI;
                }
                tripletInd++;
            }
        }
    }
    
    void IglUtils::addBlockToMatrix(const Eigen::MatrixXd& block, const Eigen::VectorXi& index, int dim,
                                 Eigen::VectorXd* V, Eigen::VectorXi* I, Eigen::VectorXi* J)
    {
        int num_free = 0;
        for(int indI = 0; indI < index.size(); indI++) {
            if(index[indI] >= 0) {
                num_free++;
            }
        }
        if(!num_free) {
            return;
        }
        
        assert(block.rows() == block.cols());
        assert(index.size() * dim == block.rows());
        
        assert(V);
        int tripletInd = static_cast<int>(V->size());
        const int entryAmt = static_cast<int>(dim * dim * num_free * num_free);
        V->conservativeResize(tripletInd + entryAmt);
        if(I) {
            assert(J);
            assert(I->size() == tripletInd);
            assert(J->size() == tripletInd);
            I->conservativeResize(tripletInd + entryAmt);
            J->conservativeResize(tripletInd + entryAmt);
        }
        
        for(int indI = 0; indI < index.size(); indI++) {
            if(index[indI] < 0) {
                continue;
            }
            int startIndI = index[indI] * dim;
            int startIndI_block = indI * dim;
            
            for(int indJ = 0; indJ < index.size(); indJ++) {
                if(index[indJ] < 0) {
                    continue;
                }
                int startIndJ = index[indJ] * dim;
                int startIndJ_block = indJ * dim;
                
                for(int dimI = 0; dimI < dim; dimI++) {
                    for(int dimJ = 0; dimJ < dim; dimJ++) {
                        (*V)[tripletInd] = block(startIndI_block + dimI, startIndJ_block + dimJ);
                        if(I) {
                            (*I)[tripletInd] = startIndI + dimI;
                            (*J)[tripletInd] = startIndJ + dimJ;
                        }
                        tripletInd++;
                    }
                }
            }
        }
        assert(tripletInd == V->size());
    }
    
    void IglUtils::writeSparseMatrixToFile(const std::string& filePath, const Eigen::VectorXi& I, const Eigen::VectorXi& J,
                                        const Eigen::VectorXd& V, bool MATLAB)
    {
        assert(I.size() == J.size());
        assert(V.size() == I.size());
        
        std::ofstream out;
        out.open(filePath);
        if(out.is_open()) {
            if(!MATLAB) {
                out << I.maxCoeff() + 1 << " " << J.maxCoeff() + 1 << " " << I.size() << std::endl;
            }
            for (int k = 0; k < I.size(); k++) {
                out << I[k] + MATLAB << " " << J[k] + MATLAB << " " << V[k] << std::endl;
            }
            out.close();
        }
        else {
            std::cout << "writeSparseMatrixToFile failed! file open error!" << std::endl;
        }

    }
    
    void IglUtils::writeSparseMatrixToFile(const std::string& filePath, const Eigen::SparseMatrix<double>& mtr, bool MATLAB)
    {
        std::ofstream out;
        out.open(filePath);
        if(out.is_open()) {
            if(!MATLAB) {
                out << mtr.rows() << " " << mtr.cols() << " " << mtr.nonZeros() << std::endl;
            }
            for (int k = 0; k < mtr.outerSize(); ++k)
            {
                for (Eigen::SparseMatrix<double>::InnerIterator it(mtr, k); it; ++it)
                {
                    out << it.row() + MATLAB << " " << it.col() + MATLAB << " " << it.value() << std::endl;
                }
            }
            out.close();
        }
        else {
            std::cout << "writeSparseMatrixToFile failed! file open error!" << std::endl;
        }
    }
    
    void IglUtils::loadSparseMatrixFromFile(const std::string& filePath, Eigen::SparseMatrix<double>& mtr)
    {
        std::ifstream in;
        in.open(filePath);
        if(in.is_open()) {
            int rows, cols, nonZeroAmt;
            in >> rows >> cols >> nonZeroAmt;
            mtr.resize(rows, cols);
            std::vector<Eigen::Triplet<double>> IJV;
            IJV.reserve(nonZeroAmt);
            int i, j;
            double v;
            for(int nzI = 0; nzI < nonZeroAmt; nzI++) {
                assert(!in.eof());
                in >> i >> j >> v;
                IJV.emplace_back(Eigen::Triplet<double>(i, j, v));
            }
            in.close();
            mtr.setFromTriplets(IJV.begin(), IJV.end());
        }
        else {
            std::cout << "loadSparseMatrixToFile failed! file open error!" << std::endl;
        }
    }
    
    void IglUtils::sparseMatrixToTriplet(const Eigen::SparseMatrix<double>& mtr,
                                         Eigen::VectorXi& I, Eigen::VectorXi& J, Eigen::VectorXd& V)
    {
        I.resize(mtr.nonZeros());
        J.resize(mtr.nonZeros());
        V.resize(mtr.nonZeros());
        int entryI = 0;
        for (int k = 0; k < mtr.outerSize(); ++k)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(mtr, k); it; ++it)
            {
                I[entryI] = static_cast<int>(it.row());
                J[entryI] = static_cast<int>(it.col());
                V[entryI] = it.value();
                entryI++;
            }
        }
    }
    
    void IglUtils::sparseMatrixToTriplet(const Eigen::SparseMatrix<double>& mtr, Eigen::VectorXd& V)
    {
        V.resize(mtr.nonZeros());
        int entryI = 0;
        for (int k = 0; k < mtr.outerSize(); ++k)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(mtr, k); it; ++it)
            {
                V[entryI++] = it.value();
            }
        }
    }
    
    const std::string IglUtils::rtos(double real)
    {
        std::string str_real = std::to_string(real);
        size_t pointPos = str_real.find_last_of('.');
        if(pointPos == std::string::npos) {
            return str_real;
        }
        else {
            const char* str = str_real.c_str();
            size_t cI = str_real.length() - 1;
            while((cI > pointPos) && (str[cI] == '0')) {
                cI--;
            }
            
            if(cI == pointPos) {
                return str_real.substr(0, pointPos);
            }
            else {
                return str_real.substr(0, cI + 1);
            }
        }
    }
    
    void IglUtils::differentiate_normalize(const Eigen::Vector2d& var, Eigen::Matrix2d& deriv)
    {
        const double x2 = var[0] * var[0];
        const double y2 = var[1] * var[1];
        const double mxy = -var[0] * var[1];
        deriv << y2, mxy,
            mxy, x2;
        deriv /= std::pow(x2 + y2, 1.5);
    }
    
    void IglUtils::differentiate_xxT(const Eigen::Vector2d& var, Eigen::Matrix<Eigen::RowVector2d, 2, 2>& deriv,
                                     double param)
    {
        deriv(0, 0) = param * Eigen::RowVector2d(2 * var[0], 0.0);
        deriv(0, 1) = param * Eigen::RowVector2d(var[1], var[0]);
        deriv(1, 0) = param * Eigen::RowVector2d(var[1], var[0]);
        deriv(1, 1) = param * Eigen::RowVector2d(0.0, 2 * var[1]);
    }
    
    double IglUtils::computeRotAngle(const Eigen::RowVector2d& from, const Eigen::RowVector2d& to)
    {
        double angle = std::acos(std::max(-1.0, std::min(1.0, from.dot(to) / from.norm() / to.norm())));
        return ((from[0] * to[1] - from[1] * to[0] < 0.0) ? -angle : angle);
    }
}
