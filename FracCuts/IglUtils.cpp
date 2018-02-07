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
    void getColor(double scalar, double rgb[3], double center, double scale)
    {
        static char32_t colorMap[2][100] = {
            // green to red
            {0x008000,0x0f8000,0x198000,0x217f00,0x277f00,0x2c7f00,0x317f00,0x367e00,0x3a7e00,0x3e7e00,0x417e00,0x457d00,0x487d00,0x4b7d00,0x4e7d00,0x517c00,0x547c00,0x577c00,0x5a7b00,0x5d7b00,0x5f7b00,0x627a00,0x657a00,0x677a00,0x6a7900,0x6c7900,0x6f7800,0x717800,0x737700,0x767700,0x787700,0x7a7600,0x7c7600,0x7f7500,0x817500,0x837400,0x857400,0x877300,0x8a7300,0x8c7200,0x8e7200,0x907100,0x927000,0x947000,0x966f00,0x986e00,0x9a6e00,0x9c6d00,0x9e6c00,0xa06c00,0xa26b00,0xa46a00,0xa66a00,0xa86900,0xaa6800,0xac6700,0xae6600,0xb06500,0xb26500,0xb46400,0xb66300,0xb86200,0xba6100,0xbc6000,0xbe5f00,0xc05e00,0xc25d00,0xc35c00,0xc55b00,0xc75900,0xc95800,0xcb5700,0xcd5600,0xcf5400,0xd15300,0xd35200,0xd45000,0xd64f00,0xd84d00,0xda4b00,0xdc4a00,0xde4800,0xe04600,0xe14400,0xe34200,0xe54000,0xe73e00,0xe93c00,0xeb3900,0xed3700,0xee3400,0xf03100,0xf22d00,0xf42a00,0xf62600,0xf82100,0xf91c00,0xfb1600,0xfd0d00,0xff0000},
            
            // green to deepskyblue
            {0x008000,0x048106,0x08810c,0x0c8211,0x0f8215,0x128319,0x14841c,0x168420,0x198523,0x1a8626,0x1c8629,0x1e872b,0x1f872e,0x218831,0x228933,0x248936,0x258a38,0x268a3b,0x278b3d,0x298c40,0x2a8c42,0x2b8d45,0x2c8e47,0x2d8e4a,0x2e8f4c,0x2e8f4e,0x2f9051,0x309153,0x319155,0x329258,0x32935a,0x33935c,0x34945f,0x349461,0x359563,0x369666,0x369668,0x37976a,0x37986d,0x38986f,0x389971,0x399974,0x399a76,0x399b78,0x3a9b7b,0x3a9c7d,0x3a9d7f,0x3b9d82,0x3b9e84,0x3b9f86,0x3b9f89,0x3ba08b,0x3ca08d,0x3ca190,0x3ca292,0x3ca294,0x3ca397,0x3ca499,0x3ca49b,0x3ca59e,0x3ca6a0,0x3ca6a3,0x3ca7a5,0x3ba7a7,0x3ba8aa,0x3ba9ac,0x3ba9ae,0x3aaab1,0x3aabb3,0x3aabb6,0x39acb8,0x39adba,0x38adbd,0x38aebf,0x37afc2,0x37afc4,0x36b0c6,0x35b1c9,0x35b1cb,0x34b2ce,0x33b2d0,0x32b3d3,0x31b4d5,0x30b4d7,0x2eb5da,0x2db6dc,0x2cb6df,0x2ab7e1,0x29b8e4,0x27b8e6,0x25b9e9,0x23baeb,0x21baee,0x1ebbf0,0x1bbcf3,0x18bcf5,0x14bdf8,0x0fbefa,0x08befc,0x00bfff}
        };
        
        scalar -= center;
        scalar /= scale;
        
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
        const double range = upperBound - lowerBound;
        color.resize(scalar.size(), 3);
        for(int elemI = 0; elemI < scalar.size(); elemI++)
        {
//            getColor(scalar[elemI], color.row(elemI).data(), 0.0, upperBound);
            const double s = 0.8 * std::max(0.0, std::min((scalar[elemI] - lowerBound) / range, 1.0));
            color.row(elemI) = Eigen::RowVector3d(s, 0.8 - s, 0.0);
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
}
