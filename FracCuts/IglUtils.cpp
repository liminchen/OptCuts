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
    
    void IglUtils::mapTriangleTo2D(const Eigen::Vector3d v[3], Eigen::Vector2d u[3])
    {
        const Eigen::Vector3d e[2] = {
            v[1] - v[0], v[2] - v[0]
        };
        u[0] << 0.0, 0.0;
        u[1] << e[0].norm(), 0.0;
        u[2] << e[0].dot(e[1]) / u[1][0], e[0].cross(e[1]).norm() / u[1][0];
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
    void IglUtils::mapScalarToColor_bin(const Eigen::VectorXd& scalar, Eigen::MatrixXd& color)
    {
        color.resize(scalar.size(), 3);
        for(int elemI = 0; elemI < scalar.size(); elemI++)
        {
            const double s = ((scalar[elemI] > 1.0e-1) ? 1.0 : 0.0);
            color.row(elemI) = Eigen::RowVector3d(1.0 - s, 1.0 - s, 1.0 - s);
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
        assert(index.minCoeff() >= 0);
        
        for(int indI = 0; indI < index.size(); indI++) {
            int startIndI = index[indI] * dim;
            int startIndI_block = indI * dim;
            
            for(int indJ = 0; indJ < index.size(); indJ++) {
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
}
