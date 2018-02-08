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

#include <set>
#include <array>

namespace FracCuts{
    
    enum Primitive
    {
        P_SQUARE,
        P_CYLINDER
    };
    
    // duplicate the vertices and edges of a mesh to separate its triangles,
    // adjacent triangles in the original mesh will have a cohesive edge structure to
    // indicate the connectivity
    class TriangleSoup{
    public: // owned data
        Eigen::MatrixXd V_rest; // duplicated rest vertex coordinates in 3D
        Eigen::MatrixXd V; // duplicated vertex coordinates, the dimension depends on the search space
        Eigen::MatrixXi F; // reordered triangle draw list (0, 1, 2, ...), indices based on V
        Eigen::MatrixXi cohE; // cohesive edge pairs with the 4 end vertex indices based on V
        Eigen::MatrixXi initSeams; // initial cohesive edge pairs actually
        
    public: // owned features
        Eigen::VectorXi boundaryEdge; // 1: boundary edge, 0: interior edge
        Eigen::VectorXd edgeLen; // cohesive edge rest length, used as weights
        Eigen::SparseMatrix<double> LaplacianMtr; // 2 * V.rows() wide
        Eigen::VectorXd triArea; // triangle rest area
        double surfaceArea;
        Eigen::VectorXd triAreaSq; // triangle rest squared area
        Eigen::VectorXd e0dote1; // triangle rest edge dot product
        Eigen::VectorXd e0SqLen, e1SqLen; // triangle edge rest squared length
        Eigen::VectorXd e0SqLen_div_dbAreaSq;
        Eigen::VectorXd e1SqLen_div_dbAreaSq;
        Eigen::VectorXd e0dote1_div_dbAreaSq;
        double avgEdgeLen;
        double virtualRadius;
        std::vector<std::set<std::pair<int, int>>> validSplit;
        std::set<int> fixedVert; // for linear solve
        Eigen::Matrix<double, 2, 3> bbox;
//        Eigen::MatrixXd cotVals; // cotangent values of rest triangle corners
        
        // indices for fast access
        std::map<std::pair<int, int>, int> edge2Tri;
        std::vector<std::set<int>> vNeighbor;
        std::map<std::pair<int, int>, int> cohEIndex;
        
        std::set<int> fracTail;
        int curFracTail;
        double initSeamLen;
        
    public: // constructor
        // default constructor that doesn't do anything
        TriangleSoup(void);
        
        // initialize from a triangle mesh, V will be constructed from UV_mesh in 2D,
        // V_mesh will be used to initialize restShape
        TriangleSoup(const Eigen::MatrixXd& V_mesh, const Eigen::MatrixXi& F_mesh,
                     const Eigen::MatrixXd& UV_mesh, const Eigen::MatrixXi& FUV_mesh = Eigen::MatrixXi(),
                     bool separateTri = true, double p_initSeamLen = 0.0);
        
        TriangleSoup(Primitive primitive, double size = 1.0, double spacing = 0.1, bool separateTri = true);
        
    public: // API
        void computeFeatures(bool multiComp = false, bool resetFixedV = true);
        void updateFeatures(void);
        void resetFixedVert(const std::set<int>& p_fixedVert);
        
        bool separateTriangle(const Eigen::VectorXd& measure, double thres);
        bool splitVertex(const Eigen::VectorXd& measure, double thres);
        void resetSubOptInfo(void);
        void querySplit(double lambda_t, bool propagate, bool splitInterior,
                        double& EwDec_max, std::vector<int>& path_max, Eigen::MatrixXd& newVertPos_max);
        bool splitEdge(double lambda_t, double EDecThres = 0.0, bool propagate = false, bool splitInterior = false);
        void queryMerge(double lambda,
                        double& EwDec_max, std::vector<int>& path_max, Eigen::MatrixXd& newVertPos_max);
        bool mergeEdge(double lambda, double EDecThres);
        bool splitOrMerge(double lambda_t, double EDecThres, bool propagate, bool splitInterior, bool& isMerge);
        
        void onePointCut(int vI = 0);
        void highCurvOnePointCut(void);
        void farthestPointCut(void);
        void geomImgCut(void);
        void cutPath(std::vector<int> path, bool makeCoh = false, int changePos = 0, const Eigen::MatrixXd& newVertPos = Eigen::MatrixXd());
        
        void computeSeamScore(Eigen::VectorXd& seamScore) const;
        void computeBoundaryLen(double& boundaryLen) const;
        void computeSeamSparsity(double& sparsity, bool triSoup = false) const;
        void computeStandardStretch(double& stretch_l2, double& stretch_inf, double& stretch_shear, double& compress_inf) const;
        void computeL2StretchPerElem(Eigen::VectorXd& L2StretchPerElem) const;
        void outputStandardStretch(std::ofstream& file) const;
        void computeAbsGaussianCurv(double& absGaussianCurv) const;
        
        void initRigidUV(void);
        
        bool checkInversion(bool mute = false) const;
        
        void save(const std::string& filePath, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                  const Eigen::MatrixXd UV, const Eigen::MatrixXi& FUV = Eigen::MatrixXi()) const;
        void save(const std::string& filePath) const;
        
        void saveAsMesh(const std::string& filePath, bool scaleUV = false) const;
        
    public: // helper function
        void computeLaplacianMtr(void);
        
        bool findBoundaryEdge(int vI, const std::pair<int, int>& startEdge,
                              const std::map<std::pair<int, int>, int>& edge2Tri,
                              std::pair<int, int>& boundaryEdge);
        
        // toBound = false indicate counter-clockwise
        bool isBoundaryVert(const std::map<std::pair<int, int>, int>& edge2Tri, int vI, int vI_neighbor,
                            std::vector<int>& tri_toSep, std::pair<int, int>& boundaryEdge, bool toBound = true) const;
        bool isBoundaryVert(const std::map<std::pair<int, int>, int>& edge2Tri, const std::vector<std::set<int>>& vNeighbor, int vI) const;
        
        void splitEdgeOnBoundary(const std::pair<int, int>& edge, const Eigen::MatrixXd& newVertPos,
            std::map<std::pair<int, int>, int>& edge2Tri, std::vector<std::set<int>>& vNeighbor,
            std::map<std::pair<int, int>, int>& cohEIndex, bool changeVertPos = true);
        void mergeBoundaryEdges(const std::pair<int, int>& edge0, const std::pair<int, int>& edge1,
                                const Eigen::RowVectorXd& mergedPos);
        
        // query vertex candidate
        double computeLocalEwDec(int vI, double lambda_t, std::vector<int>& path, Eigen::MatrixXd& newVertPos) const;
        // query incident edge of a vertex candidate
        double computeLocalEDec(const std::pair<int, int>& edge,
            const std::map<std::pair<int, int>, int>& edge2Tri, const std::vector<std::set<int>>& vNeighbor,
            const std::map<std::pair<int, int>, int>& cohEIndex, Eigen::MatrixXd& newVertPos) const; //TODO: write this in a new class
        // minimize SD on the local stencil
        double computeLocalEDec(const std::vector<int>& triangles, const std::set<int>& freeVert,
                                std::map<int, Eigen::RowVector2d>& newVertPos,
                                const std::map<int, int>& mergeVert = std::map<int, int>(), int maxIter = 100) const;
    };
    
}

#endif /* TriangleSoup_hpp */
