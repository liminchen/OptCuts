//
//  TriMesh.hpp
//  OptCuts
//
//  Created by Minchen Li on 8/30/17.
//  Copyright © 2017 Minchen Li. All rights reserved.
//

#ifndef TriMesh_hpp
#define TriMesh_hpp

#include <Eigen/Eigen>

#include <set>
#include <array>

namespace OptCuts{
    
    enum Primitive
    {
        P_SQUARE,
        P_CYLINDER
    };
    class Scaffold;
    
    // duplicate the vertices and edges of a mesh to separate its triangles,
    // adjacent triangles in the original mesh will have a cohesive edge structure to
    // indicate the connectivity
    class TriMesh{
    public: // owned data
        Eigen::MatrixXd V_rest; // duplicated rest vertex coordinates in 3D
        Eigen::MatrixXd V; // duplicated vertex coordinates, the dimension depends on the search space
        Eigen::MatrixXi F; // reordered triangle draw list (0, 1, 2, ...), indices based on V
        Eigen::MatrixXi cohE; // cohesive edge pairs with the 4 end vertex indices based on V
        Eigen::MatrixXi initSeams; // initial cohesive edge pairs actually
        
    public:
        const Scaffold* scaffold = NULL;
        double areaThres_AM; // for preventing degeneracy of air mesh triangles
        
    public: // owned features
        Eigen::VectorXi boundaryEdge; // 1: boundary edge, 0: interior edge
        Eigen::VectorXd edgeLen; // cohesive edge rest length, used as weights
        Eigen::SparseMatrix<double> LaplacianMtr; // 2 * V.rows() wide
        Eigen::VectorXd triArea; // triangle rest area
        Eigen::MatrixXd triNormal;
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
        Eigen::VectorXd vertWeight; // for regional seam placement
        
        // indices for fast access
        std::map<std::pair<int, int>, int> edge2Tri;
        std::vector<std::set<int>> vNeighbor;
        std::map<std::pair<int, int>, int> cohEIndex;
        
        std::set<int> fracTail;
        int curFracTail;
        std::pair<int, int> curInteriorFracTails;
        double initSeamLen;
        
    public: // constructor
        // default constructor that doesn't do anything
        TriMesh(void);
        
        // initialize from a triangle mesh, V will be constructed from UV_mesh in 2D,
        // V_mesh will be used to initialize restShape
        TriMesh(const Eigen::MatrixXd& V_mesh, const Eigen::MatrixXi& F_mesh,
                     const Eigen::MatrixXd& UV_mesh, const Eigen::MatrixXi& FUV_mesh = Eigen::MatrixXi(),
                     bool separateTri = true, double p_initSeamLen = 0.0, double p_areaThres_AM = 0.0);
        
        TriMesh(Primitive primitive, double size = 1.0, double spacing = 0.1, bool separateTri = true);
        
    public: // API
        void computeFeatures(bool multiComp = false, bool resetFixedV = false);
        void updateFeatures(void);
        void resetFixedVert(const std::set<int>& p_fixedVert);
        void buildCohEfromRecord(const Eigen::MatrixXi& cohERecord);
        
        void querySplit(double lambda_t, bool propagate, bool splitInterior,
                        double& EwDec_max, std::vector<int>& path_max, Eigen::MatrixXd& newVertPos_max,
                        std::pair<double, double>& energyChanges_max) const;
        bool splitEdge(double lambda_t, double EDecThres = 0.0, bool propagate = false, bool splitInterior = false);
        void queryMerge(double lambda, bool propagate,
                        double& EwDec_max, std::vector<int>& path_max, Eigen::MatrixXd& newVertPos_max,
                        std::pair<double, double>& energyChanges_max);
        bool mergeEdge(double lambda, double EDecThres, bool propagate);
        bool splitOrMerge(double lambda_t, double EDecThres, bool propagate, bool splitInterior,
                          bool& isMerge);
        
        void onePointCut(int vI = 0);
        void highCurvOnePointCut(void);
        void farthestPointCut(void);
        void geomImgCut(TriMesh& data_findExtrema);
        void cutPath(std::vector<int> path, bool makeCoh = false, int changePos = 0,
                     const Eigen::MatrixXd& newVertPos = Eigen::MatrixXd(), bool allowCutThrough = true);
        
        void computeSeamScore(Eigen::VectorXd& seamScore) const;
        void computeBoundaryLen(double& boundaryLen) const;
        void computeSeamSparsity(double& sparsity, bool triSoup = false) const;
        void computeStandardStretch(double& stretch_l2, double& stretch_inf, double& stretch_shear, double& compress_inf) const;
        void computeL2StretchPerElem(Eigen::VectorXd& L2StretchPerElem) const;
        void outputStandardStretch(std::ofstream& file) const;
        void computeAbsGaussianCurv(double& absGaussianCurv) const;
        
        void initRigidUV(void);
        
        bool checkInversion(int triI, bool mute) const;
        bool checkInversion(bool mute = false, const std::vector<int>& triangles = std::vector<int>()) const;
        
        void save(const std::string& filePath, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                  const Eigen::MatrixXd UV, const Eigen::MatrixXi& FUV = Eigen::MatrixXi()) const;
        void save(const std::string& filePath) const;
        
        void saveAsMesh(const std::string& filePath, bool scaleUV = false) const;
        
        void saveAsMesh(const std::string& filePath,
                        const Eigen::MatrixXi& F0) const;
        
    public: // helper function
        void computeLaplacianMtr(void);
        
        bool findBoundaryEdge(int vI, const std::pair<int, int>& startEdge,
                              std::pair<int, int>& boundaryEdge);
        
        bool insideTri(int triI, const Eigen::RowVector2d& pos) const;
        bool insideUVRegion(const std::vector<int>& triangles, const Eigen::RowVector2d& pos) const;
        
        // toBound = false indicate counter-clockwise
        bool isBoundaryVert(int vI, int vI_neighbor,
                            std::vector<int>& tri_toSep, std::pair<int, int>& boundaryEdge, bool toBound = true) const;
        bool isBoundaryVert(int vI) const;
        
        void compute2DInwardNormal(int vI, Eigen::RowVector2d& normal) const;
        
        void splitEdgeOnBoundary(const std::pair<int, int>& edge, const Eigen::MatrixXd& newVertPos,
                                bool changeVertPos = true, bool allowCutThrough = true);
        void mergeBoundaryEdges(const std::pair<int, int>& edge0, const std::pair<int, int>& edge1,
                                const Eigen::RowVectorXd& mergedPos);
        
        // query vertex candidate for either split or merge
        double computeLocalLDec(int vI, double lambda_t,
                                std::vector<int>& path,
                                Eigen::MatrixXd& newVertPos,
                                std::pair<double, double>& energyChanges,
                                const std::vector<int>& incTris = std::vector<int>(),
                                const Eigen::RowVector2d& initMergedPos = Eigen::RowVector2d()) const;
        // query interior incident edge of a boundary vertex candidate
        double queryLocalEdDec_bSplit(const std::pair<int, int>& edge,
                                      Eigen::MatrixXd& newVertPos) const;
        
        // boundary split
        double computeLocalEdDec_bSplit(const std::vector<int>& triangles,
                                        const std::set<int>& freeVert,
                                        const std::vector<int>& splitPath,
                                        Eigen::MatrixXd& newVertPos,
                                        int maxIter = 100) const;
        // interior split
        double computeLocalEdDec_inSplit(const std::vector<int>& triangles,
                                         const std::set<int>& freeVert,
                                         const std::vector<int>& path,
                                         Eigen::MatrixXd& newVertPos,
                                         int maxIter = 100) const;
        // merge
        double computeLocalEdDec_merge(const std::vector<int>& path,
                                       const std::vector<int>& triangles,
                                       const std::set<int>& freeVert,
                                       std::map<int, Eigen::RowVector2d>& newVertPos,
                                       const std::map<int, int>& mergeVert,
                                       const Eigen::RowVector2d& initMergedPos,
                                       bool closeup = false, int maxIter = 100) const;
    };
    
}

#endif /* TriMesh_hpp */
