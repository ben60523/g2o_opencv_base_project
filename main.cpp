/**
 * BA Example
 * Author: Xiang Gao
 * Date: 2016.3
 * Email: gaoxiang12@mails.tsinghua.edu.cn
 *
*/
# define G2O_USE_VENDORED_CERES 1
 // for std
#include <iostream>
// for opencv 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <boost/concept_check.hpp>
// for g2o
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/types/sba/types_six_dof_expmap.h>


using namespace std;

int findCorrespondingPoints(const cv::Mat& img1, const cv::Mat& img2, vector<cv::Point2f>& points1, vector<cv::Point2f>& points2);

// Camera Intrinsic Parameters
double cx = 314.4027927179677;
double cy = 227.3815178648873;
double fx = 448.1356443828535;
double fy = 448.4100183240756;

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        cout << "Usage: ba_example img1, img2" << endl;
        exit(1);
    }

    cv::Mat img1 = cv::imread(argv[1]);
    cv::Mat img2 = cv::imread(argv[2]);
    cv::imshow("a", img1);
    cv::imshow("b", img2);
    cv::waitKey(0);

    vector<cv::Point2f> pts1, pts2;
    if (findCorrespondingPoints(img1, img2, pts1, pts2) == false)
    {
        cout << "Not Enough Points" << endl;
        return 0;
    }
    cout << "Found " << pts1.size() << " feature points" << endl;
    // Construct graph in g2o
    // Construct optimizer
    g2o::SparseOptimizer    optimizer;
    // 6*3
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;
    // Use Cholmod linear solover
    std::unique_ptr<Block::LinearSolverType> linearSolver(new g2o::LinearSolverCholmod<Block::PoseMatrixType>());
    std::unique_ptr<Block> block_solver(new Block(std::move(linearSolver)));
    // L-M
    g2o::OptimizationAlgorithmLevenberg* algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

    optimizer.setAlgorithm(algorithm);
    optimizer.setVerbose(false);

    // Add vertex in g2o
    // two poses of camera
    for (int i = 0; i < 2; i++)
    {
        g2o::VertexSE3Expmap* v = new g2o::VertexSE3Expmap();
        v->setId(i);
        if (i == 0)
            v->setFixed(true); // Fix first pose
        v->setEstimate(g2o::SE3Quat());
        optimizer.addVertex(v);
    }
    // Add vertex in g2o
    // feature points
    for (size_t i = 0; i < pts1.size(); i++)
    {
        g2o::VertexPointXYZ* v = new g2o::VertexPointXYZ();
        v->setId(2 + i);
        double z = 1;
        double x = (pts1[i].x - cx) * z / fx;
        double y = (pts1[i].y - cy) * z / fy;
        v->setMarginalized(true);
        v->setEstimate(Eigen::Vector3d(x, y, z));
        optimizer.addVertex(v);
    }

    // Camera Matrix
    g2o::CameraParameters* camera = new g2o::CameraParameters(fx, Eigen::Vector2d(cx, cy), 0);
    camera->setId(0);
    optimizer.addParameter(camera);

    // Edge in g2o
    // between each two edges
    vector<g2o::EdgeProjectXYZ2UV*> edges;
    // First image
    for (size_t i = 0; i < pts1.size(); i++)
    {
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex(0, dynamic_cast<g2o::VertexPointXYZ*>   (optimizer.vertex(i + 2)));
        edge->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap*>     (optimizer.vertex(0)));
        edge->setMeasurement(Eigen::Vector2d(pts1[i].x, pts1[i].y));
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setParameterId(0, 0);
        // kernel of the edge
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(edge);
        edges.push_back(edge);
    }
    // Second image
    for (size_t i = 0; i < pts2.size(); i++)
    {
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex(0, dynamic_cast<g2o::VertexPointXYZ*>   (optimizer.vertex(i + 2)));
        edge->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap*>     (optimizer.vertex(1)));
        edge->setMeasurement(Eigen::Vector2d(pts2[i].x, pts2[i].y));
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setParameterId(0, 0);
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(edge);
        edges.push_back(edge);
    }

    cout << "Start optimize" << endl;
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(15);
    cout << "Optimize Done" << endl;

    // Transformation matrix between two frames
    g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(1));
    Eigen::Isometry3d pose = v->estimate();
    cout << "Pose=" << endl << pose.matrix() << endl;

    // The pos of each feature points
    for (size_t i = 0; i < pts1.size(); i++)
    {
        g2o::VertexPointXYZ* v = dynamic_cast<g2o::VertexPointXYZ*> (optimizer.vertex(i + 2));
        cout << "vertex id " << i + 2 << ", pos = ";
        Eigen::Vector3d pos = v->estimate();
        cout << pos(0) << "," << pos(1) << "," << pos(2) << endl;
    }

    int inliers = 0;
    for (auto e : edges)
    {
        e->computeError();
        // chi2 = error*\Omega*error
        if (e->chi2() > 1)
        {
            cout << "error = " << e->chi2() << endl;
        }
        else
        {
            inliers++;
        }
    }

    cout << "inliers in total points: " << inliers << "/" << pts1.size() + pts2.size() << endl;
    optimizer.save("ba.g2o");
    return 0;
}


int findCorrespondingPoints(const cv::Mat& img1, const cv::Mat& img2, vector<cv::Point2f>& points1, vector<cv::Point2f>& points2)
{
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desp1, desp2;
    orb->detectAndCompute(img1, cv::Mat(), kp1, desp1);
    orb->detectAndCompute(img2, cv::Mat(), kp2, desp2);
    cout << "Found " << kp1.size() << " and " << kp2.size() << " feature points" << endl;

    cv::Ptr<cv::DescriptorMatcher>  matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    double knn_match_ratio = 0.8;
    vector< vector<cv::DMatch> > matches_knn;
    matcher->knnMatch(desp1, desp2, matches_knn, 2);
    vector< cv::DMatch > matches;
    for (size_t i = 0; i < matches_knn.size(); i++)
    {
        if (matches_knn[i][0].distance < knn_match_ratio * matches_knn[i][1].distance)
            matches.push_back(matches_knn[i][0]);
    }

    if (matches.size() <= 20)
        return false;

    for (auto m : matches)
    {
        points1.push_back(kp1[m.queryIdx].pt);
        points2.push_back(kp2[m.trainIdx].pt);
    }

    return true;
}

