#ifndef ORB_FEATURE_DETECTOR_HPP_
#define ORB_FEATURE_DETECTOR_HPP_

#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "sensor_msgs/msg/image.hpp"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "image_subscriber.hpp"

namespace jwko
{
    namespace orb_feature_detector
    {

        // -------------------------------------------------------------------------
        // ExtractorNode
        // -------------------------------------------------------------------------
        struct ExtractorNode
        {
            cv::Point2i UL, UR, BL, BR;
            std::vector<cv::KeyPoint> vKeys;
            std::list<ExtractorNode>::iterator lit;
            bool bNoMore{false};

            void DivideNode(ExtractorNode &n1, ExtractorNode &n2,
                            ExtractorNode &n3, ExtractorNode &n4);
        };

        bool compareNodes(const std::pair<int, ExtractorNode *> &e1,
                          const std::pair<int, ExtractorNode *> &e2);

        std::vector<cv::KeyPoint> DistributeOctTree(
            const std::vector<cv::KeyPoint> &vToDistributeKeys,
            const int &minX, const int &maxX,
            const int &minY, const int &maxY,
            const int &N, const int &level);

        // ORB-SLAM3 방식: 키포인트 검출 + 방향 계산 + 디스크립터까지 한 번에
        // descriptors: CV_8UC1, (N x 32)
        void ComputeORBFeatures(
            const cv::Mat &image,
            int nfeatures, float scaleFactor, int nlevels,
            int edgeThreshold, int iniThFAST, int minThFAST,
            std::vector<cv::KeyPoint> &keypoints,
            cv::Mat &descriptors);

        // -------------------------------------------------------------------------
        // OrbFeatureDetector
        // -------------------------------------------------------------------------
        class OrbFeatureDetector
        {
        public:
            explicit OrbFeatureDetector(const std::string &instance_name = "orb_feature_detector");
            virtual ~OrbFeatureDetector() = default;

            // Lifecycle-like methods (같은 패턴 as ImageSubscriber)
            bool configure(rclcpp_lifecycle::LifecycleNode *node,
                           jwko::image_subscriber::ImageSubscriber *img_sub);
            bool activate();
            bool deactivate();
            bool cleanup();
            bool shutdown();

            std::string instance_name_;

        private:
            bool initialize();
            void setupParameters();

            // ORB 검출 및 결과 이미지 퍼블리시 (타이머 콜백)
            void process();

            // Node / ImageSubscriber 포인터 (lifecycle node가 소유)
            rclcpp_lifecycle::LifecycleNode *node_;
            jwko::image_subscriber::ImageSubscriber *img_sub_;

            // Publisher / Timer
            rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::Image>::SharedPtr feature_pub_;
            rclcpp::TimerBase::SharedPtr timer_;

            // Parameters
            int num_features_;
            double timer_hz_;
            std::string publish_topic_;
            std::string frame_id_;

            // ORB tuning parameters
            float scale_factor_;
            int n_levels_;
            int edge_threshold_;
            int fast_threshold_;
            float min_kp_dist_; // 크로스-레벨 최소 키포인트 거리 (픽셀, 0=비활성)
        };

    } // namespace orb_feature_detector
} // namespace jwko

#endif // ORB_FEATURE_DETECTOR_HPP_
