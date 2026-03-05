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

        std::vector<cv::KeyPoint> ComputeKeyPointsOctTree(
            const cv::Mat &image,
            int nfeatures, float scaleFactor, int nlevels,
            int edgeThreshold, int iniThFAST, int minThFAST);

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

            // ORB detector
            cv::Ptr<cv::ORB> orb_detector_;

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
            int first_level_;
            int wta_k_;
            int score_type_; // 0 = HARRIS_SCORE, 1 = FAST_SCORE
            int patch_size_;
            int fast_threshold_;
        };

    } // namespace orb_feature_detector
} // namespace jwko

#endif // ORB_FEATURE_DETECTOR_HPP_
