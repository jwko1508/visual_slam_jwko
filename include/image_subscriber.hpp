#ifndef IMAGE_SUBSCRIBER_HPP_
#define IMAGE_SUBSCRIBER_HPP_

#include <memory>
#include <string>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "lifecycle_msgs/msg/state.hpp"

#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include <opencv2/core.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs.hpp>

namespace jwko
{
    namespace image_subscriber
    {
        class ImageSubscriber
        {
        public:
            ImageSubscriber(const std::string &instance_name = "image_subscriber");
            virtual ~ImageSubscriber() = default;

            // Consumer 인스턴스들 설정
            // void setConsumers(
            //     std::shared_ptr<jwko::line_detector::LineDetector> *line_detector,
            //     std::shared_ptr<jwko::apriltag_transform::AprilTagTransform> *apriltag_transform);

            // Lifecycle-like methods as regular member functions
            bool configure(rclcpp_lifecycle::LifecycleNode *node);
            bool activate();
            bool deactivate();
            bool cleanup();
            bool shutdown();

            void setupSubscribers();
            void unsetSubscribers();

            // Getter methods for image data
            cv::Mat getLatestImage();
            cv::Mat getLatestImageUnsafe() { return latest_image_.clone(); } // lock 없이 호출 (외부에서 이미 lock 잡은 경우)
            sensor_msgs::msg::CameraInfo::SharedPtr getCameraInfo();
            sensor_msgs::msg::CameraInfo::SharedPtr getCameraInfoUnsafe() { return camera_info_; } // lock 없이 호출
            bool isCameraInfoReceived() const { return camera_info_received_; }

            // Access to synchronization primitives
            std::mutex &getImageMutex() { return image_mutex_; }
            std::condition_variable &getImageCV() { return image_cv_; }
            bool isImageReady() const { return image_ready_; }
            void resetImageReady() { image_ready_ = false; }

            std::mutex &getCameraInfoMutex() { return camera_info_mutex_; }
            std::condition_variable &getCameraInfoCV() { return camera_info_cv_; }
            bool isCameraInfoReady() const { return camera_info_ready_; }
            void resetCameraInfoReady() { camera_info_ready_ = false; }

            std::string instance_name_;

        protected:
            bool initialize();

        private:
            rclcpp_lifecycle::LifecycleNode *node_;

            // Subscribers
            rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
            rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_image_subscription_;
            rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_subscription_;

            // Parameters
            std::string image_topic_;
            std::string compressed_image_topic_;
            std::string camera_info_topic_;
            bool use_compressed_; // compressed 이미지 사용 여부

            // Camera info
            sensor_msgs::msg::CameraInfo::SharedPtr camera_info_;
            bool camera_info_received_;
            std::mutex camera_info_mutex_;
            std::condition_variable camera_info_cv_;
            bool camera_info_ready_;

            // Image data
            cv::Mat latest_image_;
            std::mutex image_mutex_;
            std::condition_variable image_cv_;
            bool image_ready_;

            // Consumer 인스턴스들 (포인터로 저장)
            // std::shared_ptr<jwko::line_detector::LineDetector> *line_detector_;

            // Setup methods
            void setupParameters();

            // Callback functions
            void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
            void compressedImageCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg);
            void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

            // Helper function to decode compressed image
            cv::Mat decodeCompressedImage(const sensor_msgs::msg::CompressedImage::SharedPtr msg);
        };

    } // namespace image_subscriber
} // namespace jwko

#endif // IMAGE_SUBSCRIBER_HPP_
