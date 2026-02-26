#include "image_subscriber.hpp"

namespace jwko
{
    namespace image_subscriber
    {

        ImageSubscriber::ImageSubscriber(const std::string &instance_name)
            : instance_name_(instance_name), node_(nullptr),
              camera_info_received_(false), camera_info_ready_(false),
              image_ready_(false),
              use_compressed_(false)
        {
            RCLCPP_INFO(rclcpp::get_logger("ImageSubscriber"), "[%s] ImageSubscriber constructor called", instance_name_.c_str());
        }

        // void ImageSubscriber::setConsumers(
        //     std::shared_ptr<jwko::line_detector::LineDetector> *line_detector,
        //     std::shared_ptr<jwko::apriltag_transform::AprilTagTransform> *apriltag_transform)
        // {
        //     line_detector_ = line_detector;
        //     apriltag_transform_ = apriltag_transform;
        //     RCLCPP_INFO(rclcpp::get_logger("ImageSubscriber"), "[%s] Consumers set: %d line_detectors, %d apriltag_transforms",
        //                 instance_name_.c_str(), line_detector ? 1 : 0, apriltag_transform ? 1 : 0);
        //                   line_detector ? 1 : 0,
        //                   apriltag_transform ? 1 : 0);
        // }

        bool ImageSubscriber::configure(rclcpp_lifecycle::LifecycleNode *node)
        {
            node_ = node;
            try
            {
                RCLCPP_INFO(rclcpp::get_logger("ImageSubscriber"), "[%s] Configuring ImageSubscriber...", instance_name_.c_str());
                setupParameters();
                initialize();
                RCLCPP_INFO(rclcpp::get_logger("ImageSubscriber"), "[%s] ImageSubscriber configured successfully", instance_name_.c_str());
                return true;
            }
            catch (const std::exception &e)
            {
                RCLCPP_ERROR(rclcpp::get_logger("ImageSubscriber"), "[%s] Exception during configuration: %s", instance_name_.c_str(), e.what());
                return false;
            }
        }

        bool ImageSubscriber::activate()
        {
            RCLCPP_INFO(rclcpp::get_logger("ImageSubscriber"), "[%s] Activating ImageSubscriber...", instance_name_.c_str());
            RCLCPP_INFO(rclcpp::get_logger("ImageSubscriber"), "[%s] ImageSubscriber activated successfully", instance_name_.c_str());
            return true;
        }

        bool ImageSubscriber::deactivate()
        {
            RCLCPP_INFO(rclcpp::get_logger("ImageSubscriber"), "[%s] Deactivating ImageSubscriber...", instance_name_.c_str());
            RCLCPP_INFO(rclcpp::get_logger("ImageSubscriber"), "[%s] ImageSubscriber deactivated successfully", instance_name_.c_str());
            return true;
        }

        bool ImageSubscriber::cleanup()
        {
            RCLCPP_INFO(rclcpp::get_logger("ImageSubscriber"), "[%s] Cleaning up ImageSubscriber...", instance_name_.c_str());

            // Reset all components
            image_subscription_.reset();
            compressed_image_subscription_.reset();
            camera_info_subscription_.reset();

            RCLCPP_INFO(rclcpp::get_logger("ImageSubscriber"), "[%s] ImageSubscriber cleaned up successfully", instance_name_.c_str());
            return true;
        }

        bool ImageSubscriber::shutdown()
        {
            RCLCPP_INFO(rclcpp::get_logger("ImageSubscriber"), "[%s] Shutting down ImageSubscriber...", instance_name_.c_str());

            // Reset all components
            image_subscription_.reset();
            compressed_image_subscription_.reset();
            camera_info_subscription_.reset();

            RCLCPP_INFO(rclcpp::get_logger("ImageSubscriber"), "[%s] ImageSubscriber shut down successfully", instance_name_.c_str());
            return true;
        }

        void ImageSubscriber::setupParameters()
        {
            RCLCPP_INFO(rclcpp::get_logger("ImageSubscriber"), "[%s] Setting up parameters...", instance_name_.c_str());

            if (!node_->has_parameter(instance_name_ + "/image_topic"))
            {
                image_topic_ = node_->declare_parameter(instance_name_ + "/image_topic", "camera/front/color/image_raw");
                compressed_image_topic_ = node_->declare_parameter(instance_name_ + "/compressed_image_topic", "camera/front/color/image_raw/compressed");
                camera_info_topic_ = node_->declare_parameter(instance_name_ + "/camera_info_topic", "camera/front/color/camera_info");
            }
            use_compressed_ = node_->get_parameter("use_compressed").as_bool();

            RCLCPP_INFO(rclcpp::get_logger("ImageSubscriber"), "[%s] Image topic: %s", instance_name_.c_str(), image_topic_.c_str());
            RCLCPP_INFO(rclcpp::get_logger("ImageSubscriber"), "[%s] Compressed image topic: %s", instance_name_.c_str(), compressed_image_topic_.c_str());
            RCLCPP_INFO(rclcpp::get_logger("ImageSubscriber"), "[%s] Camera info topic: %s", instance_name_.c_str(), camera_info_topic_.c_str());
            RCLCPP_INFO(rclcpp::get_logger("ImageSubscriber"), "[%s] Use compressed: %s", instance_name_.c_str(), use_compressed_ ? "true" : "false");
        }

        void ImageSubscriber::setupSubscribers()
        {
            if (use_compressed_)
            {
                // Compressed image subscriber
                compressed_image_subscription_ = node_->create_subscription<sensor_msgs::msg::CompressedImage>(
                    compressed_image_topic_, 10,
                    std::bind(&ImageSubscriber::compressedImageCallback, this, std::placeholders::_1));

                RCLCPP_INFO(rclcpp::get_logger("ImageSubscriber"), "[%s] Compressed image subscriber created", instance_name_.c_str());
            }
            else
            {
                // Raw image subscriber
                image_subscription_ = node_->create_subscription<sensor_msgs::msg::Image>(
                    image_topic_, 10,
                    std::bind(&ImageSubscriber::imageCallback, this, std::placeholders::_1));

                RCLCPP_INFO(rclcpp::get_logger("ImageSubscriber"), "[%s] Raw image subscriber created", instance_name_.c_str());
            }

            camera_info_subscription_ = node_->create_subscription<sensor_msgs::msg::CameraInfo>(
                camera_info_topic_, 10,
                std::bind(&ImageSubscriber::cameraInfoCallback, this, std::placeholders::_1));

            RCLCPP_INFO(rclcpp::get_logger("ImageSubscriber"), "[%s] Subscribers created", instance_name_.c_str());
        }

        void ImageSubscriber::unsetSubscribers()
        {
            image_subscription_.reset();
            compressed_image_subscription_.reset();
            camera_info_subscription_.reset();
            RCLCPP_INFO(rclcpp::get_logger("ImageSubscriber"), "[%s] Subscribers unset", instance_name_.c_str());
        }

        bool ImageSubscriber::initialize()
        {
            RCLCPP_INFO(rclcpp::get_logger("ImageSubscriber"), "[%s] Initializing ImageSubscriber...", instance_name_.c_str());

            // Initialize camera info received flag
            camera_info_received_ = false;
            image_ready_ = false;

            RCLCPP_INFO(rclcpp::get_logger("ImageSubscriber"), "[%s] ImageSubscriber initialized successfully", instance_name_.c_str());
            return true;
        }

        cv::Mat ImageSubscriber::decodeCompressedImage(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
        {
            try
            {
                // CompressedImage의 data는 JPEG, PNG 등의 압축된 이미지 데이터
                // cv::imdecode를 사용하여 디코딩
                cv::Mat image = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);

                if (image.empty())
                {
                    RCLCPP_ERROR(rclcpp::get_logger("ImageSubscriber"), "[%s] Failed to decode compressed image", instance_name_.c_str());
                    return cv::Mat();
                }

                RCLCPP_DEBUG(rclcpp::get_logger("ImageSubscriber"), "[%s] Compressed image decoded successfully (%dx%d)",
                             instance_name_.c_str(), image.cols, image.rows);
                return image;
            }
            catch (const std::exception &e)
            {
                RCLCPP_ERROR(rclcpp::get_logger("ImageSubscriber"), "[%s] Exception while decoding compressed image: %s", instance_name_.c_str(), e.what());
                return cv::Mat();
            }
        }

        void ImageSubscriber::compressedImageCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
        {
            try
            {
                // Decode compressed image
                cv::Mat image = decodeCompressedImage(msg);

                if (image.empty())
                {
                    RCLCPP_ERROR(rclcpp::get_logger("ImageSubscriber"), "[%s] Empty image after decoding", instance_name_.c_str());
                    return;
                }
                // 모든 consumer에게 데이터 전달
                // if (line_detector_)
                // {
                //     if (*line_detector_)
                //     {
                //         (*line_detector_)->storeImageData(image, camera_info_);
                //     }
                // }

                // 기존 저장 로직은 유지 (호환성을 위해)
                {
                    std::lock_guard<std::mutex> lock(image_mutex_);
                    latest_image_ = image.clone();
                    RCLCPP_DEBUG(rclcpp::get_logger("ImageSubscriber"), "[%s] Compressed image received and distributed", instance_name_.c_str());
                    image_ready_ = true;
                }

                // Notify waiting threads
                image_cv_.notify_all();
            }
            catch (const std::exception &e)
            {
                RCLCPP_ERROR(rclcpp::get_logger("ImageSubscriber"), "[%s] Exception in compressedImageCallback: %s", instance_name_.c_str(), e.what());
            }
        }

        void ImageSubscriber::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
        {
            try
            {
                // Convert ROS image message to OpenCV Mat
                cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
                cv::Mat image = cv_ptr->image;

                // 모든 consumer에게 데이터 전달
                // if (line_detector_)
                // {
                //     if (*line_detector_)
                //     {
                //         (*line_detector_)->storeImageData(image, camera_info_);
                //     }
                // }

                // 기존 저장 로직은 유지 (호환성을 위해)
                {
                    std::lock_guard<std::mutex> lock(image_mutex_);
                    latest_image_ = image.clone();
                    RCLCPP_DEBUG(rclcpp::get_logger("ImageSubscriber"), "[%s] Image received and distributed", instance_name_.c_str());
                    image_ready_ = true;
                }

                // Notify waiting threads
                image_cv_.notify_all();
            }
            catch (const cv_bridge::Exception &e)
            {
                RCLCPP_ERROR(rclcpp::get_logger("ImageSubscriber"), "[%s] cv_bridge exception: %s", instance_name_.c_str(), e.what());
            }
        }

        void ImageSubscriber::cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
        {
            std::lock_guard<std::mutex> lock(camera_info_mutex_);
            camera_info_ = msg;

            if (!camera_info_received_)
            {
                camera_info_received_ = true;
                RCLCPP_DEBUG(rclcpp::get_logger("ImageSubscriber"), "[%s] Camera info received", instance_name_.c_str());
            }

            camera_info_ready_ = true;
            camera_info_cv_.notify_all();
        }

        cv::Mat ImageSubscriber::getLatestImage()
        {
            std::lock_guard<std::mutex> lock(image_mutex_);
            return latest_image_.clone();
        }

        sensor_msgs::msg::CameraInfo::SharedPtr ImageSubscriber::getCameraInfo()
        {
            std::lock_guard<std::mutex> lock(camera_info_mutex_);
            return camera_info_;
        }

    } // namespace image_subscriber
} // namespace jwko
