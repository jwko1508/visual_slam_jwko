#include "visual_slam.hpp"

namespace jwko
{
    namespace visual_slam
    {

        VisualSlam::VisualSlam()
            : rclcpp_lifecycle::LifecycleNode("visual_slam")
        {
            RCLCPP_INFO(this->get_logger(), "visual_slam starting...");
        }

        // -----------------------------------------------------------------------------
        // Lifecycle Functions
        // -----------------------------------------------------------------------------

        rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
        VisualSlam::on_configure(const rclcpp_lifecycle::State & /*state*/)
        {
            try
            {
                RCLCPP_INFO(this->get_logger(), "Configuring VisualSlam...");

                setParameters();
                setupSubscribers();
                initialize();

                image_subscriber_instances_.clear();

                RCLCPP_INFO(this->get_logger(), "Creating ImageSubscriber instances");

                for (size_t i = 0; i < camera_list_.size(); ++i)
                {
                    const std::string &cam = camera_list_[i];
                    std::string img_sub_name = "image_subscriber_" + cam;
                    image_subscriber_instances_.push_back(
                        std::make_shared<jwko::image_subscriber::ImageSubscriber>(img_sub_name));
                }

                bool image_subscriber_configured = true;
                for (size_t i = 0; i < image_subscriber_instances_.size(); ++i)
                {
                    image_subscriber_configured &= image_subscriber_instances_[i]->configure(this);
                }

                if (!image_subscriber_configured)
                {
                    RCLCPP_ERROR(this->get_logger(), "ImageSubscriber configuration failed");
                    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::FAILURE;
                }

                // OrbFeatureDetector 인스턴스 생성 및 configure
                orb_detector_instances_.clear();
                for (size_t i = 0; i < camera_list_.size(); ++i)
                {
                    std::string orb_name = "orb_feature_detector_" + camera_list_[i];
                    orb_detector_instances_.push_back(
                        std::make_shared<jwko::orb_feature_detector::OrbFeatureDetector>(orb_name));
                }

                bool orb_configured = true;
                for (size_t i = 0; i < orb_detector_instances_.size(); ++i)
                {
                    orb_configured &= orb_detector_instances_[i]->configure(
                        this, image_subscriber_instances_[i].get());
                }

                if (!orb_configured)
                {
                    RCLCPP_ERROR(this->get_logger(), "OrbFeatureDetector configuration failed");
                    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::FAILURE;
                }

                RCLCPP_INFO(this->get_logger(), "VisualSlam configured successfully");
                return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
            }
            catch (const std::exception &e)
            {
                RCLCPP_ERROR(this->get_logger(), "Exception during configuration: %s", e.what());
                return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::FAILURE;
            }
        }

        rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
        VisualSlam::on_activate(const rclcpp_lifecycle::State & /*state*/)
        {
            RCLCPP_INFO(this->get_logger(), "Activating VisualSlam...");

            for (size_t i = 0; i < image_subscriber_instances_.size(); ++i)
            {
                image_subscriber_instances_[i]->activate();
                image_subscriber_instances_[i]->setupSubscribers();
            }

            for (size_t i = 0; i < orb_detector_instances_.size(); ++i)
            {
                orb_detector_instances_[i]->activate();
            }

            RCLCPP_INFO(this->get_logger(), "VisualSlam activated successfully");
            return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
        }

        rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
        VisualSlam::on_deactivate(const rclcpp_lifecycle::State & /*state*/)
        {
            RCLCPP_INFO(this->get_logger(), "Deactivating VisualSlam...");

            for (size_t i = 0; i < orb_detector_instances_.size(); ++i)
            {
                orb_detector_instances_[i]->deactivate();
            }

            for (size_t i = 0; i < image_subscriber_instances_.size(); ++i)
            {
                image_subscriber_instances_[i]->deactivate();
            }

            RCLCPP_INFO(this->get_logger(), "VisualSlam deactivated successfully");
            return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
        }

        rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
        VisualSlam::on_cleanup(const rclcpp_lifecycle::State & /*state*/)
        {
            RCLCPP_INFO(this->get_logger(), "Cleaning up VisualSlam...");

            for (size_t i = 0; i < orb_detector_instances_.size(); ++i)
            {
                orb_detector_instances_[i]->cleanup();
            }
            orb_detector_instances_.clear();

            for (size_t i = 0; i < image_subscriber_instances_.size(); ++i)
            {
                image_subscriber_instances_[i]->cleanup();
            }

            image_subscriber_instances_.clear();

            RCLCPP_INFO(this->get_logger(), "VisualSlam cleaned up successfully");
            return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
        }

        rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
        VisualSlam::on_shutdown(const rclcpp_lifecycle::State & /*state*/)
        {
            RCLCPP_INFO(this->get_logger(), "Shutting down VisualSlam...");

            for (size_t i = 0; i < orb_detector_instances_.size(); ++i)
            {
                orb_detector_instances_[i]->shutdown();
            }

            for (size_t i = 0; i < image_subscriber_instances_.size(); ++i)
            {
                image_subscriber_instances_[i]->shutdown();
            }

            RCLCPP_INFO(this->get_logger(), "VisualSlam shut down successfully");
            return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
        }

        // -----------------------------------------------------------------------------
        // Private helpers
        // -----------------------------------------------------------------------------

        void VisualSlam::setParameters()
        {
            camera_list_ = this->declare_parameter<std::vector<std::string>>(
                "camera_list", std::vector<std::string>{"front"});
            use_compressed_ = this->declare_parameter<bool>("use_compressed", false);
        }

        void VisualSlam::setupSubscribers()
        {
            // 필요한 구독자를 여기에 추가하세요
            RCLCPP_INFO(this->get_logger(), "Subscribers created");
        }

        bool VisualSlam::initialize()
        {
            // 필요한 초기화 작업을 여기에 추가하세요
            return true;
        }

    } // namespace visual_slam
} // namespace jwko
