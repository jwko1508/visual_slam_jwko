#ifndef VISUAL_SLAM_HPP_
#define VISUAL_SLAM_HPP_

#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp"
#include "lifecycle_msgs/msg/state.hpp"

#include "image_subscriber.hpp"
#include "orb_feature_detector.hpp"

namespace jwko
{
    namespace visual_slam
    {

        class VisualSlam : public rclcpp_lifecycle::LifecycleNode
        {
        public:
            explicit VisualSlam();
            virtual ~VisualSlam() = default;

        protected:
            // -------------------------------------------------------------------------
            // Lifecycle callbacks
            // -------------------------------------------------------------------------
            rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
            on_configure(const rclcpp_lifecycle::State &state) override;

            rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
            on_activate(const rclcpp_lifecycle::State &state) override;

            rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
            on_deactivate(const rclcpp_lifecycle::State &state) override;

            rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
            on_cleanup(const rclcpp_lifecycle::State &state) override;

            rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
            on_shutdown(const rclcpp_lifecycle::State &state) override;

        private:
            // -------------------------------------------------------------------------
            // Setup helpers
            // -------------------------------------------------------------------------
            void setParameters();
            void setupSubscribers();
            bool initialize();

            // -------------------------------------------------------------------------
            // Member variables
            // -------------------------------------------------------------------------
            std::vector<std::shared_ptr<jwko::image_subscriber::ImageSubscriber>> image_subscriber_instances_;
            std::vector<std::shared_ptr<jwko::orb_feature_detector::OrbFeatureDetector>> orb_detector_instances_;

            // Parameters
            std::vector<std::string> camera_list_;
            bool use_compressed_;
        };

    } // namespace visual_slam
} // namespace jwko

#endif // VISUAL_SLAM_HPP_
