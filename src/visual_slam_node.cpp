#include "visual_slam.hpp"

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);

    try
    {
        auto node = std::make_shared<jwko::visual_slam::VisualSlam>();

        RCLCPP_INFO(node->get_logger(), "visual_slam Node started");

        rclcpp::spin(node->get_node_base_interface());
    }
    catch (const std::exception &e)
    {
        RCLCPP_ERROR(rclcpp::get_logger("visual_slam"), "Exception: %s", e.what());
        rclcpp::shutdown();
        return 1;
    }

    rclcpp::shutdown();
    return 0;
}
