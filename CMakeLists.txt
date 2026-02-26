cmake_minimum_required(VERSION 3.8)
project(visual_slam_jwko)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Debug mode configuration
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_definitions(DEBUG_MODE)
  message(STATUS "Building in Debug mode - Debug logging enabled")
endif()

# 필요한 패키지 찾기
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
# OpenCV 모듈 찾기
find_package(OpenCV REQUIRED)

# 헤더 파일 포함 디렉토리
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

file(GLOB_RECURSE CPP_SRC
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

add_executable(${PROJECT_NAME}_node ${CPP_SRC})
target_include_directories(${PROJECT_NAME}_node PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# 의존성 지정
ament_target_dependencies(${PROJECT_NAME}_node
  rclcpp
  rclcpp_lifecycle
  rclcpp_components
  sensor_msgs
  visualization_msgs
  geometry_msgs
  std_msgs
  tf2_ros
  tf2_geometry_msgs
  cv_bridge
  OpenCV
)

# AprilTag 라이브러리 링크
target_link_libraries(${PROJECT_NAME}_node
)

# Install executables
install(TARGETS ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME}
)

# 헤더 파일 설치
install(DIRECTORY include/
  DESTINATION include/
)

# launch 파일 설치
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# config 파일 설치
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

# 패키지 빌드 완료
ament_package()