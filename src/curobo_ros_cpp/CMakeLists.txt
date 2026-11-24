cmake_minimum_required(VERSION 3.22)
project(curobo_ros_cpp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(ament_cmake REQUIRED)
foreach(dependency ament_index_cpp rclcpp rclcpp_action sensor_msgs geometry_msgs trajectory_msgs control_msgs controller_manager_msgs moveit_msgs visualization_msgs tf2_ros tf2_eigen urdf curobo_ros_interfaces std_srvs)
 find_package(${dependency} REQUIRED)
endforeach()
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(fcl REQUIRED)
find_package(CUDAToolkit REQUIRED)
if(DEFINED ENV{CUROBO_SOURCE_DIR})
  set(_curobo_default "$ENV{CUROBO_SOURCE_DIR}")
else()
  set(_curobo_default "$ENV{HOME}/wh/curobo")
endif()
set(CUROBO_SOURCE "${_curobo_default}" CACHE PATH "cuRobo source checkout")
set(CUROBO_NATIVE_BUILD "${CMAKE_CURRENT_SOURCE_DIR}/../../build/native" CACHE PATH "Built native cuRobo runtime")
set(WARP_LIBRARY "${CUROBO_SOURCE}/.venv/lib/python3.10/site-packages/warp/bin/warp.so" CACHE FILEPATH "Native Warp mesh runtime")
add_library(curobo_runtime SHARED IMPORTED)
set_target_properties(curobo_runtime PROPERTIES IMPORTED_LOCATION "${CUROBO_NATIVE_BUILD}/libcurobo_robot_runtime.so" INTERFACE_INCLUDE_DIRECTORIES "${CUROBO_SOURCE}/cpp_runtime/robot/include")
include(CheckCXXSourceCompiles)
include(CMakePushCheckState)
cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_LIBRARIES curobo_runtime CUDA::cudart)
unset(CUROBO_RUNTIME_API_MATCHES CACHE)
check_cxx_source_compiles([=[
  #include <curobo_robot_runtime/motion_planner.h>
  #include <curobo_robot_runtime/mpc.h>
  using curobo::robot::MotionPlanner;
  using curobo::robot::MpcController;
  auto volatile rrt = &MotionPlanner::plan_with_next_rrt_samples;
  auto volatile classify = &MotionPlanner::classify_configurations;
  auto volatile timing = &MotionPlanner::last_trajopt_graph_timing;
  auto volatile retained = &MotionPlanner::retained_rrt_seed;
  auto volatile joint_goal = &MpcController::update_joint_goal;
  int main() { return !rrt || !classify || !timing || !retained || !joint_goal; }
]=] CUROBO_RUNTIME_API_MATCHES)
cmake_pop_check_state()
if(NOT CUROBO_RUNTIME_API_MATCHES)
  message(FATAL_ERROR "The cuRobo fast headers and runtime must provide the RRT retry, configuration classification, TrajOpt diagnostics, and MPC joint-goal APIs. Rebuild the matching fast source with scripts/build.sh (see README.md).")
endif()
add_library(curobo_planning
  src/robot_model.cpp
  src/pointcloud.cpp
  src/trajectory.cpp
  src/native_planner.cpp
  src/native_planner_world.cpp
  src/native_planner_ik.cpp
  src/native_planner_trajopt.cpp
  src/native_planner_contact.cpp
  src/native_planner_mpc.cpp
)
target_include_directories(curobo_planning PUBLIC include ${FCL_INCLUDE_DIRS})
ament_target_dependencies(curobo_planning rclcpp sensor_msgs geometry_msgs trajectory_msgs urdf curobo_ros_interfaces)
target_link_libraries(curobo_planning curobo_runtime CUDA::cudart Eigen3::Eigen yaml-cpp fcl)
target_compile_options(curobo_planning PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(curobo_planning PROPERTIES BUILD_RPATH "${CUROBO_NATIVE_BUILD}" INSTALL_RPATH "$ORIGIN")
install(TARGETS curobo_planning ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include PATTERN "detail" EXCLUDE)
install(FILES "${CUROBO_NATIVE_BUILD}/libcurobo_robot_runtime.so" "${WARP_LIBRARY}" DESTINATION lib)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../artifacts DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/export_artifacts.py scripts/export_geometry.py scripts/export_scene.py DESTINATION lib/${PROJECT_NAME})
add_library(curobo_ros_node src/node.cpp src/manipulation.cpp src/mpc_follow.cpp src/mpc_detour.cpp src/executor.cpp)
ament_target_dependencies(curobo_ros_node ament_index_cpp rclcpp rclcpp_action control_msgs controller_manager_msgs moveit_msgs visualization_msgs tf2_ros tf2_eigen std_srvs)
target_link_libraries(curobo_ros_node curobo_planning)
add_executable(curobo_motion_planning_node src/main.cpp)
ament_target_dependencies(curobo_motion_planning_node ament_index_cpp rclcpp rclcpp_action control_msgs controller_manager_msgs moveit_msgs visualization_msgs tf2_ros tf2_eigen)
target_link_libraries(curobo_motion_planning_node curobo_ros_node)
set_target_properties(curobo_motion_planning_node PROPERTIES BUILD_RPATH "${CUROBO_NATIVE_BUILD}" INSTALL_RPATH "$ORIGIN/..")
install(TARGETS curobo_motion_planning_node DESTINATION lib/${PROJECT_NAME})
option(BUILD_TARGETED_VALIDATION "Build snapshot validation program" OFF)
if(BUILD_TARGETED_VALIDATION)
 add_executable(validate_joint_history tests/validate_joint_history.cpp)
 target_include_directories(validate_joint_history PRIVATE include)
 add_executable(validate_mpc_detour tests/validate_mpc_detour.cpp src/mpc_detour.cpp)
 target_include_directories(validate_mpc_detour PRIVATE include)
 target_link_libraries(validate_mpc_detour Eigen3::Eigen)
 add_executable(validate_executor tests/validate_executor.cpp)
 ament_target_dependencies(validate_executor ament_index_cpp rclcpp rclcpp_action control_msgs controller_manager_msgs)
 target_link_libraries(validate_executor curobo_ros_node)
 add_executable(validate_snapshot tests/validate_snapshot.cpp)
 ament_target_dependencies(validate_snapshot ament_index_cpp rclcpp rclcpp_action control_msgs controller_manager_msgs moveit_msgs visualization_msgs tf2_ros tf2_eigen)
 target_link_libraries(validate_snapshot curobo_ros_node)
endif()
ament_package()
