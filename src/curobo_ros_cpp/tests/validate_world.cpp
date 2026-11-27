// SPDX-License-Identifier: Apache-2.0
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "curobo_ros_cpp/native_planner.hpp"

namespace {
class Fixture {
 public:
  Fixture() {
    std::string pattern = (std::filesystem::temp_directory_path() / "curobo-world-XXXXXX").string();
    const auto directory = ::mkdtemp(pattern.data());
    if (!directory) throw std::runtime_error("cannot create temporary world fixture");
    directory_ = directory;
  }
  ~Fixture() {
    std::error_code ignored;
    std::filesystem::remove_all(directory_, ignored);
  }
  std::string write(const std::string &name, const char *text) const {
    const auto path = directory_ / name;
    std::ofstream stream(path);
    stream << text;
    stream.close();
    if (!stream) throw std::runtime_error("cannot write world fixture");
    return path.string();
  }
  std::string directory() const { return directory_.string(); }

 private:
  std::filesystem::path directory_;
};

template <typename Operation>
void expect_failure(Operation operation, const std::string &reason) {
  try {
    operation();
  } catch (const std::exception &error) {
    if (std::string(error.what()).find(reason) != std::string::npos) return;
    throw;
  }
  throw std::runtime_error("expected world validation failure: " + reason);
}
}  // namespace

int main() {
  using namespace curobo_ros_cpp;
  try {
    const Fixture fixture;
    const auto urdf = fixture.write("robot.urdf", "<robot name='fixture'><link name='base'/></robot>");
    const auto robot = fixture.write("robot.yml",
        "robot_cfg: {kinematics: {base_link: base, cspace: {joint_names: [], "
        "default_joint_position: []}, lock_joints: {}}}\n");
    const auto profile = fixture.write("profile.yml",
        "robot_id: fixture\nleft_joints: []\nright_joints: []\nee_links: {}\n"
        "gripper_joints: {}\ngripper_controllers: {}\ncamera_optical_frame: base\n"
        "camera_parent_frame: base\nfcl_self_collision: policy.yml\npayload_link: unused\n"
        "jaw_links: []\ngripper_base_link: unused\n");
    fixture.write("policy.yml", "self_collision_ignore: {}\n");
    fixture.write("geometry.yaml", "meshes: []\n");
    RobotModel model(urdf, robot, profile, fixture.directory(), fixture.directory());
    NativePlanner planner(model, fixture.directory(), {});
    const std::vector<Point> points{Point(.2f, .1f, .3f)};
    const Eigen::Vector3d center = Eigen::Vector3d::Zero();
    const Eigen::Vector3d dimensions = Eigen::Vector3d::Constant(2.);
    planner.set_world(points, .1, "voxel", center, dimensions);
    const auto valid_generation = planner.world_generation();
    for (int i = 0; i < 2; ++i)
      expect_failure([&] { planner.set_world(points, .01, "voxel", center, dimensions); },
                     "exceeds voxel capacity");
    if (planner.world_state().pitch != .1 || planner.world_generation() != valid_generation + 2)
      throw std::runtime_error("failed world replaced committed input or did not invalidate cache");
    expect_failure([&] { planner.synchronize("left_arm", {}); }, "requires successful rebuild");
    planner.set_world(points, .1, "voxel", center, dimensions);
    if (planner.world_generation() != valid_generation + 3)
      throw std::runtime_error("identical valid input did not rebuild an invalid scene");

    const auto bad_scene = fixture.write("bad_scene.yml",
        "cuboid: {invalid: {dims: [-1, 1, 1], pose: [0, 0, 0, 1, 0, 0, 0]}}\n");
    expect_failure([&] { planner.load_static_scene(bad_scene); }, "cuboid dimensions");
    expect_failure([&] { planner.synchronize("left_arm", {}); }, "requires successful rebuild");
    // A failed GPU update must also roll back the candidate static metadata.
    planner.set_world(points, .1, "voxel", center, dimensions);
    const std::vector<Point> mesh_points{Point(.25f, .25f, .25f), Point(.55f, .55f, .55f)};
    planner.set_world(mesh_points, .25, "mesh", center, dimensions);
    const auto mesh_generation = planner.world_generation();
    const std::vector<Point> same_mesh{Point(.56f, .56f, .56f), mesh_points.front()};
    planner.set_world(same_mesh, .25, "mesh", center, dimensions);
    if (planner.world_generation() != mesh_generation ||
        planner.world_state().points.front() != same_mesh.front())
      throw std::runtime_error("unchanged mesh occupancy rebuilt or failed to retain the latest points");
    planner.set_world({same_mesh.front(), Point(.26f, .25f, .25f)}, .25, "mesh", center, dimensions);
    if (planner.world_generation() != mesh_generation + 1)
      throw std::runtime_error("moving mesh origin did not rebuild the scene");
    std::cout << "World rebuild failure, recovery and occupancy reuse validation passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
