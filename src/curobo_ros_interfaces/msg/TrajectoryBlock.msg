# Cumulative immutable prefix. block_index starts at 1; final closes the task.
# cancel stops the active task and discards queued additions.
string task_id
uint32 block_index
bool final
bool cancel
trajectory_msgs/JointTrajectory trajectory
