# Exercise 3
- download packages
- download exercise 3
- Get a map of the room
- Save the map
- Run the acml simulation and try to move the robot around by giving him goals

---

- Determine 5 goals locations in the map (can use clicked points)
- Create a script that drives robot to the 5 location sequentially
- End when the robot has reached the final location

### 1. Create tosrun map_server map_saver -f ~/PROJECT/src/exercise3/maps/map1he room map 

```bash
roslaunch exercise3 rins_world.launch # Start the 3D simulation

# Build the map of the working environment of the robot.
# Using gmapping package which builds a map based on the laser scan (Kinect)
roslaunch exercise3 gmapping_simulation.launch

# View the map that we are building
roslaunch turtlebot_rviz_launchers view_navigation.launch

# Control the robot with the keyboard via terminal
# Move slowly, stop frequently and rotate around the axis
roslaunch turtlebot_teleop keyboard_teleop.launch

# Once you are satisified with the map, save it
r
```

### 2. Navigation
Once the map is built we are ready to let the robot drive itself
- **Close all the running programs**

```bash
roslaunch exercise3 rins_world.launch

# Start the localitzation node amcl (Adaptive MonteCarlo Localization pkg)
# Edit the launch file to use the map that we have created
# <arg name="map_file" default="$(find exercise3)/maps/map1.yaml"/>
roslaunch exercise3 amcl_simulation.launch

# Open a rviz visualizer of the robot
roslaunch turtlebot_rviz_launchers view_navigation.launch
```

### 3. Sending movement goals from a node
Script that sends goals to the robot, `SimpleActionClient` to communicate with the `SimpleActionServer` that is available in `move_base`.

```bash
rostopic echo /move_base_simple/goal
rostopic echo --noarr /map
```



# 4. Exercise 4

```bash
roslaunch exercise3 rins_world.launch
roslaunch exercise3 amcl_simulation.launch
roslaunch turtlebot_rviz_launchers view_navigation.launch
rosrun exercise4 breadcrumbs
rosrun exercise4 face_localizer_dnn
rosrun exercise4 face_localizer_dlib # Face detection
```


# 5. Exercise 5

Debugging (ros console)

```bash
# Debug
```

Useful services for planning (move_base)
- 


