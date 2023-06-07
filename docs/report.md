# Report

## 1. Introduction

## 2. Methods

Describe the methods that you have used; how does your system detect and recognize faces, rings, and cylinders, what is the theory behind these methods? How does the robot navigate around the competition area? Focus on the theoretical part of the methods and not on the implementation.

### Exploration

#### Initial Keypoints

We have created an exploration function that returns the best keypoints to visit. It does that in the following way:

1. We create a skeleton overlay of the map
2. We find the branch points on the skeleton overlay. We leverage the Harris corner detector to find the corners of the skeleton overlay. Following corner detection, we dilate the results and apply a threshold to isolate the optimal corners. These corners represent the branch points of the skeleton overlay
3. We then further filter the branch points by removing those that are too close to each other.

The resulting keypoints look like this:

![keypoints](./images/kmap.png)

#### Exploration Extended: Additional Keypoints from Searched Space

When the keypoints identified through branch points are not sufficient for exploration, the system dynamically generates more keypoints. These additional keypoints are derived from the space that has not yet been searched by the robot.

The robot maintains a record of its searched space that gets updated every 0.4 seconds, considering a radius of 10 pixels around the robot's current position. This ongoing update allows the robot to be aware of its immediate surroundings.

To complement this process, a function called get_additional_goals() identifies potential new goals from the unsearched space. This function creates a representation of the unsearched space, identifies separate clusters in this space, and then chooses new keypoints from these clusters.

In conclusion, this extended exploration strategy helps the robot more thoroughly explore its environment by not only following the skeleton overlay of the map but also dynamically identifying new regions to explore based on its current knowledge of the searched and unsearched spaces.

### Cylinder Detection

### Ring Detection

### Ground Ring Detection

### Face Detection

### Poster Detection

## 3. Implementation and Integration

Describe the actual implementation of the methods and integration of different components into the integrated system. Describe how have you integrated all the components in ROS.


## 4. Results

## 5. Division of work

## 6. Conclusion


## Brain

The brain is the main node of the project. It communicates with all other nodes:

- `map_manager` that handles the map and costmap
- `ring_manager` than handles the ring detection
- `face_manager` that handles the face detection
- `cylinder_manager` that handles the cylinder detection
- `ground_ring_manager` that handles the ground ring detection
- `move_base` that handles the navigation

## Map Handling

We have created an `MapManager` object for handling the map and costmap.

### Storing costmap and map

Map is stored from the `/map` topic.

Costmap is stored from the `/move_base/global_costmap/costmap` topic.

### Creating Keypoints

It is also responsible for creating the key points for the robot to visit. It does that in the
following way:

- It creates a skeleton overlay of the map
- Than it finds the branch points of the skeleton

![map](./images/map.png)

### Additional Keypoints from Searched Space

When these keypoints are not sufficient, it will also create more keypoints based on the searched space. Searched space is updated every 0.4 seconds:

```python
self.searched_space_timer = rospy.Timer(
            rospy.Duration(0.4), lambda event: self.map_manager.update_searched_space()
        )
```

- when the robot moves, it stores the circle around the robot as a searched space
- when all keypoints are reached, and not all goals have been found, based on the searched space, new keypoints are created
- It accomplishes this by processing a binary map of searched spaces, where it isolates unsearched areas and uses computer vision techniques to identify clusters of these areas. The function treats the centroid of each cluster as a potential goal. If the centroid is in a safe space or a safe point within the bounding box of the cluster can be found, it's added to the list of additional goals for further exploration.

## Face Detection

## Ring Detection

## Ground Ring Detection

## Cylinder Detection

### Detection Workflow

We have a subscriber on `/camera/depth/points` topic which calls the `cylinder_detection_callback` function:

1. Downsample the point cloud data, to reduce number of points (with `voxel_grid_filter`)
2. Filter the point cloud data based on depth (z-axis `[0, 1.9]`)
3. Filter the point cloud based on the height (y-axis `[-0.3, 0.2]`)
4. Estimate **normals**
5. Perform planar segmentation to identify largest planar component, remove points on it from the point cloud
   - largest planar component is the ground
6. Perform **cylinder segmentation** using RANSAC
   - we use a model `pcl:SACMODEL_CYLINDER` to find cylindrical shape in the data.
   - then we perform up to 10000 iterations of RANSAC to find the best model

### Robust Cylinder Detection

When cylinder is detected, we check if there have been any other detections at this `Pose`. If there were more than 5 detections of cylinder with pose close enough (`object_proximity_threshold=1`), we assume that this is a valid detection and we publish it on the `/detected_cylinders` topic.
