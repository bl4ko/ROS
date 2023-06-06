# TurtleBotExploration

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

