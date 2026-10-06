#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped


class GapFollowNode(Node):
    """
    Implement Gap Following on the car
    This is just a template, you are free to implement your own node!
    """

    def __init__(self):
        super().__init__('gap_follow_node')

        # TODO: Topics & Subs, Pubs
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # TODO: Subscribe to LIDAR
        self.scan_sub = self.create_subscription(
            LaserScan,
            lidarscan_topic,
            self.lidar_callback,
            10
        )

        # TODO: Publish to drive
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            drive_topic,
            10
        )

        self.angle = 0.0
        self.ranges = []

    def preprocess_lidar(self, ranges, angle_min, angle_increment):
        """TODO: Preprocess the LiDAR scan array. Expert implementation includes:
            1. Setting each value to the mean over some window
            2. Rejecting high values (eg. > 3m)
        """
        proc_ranges = np.array(ranges, dtype=float)

        # Only use the front part of the LiDAR
        min_angle = np.radians(-70)
        max_angle = np.radians(70)

        min_indx = int((min_angle - angle_min) / angle_increment)
        max_indx = int((max_angle - angle_min) / angle_increment)

        min_indx = max(0, min_indx)
        max_indx = min(len(proc_ranges) - 1, max_indx)

        # Clean bad LiDAR values
        proc_ranges[np.isnan(proc_ranges)] = 0.0
        proc_ranges[np.isinf(proc_ranges)] = 3.0
        proc_ranges[proc_ranges > 3.0] = 3.0
        proc_ranges[proc_ranges < 0.0] = 0.0

        self.min_indx = min_indx
        self.max_indx = max_indx

        return proc_ranges, min_indx, max_indx

    def find_max_gap(self, free_space_ranges):
        """TODO: Return the start index & end index of the max gap in free_space_ranges """
        min_indx = self.min_indx
        max_indx = self.max_indx

        start = min_indx
        end = min_indx
        current_start = -1
        longest_duration = 0

        for i in range(min_indx, max_indx + 1):

            if free_space_ranges[i] > 0.0:
                if current_start == -1:
                    current_start = i

            else:
                if current_start != -1:
                    current_duration = i - current_start

                    if current_duration > longest_duration:
                        longest_duration = current_duration
                        start = current_start
                        end = i - 1

                    current_start = -1

        # Check if the last gap reaches the end
        if current_start != -1:
            current_duration = max_indx - current_start + 1

            if current_duration > longest_duration:
                start = current_start
                end = max_indx

        self.proc_ranges = free_space_ranges

        return start, end

    def find_best_point(self, start_i, end_i, ranges,
                        angle_min, angle_increment):
        """TODO: Find the best point inside the largest gap."""

        # Choose the middle of the largest gap
        best_index = (start_i + end_i) // 2

        # Convert LiDAR index to steering angle
        best_angle = angle_min + best_index * angle_increment

        return best_angle

    def lidar_callback(self, data):
        """Process LiDAR scan and publish a drive command."""

        ranges = data.ranges
        angle_min = data.angle_min
        angle_increment = data.angle_increment

        proc_ranges, min_indx, max_indx = self.preprocess_lidar(
            ranges,
            angle_min,
            angle_increment
        )

        # TODO: Find closest point to LiDAR
        front_ranges = proc_ranges[min_indx:max_indx + 1]

        # Ignore zero values when finding the closest obstacle
        valid_ranges = np.where(
            front_ranges > 0.0,
            front_ranges,
            np.inf
        )

        # If there is no valid LiDAR data, stop the car
        if np.all(np.isinf(valid_ranges)):
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.steering_angle = 0.0
            drive_msg.drive.speed = 0.0
            self.drive_pub.publish(drive_msg)
            return

        closest_local_index = int(np.argmin(valid_ranges))
        closest_index = min_indx + closest_local_index
        closest_distance = valid_ranges[closest_local_index]

        # Check the distance directly in front of the car
        center_index = int((0.0 - angle_min) / angle_increment)

        front_width = int(
            np.radians(10) / abs(angle_increment)
        )

        front_start = max(
            min_indx,
            center_index - front_width
        )

        front_end = min(
            max_indx,
            center_index + front_width
        )

        center_ranges = proc_ranges[front_start:front_end + 1]

        valid_center = center_ranges[center_ranges > 0.0]

        if len(valid_center) > 0:
            front_distance = np.min(valid_center)
        else:
            front_distance = np.inf

        # Emergency stop if a wall is too close in front
        if front_distance < 0.45:
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.steering_angle = 0.0
            drive_msg.drive.speed = 0.0
            self.drive_pub.publish(drive_msg)

            self.get_logger().warn(
                'Obstacle too close - stopping car'
            )
            return

        # TODO: Dynamic safety bubble size
        if closest_distance < 0.75:
            bubble_radius = 40
        elif closest_distance < 1.25:
            bubble_radius = 30
        else:
            bubble_radius = 20

        # TODO: Eliminate all points inside bubble
        bubble_start = max(
            min_indx,
            closest_index - bubble_radius
        )

        bubble_end = min(
            max_indx,
            closest_index + bubble_radius
        )

        proc_ranges[bubble_start:bubble_end + 1] = 0.0

        # TODO: Find max length gap
        gap_start, gap_end = self.find_max_gap(proc_ranges)

        # TODO: Find the best point in the gap
        best_angle = self.find_best_point(
            gap_start,
            gap_end,
            proc_ranges,
            angle_min,
            angle_increment
        )

        # Limit steering angle so the car does not turn too sharply
        max_steering = np.radians(30)

        best_angle = np.clip(
            best_angle,
            -max_steering,
            max_steering
        )

        # TODO: Increase speed if best angle is close to zero
        if abs(best_angle) < np.radians(10):
            speed = 1.5

        elif abs(best_angle) < np.radians(20):
            speed = 1.0

        else:
            speed = 0.7

        # Slow down when getting closer to an obstacle
        if front_distance < 0.80:
            speed = min(speed, 0.5)

        # TODO: Smooth steering angles
        steering_angle = (
            0.6 * self.angle +
            0.4 * best_angle
        )

        self.angle = steering_angle

        # TODO: Publish Drive message
        drive_msg = AckermannDriveStamped()

        drive_msg.drive.steering_angle = float(
            steering_angle
        )

        drive_msg.drive.speed = float(speed)

        self.drive_pub.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)

    print("GapFollow Initialized")

    gap_follow_node = GapFollowNode()

    try:
        rclpy.spin(gap_follow_node)

    except KeyboardInterrupt:
        pass

    # Send a stop command before shutting down
    stop_msg = AckermannDriveStamped()
    stop_msg.drive.steering_angle = 0.0
    stop_msg.drive.speed = 0.0
    gap_follow_node.drive_pub.publish(stop_msg)

    gap_follow_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
