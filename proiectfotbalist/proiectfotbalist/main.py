import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class SoccerBot(Node):
    def __init__(self):
        super().__init__('soccer_bot')

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.image_subscription = self.create_subscription(Image,'/camera/image_raw',self.image_callback,10)

        self.bridge = CvBridge()
        self.state = 'searching'
        self.ball_detected = False
        self.goal_scored = False
        self.ball_detected_logged = False 
        self.goal_scored_logged = False

        self.black_line_detection_count = 0
        self.detection_threshold = 5 
        self.line_detected = False

    def rotate_in_place(self):
        """Rotate the robot to search for the ball."""
        twist = Twist()
        twist.angular.z = 0.5
        self.publisher.publish(twist)

    def follow_ball(self, ball_x, frame_center_x):
        """Move toward the ball while aligning with it."""
        twist = Twist()

        alignment_error = frame_center_x - ball_x
        twist.angular.z = 0.005 * alignment_error

        twist.linear.x = 0.3
        self.publisher.publish(twist)

    def gradual_stop(self):
        """Gradually slow down the robot."""
        twist = Twist()
        twist.linear.x = 0.1 
        self.publisher.publish(twist)

    def stop_robot(self):
        """Stop the robot."""
        twist = Twist()
        self.publisher.publish(twist)

    def image_callback(self, msg):
        """Process the image to detect the ball and the black line."""
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        height, width = frame.shape[:2]
        bottom_region = hsv_frame[int(0.6 * height):, :] 

        blue_lower = np.array([100, 150, 50])
        blue_upper = np.array([140, 255, 255])
        blue_mask = cv2.inRange(hsv_frame, blue_lower, blue_upper)
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        black_lower = np.array([0, 0, 0])
        black_upper = np.array([180, 255, 50])
        black_mask = cv2.inRange(bottom_region, black_lower, black_upper)
        black_contours, _ = cv2.findContours(black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        frame_center_x = frame.shape[1] // 2

        if blue_contours and not self.goal_scored:
            largest_contour = max(blue_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            ball_x = x + w // 2
            self.ball_detected = True

            if not self.ball_detected_logged:
                self.get_logger().info("Ball Detected!")
                self.ball_detected_logged = True

            if self.state == 'searching' or self.state == 'following':
                self.state = 'following'
                self.follow_ball(ball_x, frame_center_x)
        else:
            self.ball_detected = False

        if not self.ball_detected and self.state == 'searching' and not self.goal_scored:
            self.rotate_in_place()

        if black_contours and not self.goal_scored:
            largest_line = max(black_contours, key=cv2.contourArea)
            _, _, w, h = cv2.boundingRect(largest_line)

            if w * h > 1500: 
                self.black_line_detection_count += 1
            else:
                self.black_line_detection_count = 0 

            if self.black_line_detection_count > 0 and self.black_line_detection_count < self.detection_threshold:
                self.line_detected = True
                self.gradual_stop()
            elif self.black_line_detection_count >= self.detection_threshold:
                self.goal_scored = True

        if self.goal_scored and not self.goal_scored_logged:
            self.state = 'stopped'
            self.stop_robot()
            self.get_logger().info("Goal Scored!")
            self.goal_scored_logged = True


def main(args=None):
    rclpy.init(args=args)
    soccer_bot = SoccerBot()
    rclpy.spin(soccer_bot)
    soccer_bot.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
