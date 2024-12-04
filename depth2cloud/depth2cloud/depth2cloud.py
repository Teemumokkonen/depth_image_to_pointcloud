import rclpy 
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from sensor_msgs_py.point_cloud2 import create_cloud
import struct
from std_msgs.msg import Header
from scipy.spatial.transform import Rotation as R

class ImageConverter(Node):
    def __init__(self):
        super().__init__('depth_converter')
        self.camera_sub = self.create_subscription(
            Image,
            "/carla/ego_vehicle/rgb_depth/image",
            self.image_cb,
            1
        )

        self.cloud_pub = self.create_publisher(PointCloud2, "camera/cloud", 1)
        self.bridge = CvBridge()
        M = np.eye(4)
        M[:3, :3] = R.from_euler('zyx', [0, 0, 0], degrees=True).as_matrix()

        self.extrinsics = M

    def image_cb(self, msg: Image):
        self.get_logger().info("image received")
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        xyz = self.relative_depth_to_m(cv_image)
        xyz = self.transform_xyz(xyz)
        cloud_msg = self.create_pointcloud2_msg(xyz)
        self.cloud_pub.publish(cloud_msg)

    def create_pointcloud2_msg(self, points, frame_id="kitti_velo"):
        header = Header()
        header.frame_id = "ego_vehicle/rgb_depth"
        header.stamp = self.get_clock().now().to_msg()
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        # Pack the RGB colors as floats
        cloud_data = []
        for (x, y, z) in points: # drop redundant dim
            cloud_data.append([x, y, z])

        # Create the PointCloud2 message
        pointcloud_msg = create_cloud(header, fields=fields, points=cloud_data)
        return pointcloud_msg

    def transform_xyz(self, xyz):
        homogeneous_points = np.hstack((xyz, np.ones((xyz.shape[0], 1))))
        return (homogeneous_points @ self.extrinsics.T)[:, :3]


    def relative_depth_to_m(self, depth_image):
        """
        Convert depth image to XYZ points in camera coordinates.
        :param depth_image: HxW numpy array with depth values
        :param camera_intrinsics: dictionary with fx, fy, cx, cy
        :return: Nx3 numpy array of XYZ points
        """
        h, w = depth_image.shape
        fx, fy = 200, 200
        cx, cy = 200, 35
        
        # Create mesh grid for pixel coordinates
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Convert depth to camera coordinates
        z = depth_image
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        return np.dstack((x, y, z)).reshape(-1, 3)



def main():
    rclpy.init()
    node = ImageConverter()
    rclpy.spin(node)


if __name__=="__main__":
    main()