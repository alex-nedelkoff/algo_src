"""UZH-FPV dataset constants (Snapdragon stereo + ground truth)."""

ROSBAG_BASE_URL = "http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3"
CALIB_BASE_URL = "http://rpg.ifi.uzh.ch/datasets/uzh-fpv/calib"

# Sequences with Snapdragon stereo + ground truth
SEQUENCES_WITH_GT = [
    "indoor_forward_3", "indoor_forward_5", "indoor_forward_6",
    "indoor_forward_7", "indoor_forward_9", "indoor_forward_10",
    "indoor_45_2", "indoor_45_4", "indoor_45_9",
    "indoor_45_12", "indoor_45_13", "indoor_45_14",
    "outdoor_forward_1", "outdoor_forward_3", "outdoor_forward_5",
    "outdoor_45_1",
]

# Start with these 3 for validation
VALIDATION_SEQUENCES = [
    "indoor_forward_3",
    "indoor_45_2",
    "outdoor_forward_3",
]

# Snapdragon stereo camera ROS topics
LEFT_IMAGE_TOPIC = "/snappy_cam/stereo_l"
RIGHT_IMAGE_TOPIC = "/snappy_cam/stereo_r"
IMU_TOPIC = "/snappy_imu"
