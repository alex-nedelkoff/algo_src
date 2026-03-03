#!/usr/bin/env bash
set -euo pipefail

# Configuration
EUROC_DIR="/data/euroc/MH_01_easy/mav0"
BAG_DIR="/results/MH_01_easy_bag"
CONFIG_PATH="/configs/estimator_config.yaml"
RESULTS_DIR="/results"
ESTIMATE_FILE="${RESULTS_DIR}/ov_estimate.txt"
TUM_EST_FILE="${RESULTS_DIR}/ov_estimate_tum.txt"
GT_CSV="/data/euroc/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv"

echo "========================================="
echo " OpenVINS Monocular Pipeline - MH_01_easy"
echo "========================================="

# [1/5] Convert EuRoC ASL to ROS2 bag (idempotent)
if [ -d "${BAG_DIR}" ] && [ -f "${BAG_DIR}/metadata.yaml" ]; then
    echo "[1/5] ROS2 bag already exists at ${BAG_DIR}, skipping conversion."
else
    echo "[1/5] Converting EuRoC ASL to ROS2 bag..."
    python3 /scripts/euroc_to_rosbag2.py "${EUROC_DIR}" "${BAG_DIR}"
fi
echo "[1/5] Bag info:"
ros2 bag info "${BAG_DIR}"

# [2/5] Launch OpenVINS in background
# NOTE: We use ros2 run instead of ros2 launch because the launch file does not
# forward filepath_est/filepath_std parameters to the node. Without these, the
# node defaults to relative paths whose empty parent causes
# boost::filesystem::create_directories to throw "Invalid argument".
echo ""
echo "[2/5] Launching OpenVINS (monocular mode)..."
ros2 run ov_msckf run_subscribe_msckf --ros-args \
    -r __ns:=/ov_msckf \
    -r /ov_msckf/imu0:=/imu0 \
    -r /ov_msckf/cam0/image_raw:=/cam0/image_raw \
    -p config_path:="${CONFIG_PATH}" \
    -p use_stereo:=false \
    -p max_cameras:=1 \
    -p save_total_state:=true \
    -p filepath_est:="${ESTIMATE_FILE}" \
    -p filepath_std:="${RESULTS_DIR}/ov_estimate_std.txt" \
    -p filepath_gt:="${RESULTS_DIR}/ov_groundtruth.txt" \
    -p verbosity:=INFO &
OPENVINS_PID=$!

# Wait for OpenVINS node to initialize
echo "Waiting for OpenVINS to initialize..."
sleep 5

# [3/5] Play the ROS2 bag
echo ""
echo "[3/5] Playing ROS2 bag (this takes ~3 minutes at real-time rate)..."
ros2 bag play "${BAG_DIR}"
echo "Bag playback complete."

# Wait for OpenVINS to finish processing remaining data
echo "Waiting for OpenVINS to finish processing..."
sleep 15

# Send SIGINT for clean shutdown (flushes output files)
echo "Stopping OpenVINS..."
# Kill the entire process group (ros2 run spawns child processes)
kill -SIGINT -${OPENVINS_PID} 2>/dev/null || kill -SIGINT ${OPENVINS_PID} 2>/dev/null || true
sleep 3
# Force kill if still running
kill -9 -${OPENVINS_PID} 2>/dev/null || kill -9 ${OPENVINS_PID} 2>/dev/null || true
wait ${OPENVINS_PID} 2>/dev/null || true
echo "OpenVINS stopped."

# [4/5] Convert output to TUM format
echo ""
echo "[4/5] Converting OpenVINS output to TUM format..."
if [ ! -f "${ESTIMATE_FILE}" ]; then
    echo "ERROR: OpenVINS estimate file not found at ${ESTIMATE_FILE}"
    echo "OpenVINS may have failed to initialize or produce output."
    exit 1
fi
python3 /scripts/openvins_to_tum.py "${ESTIMATE_FILE}" "${TUM_EST_FILE}"

# [5/6] Evaluate accuracy with evo
echo ""
echo "[5/6] Evaluating trajectory accuracy (evo_ape)..."
evo_ape euroc "${GT_CSV}" "${TUM_EST_FILE}" \
    -vas \
    --save_results "${RESULTS_DIR}/evo_ape_results.zip" \
    2>&1 | tee "${RESULTS_DIR}/evo_ape_output.txt"

# [6/6] Validate trajectory
echo ""
echo "[6/6] Validating trajectory..."
python3 /scripts/validate_trajectory.py "${TUM_EST_FILE}" "${GT_CSV}"

echo ""
echo "========================================="
echo " Pipeline complete!"
echo "========================================="
echo "Results:"
ls -lh "${RESULTS_DIR}/"
