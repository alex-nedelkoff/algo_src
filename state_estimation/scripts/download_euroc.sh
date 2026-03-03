#!/usr/bin/env bash
#
# Download EuRoC MAV dataset sequences in ASL format.
# Default: MH_01_easy. Pass sequence names as arguments for additional ones.
#
# Source: ETH Research Collection (https://doi.org/10.3929/ethz-b-000690084)
#
# Usage:
#   ./download_euroc.sh                     # downloads MH_01_easy
#   ./download_euroc.sh MH_02_easy V1_01_easy
#
# Directory structure after download:
#   state_estimation/data/euroc/<SEQUENCE>/mav0/
#     ├── cam0/         (stereo left images)
#     ├── cam1/         (stereo right images)
#     ├── imu0/         (IMU measurements)
#     ├── leica0/       (Leica total station ground truth)
#     ├── state_groundtruth_estimate0/  (ground truth state estimates)
#     └── body.yaml, ...
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data/euroc"

# ETH Research Collection bitstream URLs (bundled by category)
get_bundle_url() {
    case "$1" in
        machine_hall)
            echo "https://www.research-collection.ethz.ch/bitstreams/7b2419c1-62b5-4714-b7f8-485e5fe3e5fe/download" ;;
        vicon_room1)
            echo "https://www.research-collection.ethz.ch/bitstreams/02ecda9a-298f-498b-970c-b7c44334d880/download" ;;
        vicon_room2)
            echo "https://www.research-collection.ethz.ch/bitstreams/ea12bc01-3677-4b4c-853d-87c7870b8c44/download" ;;
    esac
}

get_category() {
    case "$1" in
        MH_01_easy|MH_02_easy|MH_03_medium|MH_04_difficult|MH_05_difficult)
            echo "machine_hall" ;;
        V1_01_easy|V1_02_medium|V1_03_difficult)
            echo "vicon_room1" ;;
        V2_01_easy|V2_02_medium|V2_03_difficult)
            echo "vicon_room2" ;;
        *)
            echo "ERROR: Unknown sequence '$1'" >&2
            echo "Available: MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult" >&2
            echo "           V1_01_easy V1_02_medium V1_03_difficult" >&2
            echo "           V2_01_easy V2_02_medium V2_03_difficult" >&2
            return 1 ;;
    esac
}

MIN_FILE_COUNT=100

# --- Step 1: Download category bundle (idempotent) ---
download_bundle() {
    local category="$1"
    local url
    url="$(get_bundle_url "$category")"
    local bundle_zip="${DATA_DIR}/${category}_bundle.zip"

    mkdir -p "${DATA_DIR}"

    if [ -f "${bundle_zip}" ]; then
        echo "[CACHE] ${category} bundle already downloaded: ${bundle_zip}"
        return 0
    fi

    echo "[DOWN] Downloading ${category} bundle ..."
    echo "       Source: ETH Research Collection"
    curl -L --progress-bar -o "${bundle_zip}.partial" "${url}"
    mv "${bundle_zip}.partial" "${bundle_zip}"
    echo "[DOWN] Bundle saved: ${bundle_zip}"
}

# --- Step 2: Extract a single sequence from a bundle (idempotent) ---
extract_sequence() {
    local seq="$1"
    local category
    category="$(get_category "$seq")"

    local seq_dir="${DATA_DIR}/${seq}"
    local bundle_zip="${DATA_DIR}/${category}_bundle.zip"

    # Idempotent: skip if already extracted and verified
    if [ -d "${seq_dir}/mav0" ]; then
        local count
        count="$(find "${seq_dir}/mav0" -type f | wc -l | tr -d ' ')"
        if [ "$count" -ge "$MIN_FILE_COUNT" ]; then
            echo "[SKIP] ${seq} already extracted (${count} files in mav0/)"
            return 0
        else
            echo "[WARN] ${seq} exists but only has ${count} files — re-extracting"
            rm -rf "${seq_dir}"
        fi
    fi

    if [ ! -f "${bundle_zip}" ]; then
        echo "ERROR: Bundle not found: ${bundle_zip}" >&2
        echo "       Run the download step first." >&2
        return 1
    fi

    # The bundle is a zip-of-zips: <category>/<seq>/<seq>.zip contains mav0/
    # Extract only the inner ASL zip, then unzip that to get mav0/
    local inner_zip_path="${category}/${seq}/${seq}.zip"
    local inner_zip="${DATA_DIR}/${seq}.zip"

    echo "[UNZIP] Extracting inner zip: ${inner_zip_path} ..."
    unzip -q -o -j "${bundle_zip}" "${inner_zip_path}" -d "${DATA_DIR}" 2>/dev/null

    if [ ! -f "${inner_zip}" ]; then
        echo "ERROR: Could not find ${inner_zip_path} in bundle" >&2
        echo "[DEBUG] Bundle contents:" >&2
        unzip -l "${bundle_zip}" 2>/dev/null | grep "${seq}" >&2 || true
        return 1
    fi

    echo "[UNZIP] Extracting ${seq} ASL dataset ..."
    mkdir -p "${seq_dir}"
    unzip -q -o "${inner_zip}" -d "${seq_dir}"
    rm -f "${inner_zip}"

    if [ ! -d "${seq_dir}/mav0" ]; then
        echo "ERROR: Expected ${seq_dir}/mav0 after extraction but not found" >&2
        echo "[DEBUG] Contents of ${seq_dir}:" >&2
        ls -la "${seq_dir}" >&2
        return 1
    fi

    # Verify file count
    local count
    count="$(find "${seq_dir}/mav0" -type f | wc -l | tr -d ' ')"
    if [ "$count" -lt "$MIN_FILE_COUNT" ]; then
        echo "ERROR: Only ${count} files in ${seq_dir}/mav0 — may be corrupted" >&2
        return 1
    fi

    # Verify key directories
    local missing=0
    for subdir in cam0/data imu0 state_groundtruth_estimate0; do
        if [ ! -d "${seq_dir}/mav0/${subdir}" ]; then
            echo "ERROR: Missing expected directory mav0/${subdir}" >&2
            missing=1
        fi
    done
    if [ "$missing" -eq 1 ]; then
        return 1
    fi

    echo "[OK]   ${seq} verified (${count} files in mav0/)"
}

# ---- Main ----

if [ $# -eq 0 ]; then
    set -- "MH_01_easy"
fi

echo "=== EuRoC MAV Dataset Downloader ==="
echo "Target directory: ${DATA_DIR}"
echo "Sequences: $*"
echo ""

# Step 1: Download all needed bundles
failed=0
for seq in "$@"; do
    category="$(get_category "$seq")" || exit 1
    if ! download_bundle "$category"; then
        echo "[FAIL] Download failed for ${category}"
        failed=1
    fi
done

if [ "$failed" -eq 1 ]; then
    echo "Some downloads failed. Bundle zips preserved for retry."
    exit 1
fi

echo ""

# Step 2: Extract requested sequences
for seq in "$@"; do
    if ! extract_sequence "$seq"; then
        echo "[FAIL] ${seq}"
        failed=1
    fi
    echo ""
done

if [ "$failed" -eq 1 ]; then
    echo "Some extractions failed. Check errors above."
    echo "Bundle zips preserved — you can re-run to retry extraction."
    exit 1
fi

# NOTE: Bundle zips are intentionally kept. Delete manually when no longer needed:
#   rm state_estimation/data/euroc/*_bundle.zip

echo "All sequences downloaded and verified."
echo "(Bundle zips preserved in ${DATA_DIR} — delete manually when done)"
echo ""
echo "To use with Docker:"
echo "  cd state_estimation && docker compose up"
echo "  (data is mounted read-only at /data/euroc/)"
