# Decision 002: Export ROS Bag Topics to CSV

## Decision
Use a Python preprocessing script to export `/imu0` and `/leica/position` from the EuRoC ROS bag into CSV files.

## Reason
CSV format is easier to inspect, debug, and use for machine learning preprocessing than operating directly on ROS bag files.

## Consequence
The ROS bag remains the raw source of truth, while downstream preprocessing and training use exported CSV files.