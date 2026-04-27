# Decision 005: Use Interpolation for Leica-to-IMU Alignment

## Decision
Align Leica-derived velocity to IMU timestamps using linear interpolation.

## Reason
The IMU stream runs much faster than the Leica-derived velocity stream. Interpolation provides smoother and more principled label alignment than nearest-neighbor assignment.

## Consequence
Training labels are now available at each retained IMU timestamp, enabling fixed-length IMU-window construction for supervised learning.