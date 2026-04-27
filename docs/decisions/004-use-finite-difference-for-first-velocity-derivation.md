# Decision 004: Use Finite Differences for Initial Velocity Derivation

## Decision
Use first-order finite differences on Leica position to derive velocity for the initial baseline pipeline.

## Reason
The ROS bag does not expose direct velocity ground truth, and finite differences provide the simplest valid first method for generating velocity labels.

## Consequence
Derived velocities may be noisy, so later iterations may add smoothing or improved differentiation methods.