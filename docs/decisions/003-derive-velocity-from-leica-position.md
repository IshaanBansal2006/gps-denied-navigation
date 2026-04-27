# Decision 003: Derive Velocity from Leica Position

## Decision
Use timestamped Leica position measurements to derive velocity ground truth.

## Reason
The ROS bag contains Leica position ground truth but does not expose direct ground-truth velocity as a topic.

## Consequence
A preprocessing step is required to compute velocity from position before generating delta-velocity labels.