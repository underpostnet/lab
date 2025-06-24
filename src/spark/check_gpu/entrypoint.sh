#!/bin/bash

# Custom Entrypoint for Spark Applications in Kubernetes.
# This script ensures any necessary environment setup is performed
# before the main Spark command (provided by the Spark Operator) is executed.

echo "Custom Entrypoint: Starting Spark application."
echo "Arguments received by entrypoint.sh: $*"

# Define the Spark submit executable path. This should be consistent with
# where spark-submit is located within your Docker image.
SPARK_SUBMIT="/opt/spark/bin/spark-submit"

# The Spark Operator, when running in 'cluster' mode, often prefixes the actual
# command with 'driver' (for driver pods) or 'executor' (for executor pods).
# We need to skip this initial argument to correctly execute the spark-submit command
# or other Spark-related commands that follow.

# Check if the first argument is 'driver' or 'executor' and shift it if so.
if [[ "$1" == "driver" || "$1" == "executor" ]]; then
    echo "Skipping initial argument: $1"
    shift # Remove the first argument (e.g., 'driver')
fi

# Now, "$@" should contain the actual command-line arguments intended for spark-submit
# (e.g., --properties-file ..., --class ..., local:///opt/spark/work-dir/main_check_gpu.py).
# We explicitly call spark-submit with these arguments.
echo "Executing Spark Submit with arguments: $*"
exec "$SPARK_SUBMIT" "$@"
