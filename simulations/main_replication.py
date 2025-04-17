import subprocess


def info(message, color="blue"):
    if color == "green":
        print(f"\033[92m{message}\033[0m")
    if color == "red":
        print(f"\033[31m{message}\033[0m")
    if color == "blue":
        print(f"\033[94m{message}\033[0m")


script_number = 1


info(f"Script number {script_number}: Replicating the weak and strong error decompositions from Figure 2 (a) and (b)")
subprocess.run(["python", "visualise_error_decomposition.py"])
script_number = script_number + 1

info(f"Script number {script_number}: Replicating the signals from Figure 2 (c)")
subprocess.run(["python", "signals.py"])
script_number = script_number + 1

info(f"Script number {script_number}: Replicating the relative efficiencies from Figure 2 (d)")
subprocess.run(["python", "TruncatedSVD_replication.py"])
script_number = script_number + 1

info(f"Script number {script_number}: Replicating the signal estimation from Figure 9 (a) and (b)")
subprocess.run(["python", "signal_estimation_comparison.py"])
script_number = script_number + 1
