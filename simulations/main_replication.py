import subprocess


def info(message, color="blue"):
    if color == "green":
        print(f"\033[92m{message}\033[0m")
    if color == "red":
        print(f"\033[31m{message}\033[0m")
    if color == "blue":
        print(f"\033[94m{message}\033[0m")


script_number = 1


info(
    f"Script number {script_number}: Replicating the decompositions for two different signals from Figure 1 (a) and (b)"
)
subprocess.run(["python", "general_error_decomposition_plots.py"])
script_number = script_number + 1

info(f"Script number {script_number}: Replicating the weak and strong error decompositions from Figure 2 (a) and (b)")
subprocess.run(["python", "visualise_error_decomposition.py"])
script_number = script_number + 1

info(f"Script number {script_number}: Replicating the signals from Figure 2 (c)")
subprocess.run(["python", "signals.py"])
script_number = script_number + 1

info(f"Script number {script_number}: Replicating the relative efficiencies from Figure 2 (d) [Truncated SVD]")
subprocess.run(["python", "TruncatedSVD_Replication.py"])
script_number = script_number + 1

info(f"Script number {script_number}: Replicating the relative efficiencies from Figure 3 (a) and (b) [Landweber]")
subprocess.run(["python", "Landweber_Replication.py"])
script_number = script_number + 1

info(
    f"Script number {script_number}: Replicating the relative efficiencies from Figure 4 (a) and (b) [Conjugate gradients]"
)
subprocess.run(["python", "ConjugateGradients_Replication.py"])
script_number = script_number + 1


info(f"Script number {script_number}: Replicating the signals from Figure 5 (a) and (b) [L2Boost - signals]")
subprocess.run(["python", "L2Boost_signals.py"])
script_number = script_number + 1

info(f"Script number {script_number}: Replicating the relative efficiencies from Figure 6 [L2Boost - Replication]")
subprocess.run(["python", "L2Boost_Replication.py"])
script_number = script_number + 1

info(f"Script number {script_number}: Replicating the signal estimation from Figure 9 (a) and (b)")
subprocess.run(["python", "signal_estimation_comparison.py"])
script_number = script_number + 1


info(f"Script number {script_number}: Replicating the signal estimation from Figure 10 (a) and (b)")
subprocess.run(["python", "phillips_data.py"])
script_number = script_number + 1

info(f"Script number {script_number}: Replicating the stopping times and errors from Figure 11 (a) and (b)")
subprocess.run(["python", "ComparisonStudy.py"])
script_number = script_number + 1

# Note that the dataset needs to be specified in the script
info(f"Script number {script_number}: Replicating the stopping times and errors from Table 1")
subprocess.run(["python", "timing_es.py"])
script_number = script_number + 1


info(f"Script number {script_number}: Replicating the error decomposition from Figure 12 (b) and (d) [Landweber]")
subprocess.run(["python", "Simulation_counterexample_landweber.py"])
script_number = script_number + 1

info(f"Script number {script_number}: Replicating the error decomposition from Figure 12 (a) and (c) [Truncated SVD]")
subprocess.run(["python", "Simulation_counterexample_tSVD.py"])
script_number = script_number + 1
