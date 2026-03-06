import subprocess
import os

team_classifier = "solider"
os.makedirs(f"assign_teams_output/{team_classifier}/TGB", exist_ok=True)

print("Processing TGB test")
cmd = ["python", f"assign_teams_{team_classifier}.py", 
       "--data_path", "D:/CourtAthena/02_Datasets/TGB/dataset", 
       "--pred_dir", "D:/CourtAthena/03_Experiments/TrackEval/data/trackers/mot_challenge/TGB-test/mcbyte/data", 
       "--output_dir", f"assign_teams_output/{team_classifier}/TGB"]
subprocess.run(cmd, check=True)
