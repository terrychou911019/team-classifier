import subprocess

print("Processing SportsMOT val")
cmd = ["python", "assign_teams.py", 
       "--data_path", "D:/CourtAthena/02_Datasets/sportsmot_publish/dataset/val", 
       "--pred_dir", "D:/CourtAthena/03_Experiments/TrackEval/data/trackers/mot_challenge/SportsMOT-val/mcbyte+filter+interpolate/data", 
       "--output_dir", "assign_teams_output/SportsMOT"]
subprocess.run(cmd, check=True)

print("Processing TGB test ")
cmd = ["python", "assign_teams.py", 
       "--data_path", "D:/CourtAthena/02_Datasets/TGB/dataset", 
       "--pred_dir", "D:/CourtAthena/03_Experiments/TrackEval/data/trackers/mot_challenge/TGB-test/mcbyte+filter+interpolate/data", 
       "--output_dir", "assign_teams_output/TGB"]
subprocess.run(cmd, check=True)

print("Processing TrackID3x3 test")
cmd = ["python", "assign_teams.py", 
       "--data_path", "D:/CourtAthena/02_Datasets/trackid3x3_720p/test", 
       "--pred_dir", "D:/CourtAthena/03_Experiments/TrackEval/data/trackers/mot_challenge/TrackID3x3-test/mcbyte+filter+interpolate/data", 
       "--output_dir", "assign_teams_output/TrackID3x3"]
subprocess.run(cmd, check=True)
