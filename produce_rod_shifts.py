from rod_shift_functions import get_shifts, is_trackable
import numpy as np
import argparse
from tqdm import tqdm
import os, sys
from datetime import datetime, timedelta
from npy_append_array import NpyAppendArray

sys.path.insert(0,os.path.abspath('../../admx_analysis_tools/datatypes_and_database/'))
import admx_db_interface
sys.path.insert(0,os.path.abspath('../admx_analysis_tools/parameter_extraction/mainline_analysis/'))
from parameter_functions import searchId, smooth_parameter, convert_magnet_current_to_B_field

### maximum number of digitizations to pull, mostly a fail safe
max_lines=1000000


# Where the relevant egg files are kept
hires_directory='/pnfs/admx/persistent/high-res/'

parser = argparse.ArgumentParser()
parser.add_argument('start_date', help='Start date, YYY-MM-DD HH:MM:SS-TZ')
parser.add_argument('end_date', help='End date, YYY-MM-DD HH:MM:SS-TZ')
parser.add_argument('save_fname', help='Rod shift record target file path')
parser.add_argument('-t', help='Shift in kHz between initial and final measaurements that triggers the analysis on a file.', default=5.0, type=float)
parser.add_argument('-n', help='Number of files analysed per write to output file', default=1, type=int)

args = parser.parse_args()

target_path = args.save_fname
thresh = args.t
files_per_write = args.n

print(f"Will save rod shifts to {target_path}")
if os.path.isfile(target_path): 
    print("Target file already exists! Delete it if you really want to overwrite.")
    exit(-1)

print(f"Threshold: {thresh} kHz")

start_time = datetime.fromisoformat(args.start_date)
stop_time = datetime.fromisoformat(args.end_date)

db=admx_db_interface.ADMXDB()
db.hostname="admxdb01.fnal.gov"
db.dbname="admx"

print("Querying DB for metadata...")

query1 = "SELECT timestamp,digitizer_log_reference FROM axion_scan_log WHERE timestamp < '"+str(stop_time)+"' AND timestamp>'"+str(start_time)+"' ORDER BY timestamp asc LIMIT "+str(max_lines)

records = db.send_admxdb_query(query1)

if len(records) == 0:
    print("No data found in time span. Ending.")
    exit(-1)

query2 = "SELECT timestamp,power_spectrum_channel_one,digitizer_log_id,integration_time from digitizer_log WHERE timestamp < '"+str((records[len(records)-1][0] + timedelta(days=0.25)).isoformat())+"' AND timestamp > '"+str((records[0][0] - timedelta(days=0.25)).isoformat())+"' AND notes='probe_snri_baseline' ORDER BY timestamp asc LIMIT "+str(max_lines)

records2 = db.send_admxdb_query(query2)

reflection_times, f0_vals = db.get_sensor_values("channel1_reflection_JPAon_f0", args.start_date, args.end_date)
reflection_times, q_vals = db.get_sensor_values("channel1_reflection_JPAon_Q", args.start_date, args.end_date)

print(f"Found {len(records)} files.")

print("Processing data...")

idxs = np.arange(0, len(records), 1)
data = np.zeros((files_per_write,53), dtype=int)
eggs_since_write = 0
total_drifts = 0

for i in tqdm(idxs):
    
    if eggs_since_write >= files_per_write:
        if len(data) > 0:
            with open(target_path, 'ab') as filehandle:
                    np.savetxt(filehandle, data, fmt="%i "*53)
        total_drifts += len(data)
        eggs_since_write = 0
                
    start_f0 = f0_vals[searchId(records[i][0], reflection_times) - 1]
    end_f0 = f0_vals[searchId(records[i][0], reflection_times)]
    if(abs(start_f0 - end_f0) > 5000):
        if is_trackable(records[i][1], records[i][0], hires_directory, records2):
            has_shift, shifts = get_shifts(records[i][1], records[i][0], hires_directory, records2, threshold=5)
            if has_shift:
                usable = True
                for s in shifts:
                    if abs(s) > 40:
                        usable = False
                        data[eggs_since_write] = [records[i][1], 0, -1, *shifts] # "Shift greater than 40 kHz"
                        break
                if usable:
                    data[eggs_since_write] = [records[i][1], 1, 0, *shifts]
            else:
                data[eggs_since_write] = [records[i][1], 1, 0, *[0]*50]
        else:
            data[eggs_since_write] = [records[i][1], 0, -2, *[0]*50] # "Can't track resonance"
        eggs_since_write += 1
                
print(f"Found {total_drifts} rod shifts.")
