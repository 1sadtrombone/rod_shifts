from rod_shift_functions import get_shifts, is_trackable
import numpy as np
import argparse
from tqdm import tqdm
import os, sys
from datetime import datetime, timedelta
from npy_append_array import NpyAppendArray
import h5py

sys.path.insert(0,os.path.abspath('../../admx_analysis_tools/datatypes_and_database/'))
import admx_db_interface
sys.path.insert(0,os.path.abspath('../admx_analysis_tools/parameter_extraction/mainline_analysis/'))
from parameter_functions import searchId, smooth_parameter, convert_magnet_current_to_B_field

### maximum number of digitizations to pull, mostly a fail safe
max_lines=1000000


# Where the relevant egg files are kept
hires_directory='/pnfs/admx/persistent/high-res/'

parser = argparse.ArgumentParser()
parser.add_argument('param_fname', help='Path to product of parameter_extraction')
parser.add_argument('save_tag', help='Data saved to shift_results/rod_shifts_{save_tag}.h5')

args = parser.parse_args()

param_fname = args.param_fname
save_path = f"shift_results/rod_shifts_{args.save_tag}.h5"

print(f"Will save rod shifts to {save_path}")
if os.path.isfile(save_path): 
    print("Target file already exists! Delete it if you really want to overwrite.")
    exit(-1)

print(f"Reading cut information from {param_fname}")

with h5py.File(param_fname, 'r') as pf:

    digi_refs = np.array(pf['digi_ref'][:], dtype=int)
    cut_reasons = np.array(pf['cut_reason'][:], dtype=str)


start_id = np.min(digi_refs)
stop_id = np.max(digi_refs)

target_ids = digi_refs[np.where(cut_reasons == 'large f0 jump')]
N = len(target_ids)

db=admx_db_interface.ADMXDB()
db.hostname="admxdb01.fnal.gov"
db.dbname="admx"

id_margin = 0
target_times = []

print("Querying DB for metadata...")

while len(target_times) < N:

    query1 = "SELECT timestamp,digitizer_log_reference FROM axion_scan_log WHERE digitizer_log_reference < '"+str(stop_id+id_margin)+"' AND digitizer_log_reference >'"+str(start_id-id_margin)+"' ORDER BY timestamp asc LIMIT "+str(max_lines)

    records = db.send_admxdb_query(query1)
    records_arr = np.array(records)

    if len(records) == 0:
        print("No data found in time span. Ending.")
        exit(-1)

    target_times = records_arr[:,0][np.in1d(records_arr[:,1], target_ids)]

    id_margin += 100

query2 = "SELECT timestamp,power_spectrum_channel_one,digitizer_log_id,integration_time from digitizer_log WHERE timestamp < '"+str((records[len(records)-1][0] + timedelta(days=0.25)).isoformat())+"' AND timestamp > '"+str((records[0][0] - timedelta(days=0.25)).isoformat())+"' AND notes='probe_snri_baseline' ORDER BY timestamp asc LIMIT "+str(max_lines)

records2 = db.send_admxdb_query(query2)

print(f"Found {len(records)} files.")
print(f"There are {N} rod shifts.")

print("Processing data...")

is_usables = np.zeros(dtype=bool, shape=N)
all_reasons = np.zeros(dtype=str, shape=N)
all_shifts = np.zeros(dtype=float, shape=(N,50))

for i, digi_ref in tqdm(enumerate(target_ids), total=len(target_ids)):
                    
    if is_trackable(digi_ref, target_times[i], hires_directory, records2):
        has_shift, shifts = get_shifts(digi_ref, target_times[i], hires_directory, records2, threshold=5)
        if has_shift:
            usable = True
            reason = "usable"
            for s in shifts:
                if abs(s) > 40:
                    usable = False
                    reason = "shift greater than 40 kHz"
                    break
        else:
            # no shift
            usable = True
            reason = "no shift"
            shifts = [0]*50
    else:
        usable = False
        shifts = [0]*50
        reason = "cannot track resonance"

    is_usables[i] = usable
    all_reasons[i] = reason
    all_shifts[i] = shifts

print("Writing to target file...")

with h5py.File(save_path, 'w') as f:

    f.create_dataset("digi_ref", dtype=int, shape=N)
    f.create_dataset("is_usable", dtype=bool, shape=N)
    f.create_dataset("reason", dtype=h5py.string_dtype(), shape=N)
    f.create_dataset("shifts", dtype=float, shape=(N,50))


    f['digi_ref'][:] = target_ids
    f['is_usable'][:] = is_usables
    f['reason'][:] = all_reasons
    f['shifts'][:] = all_shifts

print('Done.')
