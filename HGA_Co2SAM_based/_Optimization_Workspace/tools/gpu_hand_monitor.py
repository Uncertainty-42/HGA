import subprocess, time
peaks = {}
total_range=1000
interval=0.01
gpu_id = 5
for i in range(total_range):  # 20 times × 0.5s = 10s
    out = subprocess.check_output(
        ['nvidia-smi', f'--id={gpu_id}', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader'],
        stderr=subprocess.DEVNULL, text=True
    )
    for line in out.strip().split('\n'):
        if line:
            pid, mem = line.split(',')
            pid, mem = int(pid), int(mem[:-4])
            if peaks.get(pid, 0) < mem:
                peaks[pid] = mem
    print(f'\r{i+1}/{total_range}: ' + ', '.join(f'{p}:{peaks[p]}MB' for p in peaks), end='')
    time.sleep(interval)
print('\nPeak:')
for p, m in peaks.items():
    print(f'  PID {p} → {m} MB')