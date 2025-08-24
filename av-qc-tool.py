import json
import subprocess
import sys
import os
import re
import concurrent.futures
import requests
import hashlib
import datetime
from collections import defaultdict

# ===== CONFIGURATION =====
STATUS_UPDATE_URL = "http://127.0.0.1:8888/update_status"
MAX_WORKERS = 8
FFPROBE_THREADS = 16

# Thresholds
AV_SYNC_THRESHOLD = 0.033  # seconds, Stricter when: Lower
BLACK_FRAME_THRESHOLD = 0.017  # minimum duration to detect (seconds), Stricter when: Lower
BLACK_PIXEL_THRESHOLD = 0.98  # percentage of black pixels, Stricter when: Higher
PRORES_BITRATE_THRESHOLD = 242  # Mbps for ProRes 422 HQ, Stricter when: Lower
PHASE_WARNING_THRESHOLD = 0.9  # Adjust based on how strict you want, Stricter when: Higher
CLIPPING_THRESHOLD = -1.0  # dBFS: close to 0 means near or at clipping, Stricter when: Higher (closer to 0)
CLIPPING_DURATION_THRESHOLD = 0.005  # seconds (minimum duration to consider it a clipping event), Stricter when: Lower

# ===== MAIN SCRIPT =====
def report_status(file, status):
    try:
        requests.post(STATUS_UPDATE_URL, json={"file": file, "status": status})
    except requests.exceptions.RequestException as e:
        print(f"Error reporting status: {e}")

# Clear screen
os.system('cls' if os.name == 'nt' else 'clear')

if len(sys.argv) != 2:
    print("Usage: python script.py <video_file>")
    sys.exit(1)

video_file = sys.argv[1]
video_filename = os.path.basename(video_file)

print("Media Information Reporting & Analysis")
print("\n------------------------------------")

# Get basic metadata
cmd_metadata = [
    "ffprobe", "-v", "quiet", "-threads", str(FFPROBE_THREADS), 
    "-print_format", "json", "-show_format", "-show_streams", video_file
]
result_metadata = subprocess.run(cmd_metadata, stdout=subprocess.PIPE, 
                               stderr=subprocess.DEVNULL, text=True)
metadata = json.loads(result_metadata.stdout)

video_stream = next((s for s in metadata.get("streams", []) if s["codec_type"] == "video"), None)
audio_streams = [s for s in metadata.get("streams", []) if s["codec_type"] == "audio"]

if not video_stream:
    report_status(video_filename, "Error: No video stream found.")
    sys.exit(1)

def get_scantype(video_file):
    command = [
        "ffprobe", "-threads", str(FFPROBE_THREADS), "-v", "error",
        "-select_streams", "v:0", "-show_entries", "stream=field_order",
        "-of", "default=noprint_wrappers=1:nokey=1", video_file
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    scan_type = result.stdout.decode("utf-8").strip()
    if not scan_type:
        return "Unknown"
    if scan_type == "progressive":
        return "Progressive"
    elif scan_type in ["tt", "bb"]:
        return "Interlaced"
    else:
        return "Unknown"

# Extract basic info
encoder = video_stream.get("tags", {}).get("encoder", "Unknown Encoder")
width = video_stream.get("width", 0)
height = video_stream.get("height", 0)
duration = float(metadata.get("format", {}).get("duration", 0))
frame_rate = eval(video_stream.get("r_frame_rate", "0/1")) if "r_frame_rate" in video_stream else 0
scantype = get_scantype(video_file)

audio_codec = audio_streams[0].get("codec_name", "Unknown") if audio_streams else "No Audio"
if audio_codec == "pcm_s24le":
    audio_codec = "24-bit Linear PCM"
sample_rate = int(audio_streams[0].get("sample_rate", 0)) if audio_streams else "N/A"
channel_counts = [s.get("channels", 0) for s in audio_streams] if audio_streams else []
total_channels = sum(channel_counts)
channel_breakdown = " x ".join(map(str, channel_counts)) if channel_counts else "N/A"
audio_bitrate = sum(int(s.get("bit_rate", 0)) for s in audio_streams) / 1e3 if audio_streams else "N/A"

# Analyze frames and audio packets
report_status(video_filename, "Analyzing Video...")
cmd_frames = [
    "ffprobe", "-show_packets", "-threads", str(FFPROBE_THREADS),
    "-select_streams", "v", "-print_format", "json", video_file
]
result_frames = subprocess.run(cmd_frames, stdout=subprocess.PIPE, 
                             stderr=subprocess.DEVNULL, text=True)
frames = json.loads(result_frames.stdout).get("packets", [])

report_status(video_filename, "Analyzing Audio...")
cmd_audio = [
    "ffprobe", "-show_packets", "-threads", str(FFPROBE_THREADS),
    "-select_streams", "a", "-print_format", "json", video_file
]
result_audio = subprocess.run(cmd_audio, stdout=subprocess.PIPE, 
                            stderr=subprocess.DEVNULL, text=True)
audio_packets = json.loads(result_audio.stdout).get("packets", [])

if not frames:
    report_status(video_filename, "Error: No frame data found. Check if the video file is valid.")
    sys.exit(1)

if not audio_packets:
    report_status(video_filename, "Error: No audio data found.")
    sys.exit(1)

# Process bitrate information
bitrates = []
bitrates_first_5_sec = []

def process_frame(frame):
    pkt_size = int(frame.get("size", 0))
    pts_time = float(frame.get("pts_time", 0))
    duration_time = float(frame.get("duration_time", 0) or 0.0333)
    frame_rate = 1 / duration_time if duration_time > 0 else 30
    frame_bitrate_mbps = (pkt_size * 8 * frame_rate) / 1e6
    return (pts_time, frame_bitrate_mbps)

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    results = list(executor.map(process_frame, frames))

for pts_time, frame_bitrate_mbps in results:
    bitrates.append(frame_bitrate_mbps)
    if pts_time <= 5.0:
        bitrates_first_5_sec.append(frame_bitrate_mbps)

average_bitrate = sum(bitrates) / len(bitrates) if bitrates else 0
peak_bitrate = max(bitrates) if bitrates else 0
peak_bitrate_first_5_sec = max(bitrates_first_5_sec) if bitrates_first_5_sec else 0

def seconds_to_timecode(seconds, frame_rate):
    """Convert seconds to timecode (HH:MM:SS:FF) format"""
    if not frame_rate or frame_rate <= 0:
        return f"{seconds:.2f}s"  # Fallback to seconds if no valid frame rate
    
    frames_total = round(seconds * frame_rate)
    frames = frames_total % round(frame_rate)
    seconds_total = frames_total // round(frame_rate)
    seconds = seconds_total % 60
    minutes_total = seconds_total // 60
    minutes = minutes_total % 60
    hours = minutes_total // 60
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"

# Closed caption detection
def get_closed_caption_info(video_file):
    command = [
        "ffprobe", "-threads", str(FFPROBE_THREADS), "-v", "error",
        "-select_streams", "s", "-show_entries", "stream=index,codec_name,codec_long_name,language",
        "-of", "default=noprint_wrappers=1", video_file
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    caption_info = result.stdout.decode("utf-8").strip()
    if caption_info:
        caption_lines = caption_info.split("\n")
        closed_caption_streams = [
            line for line in caption_lines if any(cc_format in line for cc_format in ['mov_text', 'scc', 'cc1', 'cc2'])
        ]
        return f"Embedded Closed Caption Info:\n" + "\n".join(closed_caption_streams) if closed_caption_streams else "No Closed Captions Detected"
    else:
        return "No Closed Captions Detected"

caption_info = get_closed_caption_info(video_file)

# Improved A/V sync detection
def analyze_av_sync(frames, audio_packets):
    video_pts = [float(frame.get("pts_time", 0)) for frame in frames]
    audio_pts = [float(pkt.get("pts_time", 0)) for pkt in audio_packets]
    
    if not video_pts or not audio_pts:
        return {"offset": "N/A", "details": "Could not calculate A/V sync"}
    
    # Calculate initial offset
    min_video_pts = min(video_pts)
    min_audio_pts = min(audio_pts)
    initial_offset = min_audio_pts - min_video_pts
    
    # Calculate drift over time
    video_duration = max(video_pts) - min_video_pts
    audio_duration = max(audio_pts) - min_audio_pts
    duration_diff = audio_duration - video_duration
    
    # Calculate sync points throughout the file
    sync_points = []
    video_intervals = len(video_pts) // 10  # Check 10 points
    audio_intervals = len(audio_pts) // 10
    
    for i in range(1, 10):
        v_idx = i * video_intervals
        a_idx = i * audio_intervals
        if v_idx < len(video_pts) and a_idx < len(audio_pts):
            sync_diff = audio_pts[a_idx] - video_pts[v_idx]
            sync_points.append((video_pts[v_idx], sync_diff))
    
    return {
        "initial_offset": initial_offset,
        "duration_diff": duration_diff,
        "sync_points": sync_points,
        "details": f"Initial offset: {initial_offset:.3f}s, Duration difference: {duration_diff:.3f}s"
    }

av_sync_info = analyze_av_sync(frames, audio_packets)

# Black frame detection with timestamps
def detect_black_frames(video_file):
    cmd = [
        "ffmpeg", "-i", video_file, "-vf", 
        f"blackdetect=d={BLACK_FRAME_THRESHOLD}:pic_th={BLACK_PIXEL_THRESHOLD}", 
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    black_segments = []
    for line in result.stderr.split("\n"):
        if "black_start" in line:
            try:
                # Extract the relevant parts of the line
                parts = line.split()
                # Find the start, end, and duration parts
                start_part = next(p for p in parts if p.startswith("black_start:"))
                end_part = next(p for p in parts if p.startswith("black_end:"))
                duration_part = next(p for p in parts if p.startswith("black_duration:"))
                
                # Extract the numeric values
                start = float(start_part.split(":")[1])
                end = float(end_part.split(":")[1])
                duration = float(duration_part.split(":")[1])
                
                black_segments.append({
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "position": f"{start:.2f}-{end:.2f}s"
                })
            except (ValueError, IndexError, StopIteration) as e:
                print(f"Warning: Could not parse black frame detection line: {line}")
                continue
    
    return black_segments

black_frames = detect_black_frames(video_file)

def detect_audio_phase(video_file):
    cmd = [
        "ffmpeg", "-i", video_file,
        "-filter_complex", "[0:a]aphasemeter=video=0:metadata=1[out]",
        "-map", "[out]",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    phase_info = {
        "out_of_phase": False,
        "max_phase_diff": 0,
        "problematic_segments": []
    }

    # Combine stdout and stderr as FFmpeg might write to either
    output = result.stdout + result.stderr
    current_segment = None
    max_diff = 0

    for line in output.split('\n'):
        if "phase:" in line:
            try:
                # Extract time and phase difference
                parts = line.split()
                time_part = next(p for p in parts if p.startswith("t:") or p.startswith("pts:"))
                phase_part = next(p for p in parts if p.startswith("phase:"))
                
                time = float(time_part.split(':')[1])
                phase_diff = abs(float(phase_part.split(':')[1]))
                
                # Track maximum phase difference
                if phase_diff > max_diff:
                    max_diff = phase_diff
                
                # Detect problematic segments
                if phase_diff > PHASE_WARNING_THRESHOLD:
                    if current_segment is None:
                        current_segment = {
                            'start': time,
                            'end': time,
                            'max_diff': phase_diff
                        }
                    else:
                        current_segment['end'] = time
                        if phase_diff > current_segment['max_diff']:
                            current_segment['max_diff'] = phase_diff
                else:
                    if current_segment is not None:
                        if (current_segment['end'] - current_segment['start']) >= 0.1:  # Minimum segment duration
                            phase_info['problematic_segments'].append(current_segment)
                        current_segment = None
            except (ValueError, StopIteration):
                continue

    # Add the last segment if it exists
    if current_segment is not None:
        if (current_segment['end'] - current_segment['start']) >= 0.1:
            phase_info['problematic_segments'].append(current_segment)

    phase_info['max_phase_diff'] = max_diff
    phase_info['out_of_phase'] = max_diff > PHASE_WARNING_THRESHOLD

    return phase_info

phase_info = detect_audio_phase(video_file)

def detect_audio_clipping(video_file):
    cmd = [
        "ffmpeg", "-i", video_file,
        "-af", "astats=metadata=1:reset=1", 
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    clipping_events = []
    current_clip = None
    output = result.stderr  # astats typically writes to stderr

    for line in output.split('\n'):
        if "Peak level dB" in line or "Peak level:" in line:
            try:
                # Extract time and peak level
                time_match = re.search(r't:\s*(\d+\.\d+)', line)
                peak_match = re.search(r'Peak level:\s*(-?\d+\.\d+)', line)
                
                if not peak_match:
                    continue
                    
                time = float(time_match.group(1)) if time_match else 0.0
                peak_level = float(peak_match.group(1))
                
                # Detect clipping
                if peak_level >= CLIPPING_THRESHOLD:
                    if current_clip is None:
                        current_clip = {
                            'start_time': time,
                            'end_time': time,
                            'max_peak': peak_level
                        }
                    else:
                        current_clip['end_time'] = time
                        if peak_level > current_clip['max_peak']:
                            current_clip['max_peak'] = peak_level
                else:
                    if current_clip is not None:
                        duration = current_clip['end_time'] - current_clip['start_time']
                        if duration >= CLIPPING_DURATION_THRESHOLD:
                            clipping_events.append(current_clip)
                        current_clip = None
            except (ValueError, AttributeError):
                continue

    # Add the last clip if it exists
    if current_clip is not None:
        duration = current_clip['end_time'] - current_clip['start_time']
        if duration >= CLIPPING_DURATION_THRESHOLD:
            clipping_events.append(current_clip)

    return clipping_events

clipping_events = detect_audio_clipping(video_file)

def generate_clean_report(metadata, video_file, av_sync_info, black_frames, 
                         phase_info, clipping_events, frame_rate, 
                         average_bitrate, peak_bitrate, peak_bitrate_first_5_sec):
    # Extract basic info
    general = metadata.get('format', {})
    video = next((s for s in metadata.get('streams', []) if s['codec_type'] == 'video'), {})
    audio = next((s for s in metadata.get('streams', []) if s['codec_type'] == 'audio'), {})
    
    # Format duration
    duration = float(general.get('duration', 0))
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_duration = f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"
    
    # Convert seconds to timecode with proper type handling
    def sec_to_tc(seconds, frame_rate):
        if frame_rate <= 0:
            return f"{seconds:.2f}s"  # Fallback if invalid frame rate
            
        total_frames = round(float(seconds) * float(frame_rate))
        frames = total_frames % int(round(frame_rate))
        seconds_total = total_frames // int(round(frame_rate))
        seconds = seconds_total % 60
        minutes_total = seconds_total // 60
        minutes = minutes_total % 60
        hours = minutes_total // 60
        
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}:{int(frames):02d}"
    
    # Generate report
    report = f"""
       Media Information Reporting & Analysis           
───────────────────────────────────────────────────

■ FILE METADATA
───────────────────────────────────────────────────
• Filename:       {os.path.basename(video_file)}
• Container:      {general.get('format_name', 'N/A')}
• Size:           {float(general.get('size', 0))/(1024**3):.2f} GB
• Duration:       {formatted_duration} ({frame_rate:.2f} fps)
• Creation Date:  {general.get('tags', {}).get('creation_time', 'N/A')}

■ VIDEO STREAM
───────────────────────────────────────────────────
• Codec:          {video.get('codec_long_name', video.get('codec_name', 'N/A'))}
• Profile:        {video.get('profile', 'N/A')}
• Resolution:     {video.get('width', 'N/A')}x{video.get('height', 'N/A')}
• Frame Rate:     {eval(video.get('r_frame_rate', '0/1')):.3f} fps
• Bitrate:        {int(video.get('bit_rate', 0))/1e6:.2f} Mbps (avg)
• Scan Type:      {video.get('field_order', 'progressive').title()}

■ BITRATE ANALYSIS
───────────────────────────────────────────────────
• Average Bitrate: {average_bitrate:.2f} Mbps
• Peak Bitrate:    {peak_bitrate:.2f} Mbps
• 5-sec Peak:      {peak_bitrate_first_5_sec:.2f} Mbps
"""

    # Add ProRes bitrate warning if applicable
    if "ProRes" in video.get('codec_long_name', ''):
        report += f"• ProRes Status:  {'⚠️ Exceeds broadcast limit!' if peak_bitrate_first_5_sec > PRORES_BITRATE_THRESHOLD else '✅ Within spec'}\n"
        report += f"  (Threshold: {PRORES_BITRATE_THRESHOLD} Mbps)\n"

    audio_codec = audio.get('codec_long_name', audio.get('codec_name', 'N/A'))
    if audio_codec == 'pcm_s24le':
        audio_codec = '24-bit Linear PCM'
    
    channels = audio.get('channels', 'N/A')
    channel_layout = audio.get('channel_layout', '')
    
    # Handle sample rate conversion safely
    sample_rate = audio.get('sample_rate', '0')
    try:
        sample_rate_khz = f"{int(sample_rate)/1000:.1f} kHz"
    except (ValueError, TypeError):
        sample_rate_khz = 'N/A'
    
    # Handle bitrate conversion safely
    audio_bitrate = audio.get('bit_rate', '0')
    try:
        bitrate_kbps = f"{int(audio_bitrate)/1000:.1f} kbps"
    except (ValueError, TypeError):
        bitrate_kbps = 'N/A'

    report += f"""
■ AUDIO STREAM
───────────────────────────────────────────────────
• Codec:          {audio_codec}
• Channels:       {channels}{f" ({channel_layout})" if channel_layout else ""}
• Sample Rate:    {sample_rate_khz}
• Bitrate:        {bitrate_kbps}

■ A/V DISCREPANCY DETECTION
───────────────────────────────────────────────────
"""

    # [Rest of your existing discrepancy detection code...]
    # A/V Sync Analysis
    av_status = (abs(av_sync_info.get('initial_offset', 0)) <= 0.1 and 
                 abs(av_sync_info.get('duration_diff', 0)) <= 0.1)
    report += f"• A/V Sync:       {'✅ Within tolerance' if av_status else '⚠️ Out of sync'}\n"
    if not av_status:
        report += f"  - Initial offset: {av_sync_info.get('initial_offset', 0):.3f}s\n"
        report += f"  - Drift: {av_sync_info.get('duration_diff', 0):.3f}s\n"
        for point in av_sync_info.get('sync_points', [])[:3]:
            report += f"  - At {sec_to_tc(float(point[0]), float(frame_rate))}: {point[1]:.3f}s offset\n"

    # Black Frame Detection
    if black_frames:
        report += f"• Black Frames:   ⚠️ Detected {len(black_frames)} segments\n"
        for seg in black_frames[:3]:
            report += (f"  - {sec_to_tc(float(seg['start']), float(frame_rate))} to "
                     f"{sec_to_tc(float(seg['end']), float(frame_rate))} "
                     f"({float(seg['duration']):.2f}s)\n")
    else:
        report += "• Black Frames:   ✅ None detected\n"

    # Phase Analysis
    if phase_info["out_of_phase"]:
        report += (f"• Audio Phase:    ⚠️ Out of phase "
                  f"(max diff: {float(phase_info['max_phase_diff']):.2f})\n")
        for seg in phase_info["problematic_segments"][:3]:
            report += (f"  - {sec_to_tc(float(seg['start']), float(frame_rate))} to "
                     f"{sec_to_tc(float(seg['end']), float(frame_rate))}\n")
    else:
        report += (f"• Audio Phase:    ✅ Normal "
                  f"(max diff: {float(phase_info['max_phase_diff']):.2f})\n")

    # Clipping Detection
    if clipping_events:
        report += f"• Audio Clipping: ⚠️ Detected {len(clipping_events)} events\n"
        for clip in clipping_events[:3]:
            report += (f"  - {sec_to_tc(float(clip['start_time']), float(frame_rate))} "
                     f"(peak: {float(clip['max_peak']):.1f} dBFS)\n")
    else:
        report += "• Audio Clipping: ✅ None detected\n"

    report += """
                 END OF REPORT                   
───────────────────────────────────────────────────
"""
    return report

# Update your function call to include the new parameters:
professional_report = generate_clean_report(
    metadata=metadata,
    video_file=video_file,
    av_sync_info=av_sync_info,
    black_frames=black_frames,
    phase_info=phase_info,
    clipping_events=clipping_events,
    frame_rate=frame_rate,
    average_bitrate=average_bitrate,
    peak_bitrate=peak_bitrate,
    peak_bitrate_first_5_sec=peak_bitrate_first_5_sec
)
report_status(video_filename, professional_report.replace("\n", "<br>"))
# Save to a text file
report_filename = os.path.splitext(video_file)[0] + "_QC_Report.txt"
with open(report_filename, 'w', encoding='utf-8') as f:
    f.write(professional_report)
print(f"\nReport also saved to: {report_filename}")