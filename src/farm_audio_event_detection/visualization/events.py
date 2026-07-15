import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Any
import json

def plot_audio_events(
    audio: np.ndarray,
    sample_rate: int,
    events: List[Any],
    output_path: str | Path,
    title: str = "Detected Events"
):
    """
    Plots the audio waveform overlaid with colored segments for each detected event.
    events: list of DetectedEvent objects (with event_start, event_end, animal)
    """
    plt.rcParams.update({
        "font.weight": "normal",
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "font.family": "sans-serif",
    })
    
    duration = len(audio) / sample_rate
    times = np.linspace(0, duration, num=len(audio))
    
    fig, ax = plt.subplots(figsize=(15, 4))
    
    # Plot waveform
    ax.plot(times, audio, color="gray", alpha=0.5, label="Waveform")
    
    # Define colors for different animals
    animal_colors = {
        "dog": "#1f77b4",     # blue
        "cat": "#ff7f0e",     # orange
        "cow": "#2ca02c",     # green
        "rooster": "#d62728", # red
        "sheep": "#9467bd",   # purple
        "others": "#7f7f7f"   # grey
    }
    
    plotted_labels = set()
    
    for event in events:
        color = animal_colors.get(event.animal, "#000000")
        label = event.animal if event.animal not in plotted_labels else ""
        if label:
            plotted_labels.add(event.animal)
            
        ax.axvspan(
            event.event_start, 
            event.event_end, 
            color=color, 
            alpha=0.3, 
            label=label
        )
        
        # Add text in the middle of the span
        mid_point = (event.event_start + event.event_end) / 2
        ax.text(
            mid_point, 
            np.max(audio) * 0.9, 
            event.animal, 
            color=color, 
            fontsize=10, 
            ha='center', 
            va='top', 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
        )
        
    ax.set_xlim(0, duration)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    if plotted_labels:
        ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

def export_events_to_json(events: List[Any], output_path: str | Path):
    """
    Exports a list of DetectedEvent objects to the requested JSON format.
    """
    report = [
        {
            "event_start": f"{event.event_start:.1f}",
            "event_end": f"{event.event_end:.1f}",
            "animal": event.animal
        }
        for event in events
    ]
    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)
