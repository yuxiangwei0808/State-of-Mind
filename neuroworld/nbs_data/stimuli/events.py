"""The event tables that the stimulus export reads (DATA.md, step 2), built from the downloaded datasets.

`stimuli.materialize --export-raw-only` runs TRIBE v2's feature extractors over an event table: rows for each recording's
video, its audio and, on Algonauts, its transcribed words, on the recording's clock. This module builds that table the
way the paper's was built:

  algonauts   TRIBE v2's Algonauts 2025 study (tribev2/studies/algonauts2025.py at the pinned commit): for each of the
              four participants and each recording with a transcript, one Video event, the words of the transcript's
              per-TR lists and one Text event, on timelines <subject>:<task>:<stimulus>. Then TRIBE v2's event transforms
              (tribev2/demo_utils.py, get_audio_and_text_events): each video's audio written next to it as .wav, audio
              and video cut into chunks of 30-60 s, and each word given its sentence and the text before it.
  hcp_movies  the four movie runs in both released versions (Pre_ and Post_20140821_version, each listed with the
              participants who saw it): one Video event per clip at its place in the run's .mp4 (CLIPS, from HCP's clip
              timing table), its audio cut from the .mp4 with ffmpeg, and the same 30-60-s chunks. HCP's features read
              video and audio only, so no words are added.

    python -m neuroworld.nbs_data.stimuli.events algonauts --source-root $DATA/algonauts/dataset --output <events.tsv>
    python -m neuroworld.nbs_data.stimuli.events hcp_movies --source-root $DATA/hcp_movies/dataset --output <events.tsv> \
        --audio-dir <folder for the clips' .wav files>

Rebuilt from the downloads, the rows of a Friends episode, a repeated Movie10 film part and two HCP runs equal the tables
the paper used. The audio can differ by decoder rounding: one 16-bit step in 0.006% of the Friends episode's samples;
7 of the 9 HCP clips identical, and in the other two a few samples or a sub-sample offset. Needs tribev2
(requirements-data.txt) and, for HCP, ffmpeg.
"""
import argparse
import ast
from itertools import product
import math
from pathlib import Path
import re
import shutil
import subprocess
import wave

from neuroworld.data.algonauts_contract import AlgonautsIdentity, algonauts_stimulus_paths, canonical_algonauts_raw_timeline

ALGONAUTS_SUBJECTS = ("sub-01", "sub-02", "sub-03", "sub-05")
# HCP's clip timing table (HCP_7T_Movie_Clip_Sources+Frames+Timing.xlsx), the same in both versions: each run's clips
# as (block, start, duration) in seconds of the run's movie; the last clip of every run is the repeated test clip.
CLIPS = {
    "7T_MOVIE1_CC1": (("1.1", 20.0, 244.0417), ("1.2", 284.0833, 221.625), ("1.3", 525.75, 188.0), ("1.4", 733.7917, 63.75),
                      ("1.5", 817.5833, 83.375)),
    "7T_MOVIE2_HO1": (("2.1", 20.0, 226.7083), ("2.2", 266.75, 258.5833), ("2.3", 545.375, 249.2083), ("2.4", 814.5833, 83.375)),
    "7T_MOVIE3_CC2": (("3.1", 20.0, 180.5417), ("3.2", 220.5833, 184.5), ("3.3", 425.125, 204.0833), ("3.4", 649.25, 142.5),
                      ("3.5", 811.5833, 83.375)),
    "7T_MOVIE4_HO2": (("4.1", 20.0, 232.2917), ("4.2", 272.3333, 229.8333), ("4.3", 522.2083, 255.1667), ("4.4", 797.5833, 83.375)),
}
HCP_VERSIONS = {"pre": ("Pre_20140821_version", ""), "post": ("Post_20140821_version", "_v2")}
HCP_FPS, AUDIO_HZ = 24.0, 44100


def algonauts_recordings(root):
    """(subject, identity) of every Algonauts recording with a transcript, in TRIBE v2's order and with its skips."""
    for subject in ALGONAUTS_SUBJECTS:
        for season, episode, part in product(range(1, 8), range(1, 26), "abcd"):
            identity = AlgonautsIdentity("friends", f"s{season:02d}e{episode:02d}{part}", 0)
            if not identity.is_tribev2_skip and algonauts_stimulus_paths(root, identity)[1].is_file():
                yield subject, identity
        for movie, part, run in product(("bourne", "wolf", "life", "figures"), range(1, 18), (1, 2)):
            identity = AlgonautsIdentity("movie10", f"{movie}{part:02d}", run)
            if (movie in ("life", "figures") or run == 1) and algonauts_stimulus_paths(root, identity)[1].is_file():
                yield subject, identity


def algonauts_events(root, subject, identity):
    """One recording's events before TRIBE v2's transforms (its Algonauts2025._load_timeline_events, without fMRI)."""
    import pandas as pd

    video, transcript = algonauts_stimulus_paths(root, identity)
    words = []
    for _, row in pd.read_csv(transcript, sep="\t").iterrows():
        for text, start, duration in zip(*(ast.literal_eval(row[k]) for k in ("words_per_tr", "onsets_per_tr", "durations_per_tr"))):
            words.append({"type": "Word", "text": text, "start": start, "duration": duration, "stop": start + duration,
                          "language": "english"})
    rows = [{"type": "Video", "filepath": str(video), "start": 0}]
    if words:
        w = pd.DataFrame(words)
        rows.append({"type": "Text", "text": " ".join(w["text"]), "start": w["start"].min(),
                     "duration": w["stop"].max() - w["start"].min(), "stop": w["stop"].max(), "language": "english"})
    events = pd.DataFrame(rows + words)
    events["split"] = "test" if identity.content.startswith("s07") else "train"  # Friends S7 has no released fMRI
    events.loc[events["type"].isin(["Word", "Text"]), "modality"] = "heard"
    timeline = canonical_algonauts_raw_timeline(subject, identity)
    if identity.task == "friends":
        movie, part = identity.content[:3], identity.content[3:]
    else:
        movie, part = re.fullmatch(r"([a-z]+)(\d+)", identity.content).groups()
        part = str(int(part))
    return events.assign(timeline=timeline, subject=subject, task=identity.task, movie=movie, chunk=part,
                         run=identity.run, stimulus=timeline.split(":", 2)[2])


def cut_audio(mp4, start, duration, out):
    """A clip's audio as 44.1-kHz 16-bit stereo .wav (as TRIBE v2 writes audio), floor(duration x 44.1 kHz) samples."""
    if out.is_file():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    tmp = out.with_name(out.stem + ".tmp.wav")
    subprocess.run([ffmpeg, "-v", "error", "-y", "-ss", f"{start}", "-i", str(mp4), "-t", f"{duration}", "-vn", "-ac", "2",
                    "-ar", str(AUDIO_HZ), "-acodec", "pcm_s16le", str(tmp)], check=True)
    with wave.open(str(tmp)) as r:
        params, frames = r.getparams(), r.readframes(math.floor(duration * AUDIO_HZ))
    with wave.open(str(out), "wb") as w:
        w.setparams(params)
        w.writeframes(frames)
    tmp.unlink()
    return out


def wav_seconds(path):
    with wave.open(str(path)) as r:
        return r.getnframes() / r.getframerate()


def hcp_events(root, audio_dir, timelines=None):
    """Every clip's Video and Audio events, cut into 30-60-s chunks with TRIBE v2's transform."""
    import pandas as pd
    from neuralset.events.transforms import ChunkEvents

    rows = []
    for version, (folder, suffix) in HCP_VERSIONS.items():
        for movie, clips in CLIPS.items():
            timeline = f"hcp7t:{version}:{movie}"
            if timelines and timeline not in timelines:
                continue
            relative = f"movie_stimulus/movie_stimulus_2025/{folder}/{movie}{suffix}.mp4"
            mp4 = Path(root) / relative
            for block, start, duration in clips:
                wav = cut_audio(mp4, start, duration, Path(audio_dir) / f"{movie}{suffix}_{block}.wav")
                common = {"study": "HCP7TMovies", "subject": "stimulus", "timeline": timeline, "split": version, "start": start,
                          "stimulus": f"{movie}{suffix}:{block}", "stim_file": relative, "version": version, "movie": movie,
                          "block": block, "trial_type": "movie"}
                rows.append({**common, "type": "Video", "filepath": str(mp4), "frequency": HCP_FPS, "duration": duration,
                             "stop": start + duration, "offset": start})
                seconds = wav_seconds(wav)
                rows.append({**common, "type": "Audio", "filepath": str(wav.resolve()), "frequency": float(AUDIO_HZ),
                             "duration": seconds, "stop": start + seconds, "offset": 0.0})
    events = pd.DataFrame(rows)
    for kind in ("Audio", "Video"):
        events = ChunkEvents(event_type_to_chunk=kind, min_duration=30, max_duration=60)(events)
    return events.sort_values(["timeline", "start", "type"]).reset_index(drop=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=("algonauts", "hcp_movies"))
    p.add_argument("--source-root", type=Path, required=True, help="the downloaded dataset (DATA.md, Sources)")
    p.add_argument("--output", type=Path, required=True, help="the event table to write (.tsv)")
    p.add_argument("--audio-dir", type=Path, help="HCP: where to write the clips' audio (default: <output folder>/audio_clips)")
    p.add_argument("--timelines", nargs="+", help="only these timelines (for a quick check)")
    a = p.parse_args(argv)
    import pandas as pd

    if a.dataset == "algonauts":
        from tribev2.demo_utils import get_audio_and_text_events

        parts = []
        for subject, identity in algonauts_recordings(a.source_root):
            if a.timelines and canonical_algonauts_raw_timeline(subject, identity) not in a.timelines:
                continue
            parts.append(get_audio_and_text_events(algonauts_events(a.source_root, subject, identity), audio_only=False))
            print(f"{len(parts)} recordings: {parts[-1]['timeline'].iloc[0]}", flush=True)
        events = pd.concat(parts, ignore_index=True)
    else:
        events = hcp_events(a.source_root, a.audio_dir or a.output.parent / "audio_clips", a.timelines)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_name(a.output.name + ".tmp")
    events.to_csv(tmp, sep="\t", index=False)
    tmp.replace(a.output)
    print(f"{a.output}: {len(events)} events, {events['timeline'].nunique()} timelines")


if __name__ == "__main__":
    main()
