import logging
import os
import subprocess
from typing import Dict, Tuple, List, Union

import numpy as np
from moviepy.audio.AudioClip import AudioArrayClip, CompositeAudioClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.VideoClip import ColorClip
from moviepy import VideoFileClip, concatenate_videoclips

import warnings
warnings.simplefilter("ignore")


from mtslinker.downloader import download_video_chunks

TARGET_FPS = 30
TARGET_AUDIO_FPS = 44100


def _set_clip_fps(clip, fps):
    """Set a clip's sampling rate using the API available in MoviePy 1 or 2."""
    if hasattr(clip, 'with_fps'):
        return clip.with_fps(fps)
    return clip.set_fps(fps)


def process_video_clips(
    directory: str,
    json_data: Dict,
    download_workers: int = 8,
) -> Tuple[float, List[VideoFileClip], List[AudioFileClip]]:
    total_duration = float(json_data.get('duration', 0))
    if not total_duration:
        raise ValueError('Duration not found in JSON data.')

    video_clips = []
    audio_clips = []
    events = []

    for event in json_data.get('eventLogs', []):
        if isinstance(event, dict):
            data = event.get('data', {})
            if isinstance(data, dict) and 'url' in data:
                url = data['url']
                start_time = event.get('relativeTime', 0)
                events.append((url, start_time))

    downloaded_paths = download_video_chunks(
        (url for url, _ in events), directory, max_workers=download_workers
    )
    for downloaded_file_path, (_, start_time) in zip(downloaded_paths, events):
        try:
            video_clip = _set_clip_fps(VideoFileClip(downloaded_file_path, fps_source='fps'), TARGET_FPS)
            video_clip = video_clip.with_start(start_time)
            video_clips.append(video_clip)
        except (KeyError, OSError):
            audio_clip = _set_clip_fps(AudioFileClip(downloaded_file_path), TARGET_AUDIO_FPS)
            audio_clip = audio_clip.with_start(start_time)
            audio_clips.append(audio_clip)
    logging.info(f'Total duration of clips: {total_duration}')

    return total_duration, video_clips, audio_clips


def create_video_with_gaps(total_duration: float, video_clips: List[VideoFileClip]) -> VideoFileClip:
    clips = []
    current_time = 0.0

    for video in video_clips:
        if video.start > current_time:
            gap_duration = video.start - current_time
            if gap_duration > 0:
                empty_clip = ColorClip(size=(1920, 1080), color=(0, 0, 0), duration=gap_duration).with_start(
                    current_time)
                clips.append(empty_clip)

        clips.append(video)
        current_time = video.end

    if current_time < total_duration:
        remaining_duration = total_duration - current_time
        empty_clip = ColorClip(size=(1920, 1080), color=(0, 0, 0), duration=remaining_duration).with_start(current_time)
        clips.append(empty_clip)

    # ``compose`` composites every frame onto a canvas and is considerably
    # slower. All clips produced by MTS Link normally have the same size, so
    # use the direct frame chain in that case and retain compose as a safe
    # fallback for recordings with mixed resolutions.
    clip_sizes = {tuple(clip.size) for clip in clips}
    concat_method = 'chain' if len(clip_sizes) == 1 else 'compose'
    logging.info('Concatenating %d clips with method=%s.', len(clips), concat_method)
    final_video = concatenate_videoclips(clips, method=concat_method)
    logging.info(f'Final video duration: {final_video.duration}')
    return final_video


def create_audio_with_gaps(total_duration: float, audio_clips: List[AudioFileClip]) -> CompositeAudioClip:
    audio_segments = []
    current_time = 0

    for audio in audio_clips:
        if audio.start > current_time:
            gap_duration = audio.start - current_time
            if gap_duration > 0:
                silence_segment = AudioArrayClip(
                    np.zeros((int(gap_duration * TARGET_AUDIO_FPS), 2), dtype=np.float32),
                    fps=TARGET_AUDIO_FPS
                ).with_start(current_time)
                audio_segments.append(silence_segment)

        audio_segments.append(audio)
        current_time = audio.end

    if current_time < total_duration:
        remaining_duration = total_duration - current_time
        silence_segment = AudioArrayClip(
            np.zeros((int(remaining_duration * TARGET_AUDIO_FPS), 2), dtype=np.float32),
            fps=TARGET_AUDIO_FPS
        ).with_start(current_time)
        audio_segments.append(silence_segment)

    final_audio = CompositeAudioClip(audio_segments)
    logging.info(f'Total audio duration: {final_audio.duration}')
    return final_audio


def compile_final_video_ffmpeg(
    total_duration: float,
    video_clips: List[VideoFileClip],
    audio_clips: List[AudioFileClip],
    output_path: str,
) -> None:
    """Render the timeline in ffmpeg without a Python frame-generation loop."""
    if not video_clips:
        raise ValueError('No video clips available for ffmpeg rendering.')

    width, height = video_clips[0].size
    command = ['ffmpeg', '-y', '-loglevel', 'warning']
    for clip in [*video_clips, *audio_clips]:
        filename = getattr(clip, 'filename', None)
        if not filename:
            raise ValueError('A clip has no source filename for ffmpeg rendering.')
        command.extend(['-i', filename])

    filters = []
    video_labels = []
    current_time = 0.0
    for index, clip in enumerate(video_clips):
        start = float(clip.start or 0)
        gap = start - current_time
        if gap > 0:
            label = f'vgap{index}'
            filters.append(
                f'color=c=black:s={width}x{height}:r={TARGET_FPS}:d={gap},format=yuv420p[{label}]'
            )
            video_labels.append(f'[{label}]')
        label = f'v{index}'
        video_filter = f'fps={TARGET_FPS},format=yuv420p,setpts=PTS-STARTPTS'
        if tuple(clip.size) != (width, height):
            video_filter = (
                f'scale={width}:{height}:force_original_aspect_ratio=decrease,'
                f'pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,' + video_filter
            )
        filters.append(f'[{index}:v]{video_filter}[{label}]')
        video_labels.append(f'[{label}]')
        current_time = max(current_time, start) + float(clip.duration)

    if current_time < total_duration:
        label = 'vgap_end'
        filters.append(
            f'color=c=black:s={width}x{height}:r={TARGET_FPS}:d={total_duration - current_time},'
            f'format=yuv420p[{label}]'
        )
        video_labels.append(f'[{label}]')

    filters.append(''.join(video_labels) + f'concat=n={len(video_labels)}:v=1:a=0[vout]')
    output_options = ['-map', '[vout]']

    if audio_clips:
        audio_labels = []
        video_count = len(video_clips)
        for index, clip in enumerate(audio_clips):
            input_index = video_count + index
            delay_ms = max(0, round(float(clip.start or 0) * 1000))
            label = f'a{index}'
            filters.append(
                f'[{input_index}:a]aresample={TARGET_AUDIO_FPS},asetpts=PTS-STARTPTS,'
                f'adelay={delay_ms}|{delay_ms}[{label}]'
            )
            audio_labels.append(f'[{label}]')
        filters.append(
            ''.join(audio_labels) + f'amix=inputs={len(audio_labels)}:duration=longest:dropout_transition=0,'
            f'apad=whole_dur={total_duration},atrim=duration={total_duration},'
            f'aresample={TARGET_AUDIO_FPS}[aout]'
        )
        output_options.extend(['-map', '[aout]', '-c:a', 'aac', '-b:a', '192k'])
    else:
        output_options.append('-an')

    command.extend([
        '-filter_complex', ';'.join(filters),
        *output_options,
        '-c:v', 'libx264', '-preset', 'ultrafast', '-r', str(TARGET_FPS),
        '-threads', str(os.cpu_count() or 1), '-stats_period', '5', '-loglevel', 'info',
        '-t', str(total_duration), output_path,
    ])
    subprocess.run(command, check=True)


def compile_final_video(total_duration: float, video_clips: List[VideoFileClip], audio_clips: List[AudioFileClip],
                        output_path: str, max_duration: Union[int, None]):
    if max_duration:
        total_duration = min(total_duration, max_duration)

    try:
        compile_final_video_ffmpeg(total_duration, video_clips, audio_clips, output_path)
        logging.info('Final video rendered with ffmpeg.')
        return
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        logging.warning('ffmpeg fast path failed (%s); falling back to MoviePy.', error)

    video_result = create_video_with_gaps(total_duration, video_clips)

    if audio_clips:
        combined_audio = create_audio_with_gaps(total_duration, audio_clips)
        video_result = video_result.with_audio(combined_audio)

    if max_duration:
        if video_result.duration > max_duration:
            logging.info(f'Duration limit! Crop!')
            video_result = video_result.subclip(0, max_duration)

    video_result.write_videofile(
        output_path,
        codec='libx264',
        audio_codec='aac',
        fps=TARGET_FPS,
        preset='ultrafast',
        threads=os.cpu_count()
    )
