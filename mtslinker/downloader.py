import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, Iterable, List, Union

import httpx
import logging
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

TIMEOUT_SETTINGS = httpx.Timeout(None, connect=None)
DEFAULT_DOWNLOAD_WORKERS = 8


def construct_json_data_url(event_session_id: str, recording_id: str) -> str:
    if not event_session_id:
        raise ValueError('Missing webinar event session ID.')
    
    if not recording_id:
        return f'https://my.mts-link.ru/api/eventsessions/{event_session_id}/record?withoutCuts=false'
    return f'https://my.mts-link.ru/api/event-sessions/{event_session_id}/record-files/{recording_id}/flow?withoutCuts=false'


def fetch_json_data(url: str, session_id: Union[str, None]) -> Dict:
    cookies = {}
    if session_id:
        cookies['sessionId'] = session_id

    with httpx.Client(timeout=TIMEOUT_SETTINGS) as client:
        response = client.get(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0',
            },
            cookies=cookies
        )
        
    try:
        error_data = response.json()
        if error_data.get("error", {}).get("code") == 403:
            logging.error(
                'Access denied: session_id token is required. '
                'Provide it using the "--session-id" parameter.'
            )
            return
    except Exception:
        logging.warning('Server response does not contain JSON.')
            
    response.raise_for_status()
    return response.json()


def download_video_chunk(
    video_url: str,
    save_directory: str,
    progress_callback: Callable[[int, int], None] = None,
) -> str:
    filename = os.path.basename(video_url)
    file_path = os.path.join(save_directory, filename)

    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        logging.info('Using existing fragment: %s', filename)
        size = os.path.getsize(file_path)
        if progress_callback:
            progress_callback(size, size)
        return file_path

    logging.info('Downloading fragment: %s', filename)
    with open(file_path, 'wb') as file:
        with httpx.Client(timeout=TIMEOUT_SETTINGS) as client:
            with client.stream('GET', video_url) as response:
                response.raise_for_status()
                content_length = int(response.headers.get('content-length') or 0)
                if progress_callback and content_length:
                    progress_callback(content_length, 0)
                downloaded = 0
                for chunk in response.iter_bytes(chunk_size=8192):
                    file.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(0, len(chunk))
                if progress_callback and not content_length:
                    progress_callback(downloaded, 0)
    logging.info('Downloaded fragment: %s (%.1f MB)', filename, os.path.getsize(file_path) / 1024**2)
    return file_path


def download_video_chunks(
    video_urls: Iterable[str],
    save_directory: str,
    max_workers: int = DEFAULT_DOWNLOAD_WORKERS,
    progress_labels: Iterable[str] = None,
) -> List[str]:
    """Download independent recording fragments concurrently.

    ``executor.map`` keeps the result order identical to ``video_urls``. This
    matters because the processor applies each downloaded file's event time
    after the downloads complete.
    """
    urls = list(video_urls)
    if not urls:
        return []
    if max_workers < 1:
        raise ValueError('max_workers must be at least 1.')

    labels = list(progress_labels) if progress_labels is not None else ['Fragments'] * len(urls)
    if len(labels) != len(urls):
        raise ValueError('progress_labels must contain one label per video URL.')

    worker_count = min(max_workers, len(urls))
    logging.info('Downloading %d fragments with %d parallel workers.', len(urls), worker_count)
    label_order = list(dict.fromkeys(labels))
    fragment_totals = {label: labels.count(label) for label in label_order}
    fragment_done = {label: 0 for label in label_order}
    bars = {
        label: tqdm.tqdm(
            total=0,
            unit='B',
            unit_scale=True,
            desc=f'Downloading {label}',
            position=position,
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
        )
        for position, label in enumerate(label_order)
    }
    progress_lock = threading.Lock()

    def progress_callback(label):
        def update(total_bytes, downloaded_bytes):
            with progress_lock:
                bar = bars[label]
                if total_bytes:
                    bar.total = (bar.total or 0) + total_bytes
                if downloaded_bytes:
                    bar.update(downloaded_bytes)
                elif total_bytes:
                    bar.refresh()
        return update

    results = [None] * len(urls)
    try:
        with logging_redirect_tqdm(), ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix='mtslinker-download',
        ) as executor:
            futures = {
                executor.submit(
                    download_video_chunk,
                    url,
                    save_directory,
                    progress_callback(label),
                ): (index, label)
                for index, (url, label) in enumerate(zip(urls, labels))
            }
            for future in as_completed(futures):
                index, label = futures[future]
                results[index] = future.result()
                with progress_lock:
                    fragment_done[label] += 1
                    bars[label].set_postfix_str(
                        f'fragments {fragment_done[label]}/{fragment_totals[label]}',
                        refresh=True,
                    )
    finally:
        for bar in bars.values():
            bar.close()
    return results
