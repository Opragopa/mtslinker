import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr

from mtslinker.downloader import download_video_chunks


class DownloadProgressTests(unittest.TestCase):
    def test_cached_fragments_keep_input_order_with_grouped_progress(self):
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as directory:
            names = ['first.mp4', 'second.mp4', 'third.mp4']
            for index, name in enumerate(names, start=1):
                with open(os.path.join(directory, name), 'wb') as file:
                    file.write(b'x' * index)

            urls = [f'https://example.invalid/{name}' for name in names]
            with redirect_stderr(io.StringIO()):
                result = download_video_chunks(
                    urls,
                    directory,
                    max_workers=3,
                    progress_labels=['Conference', 'Screen sharing', 'Conference'],
                )

            self.assertEqual([os.path.basename(path) for path in result], names)

    def test_requires_one_progress_label_per_url(self):
        with self.assertRaisesRegex(ValueError, 'one label per video URL'):
            download_video_chunks(
                ['https://example.invalid/one.mp4'],
                '.',
                progress_labels=[],
            )


if __name__ == '__main__':
    unittest.main()
