import unittest

import numpy as np
from moviepy.video.VideoClip import ColorClip

from mtslinker.processor import (
    _ordered_video_clips,
    _video_output_size,
    _video_role,
    create_video_with_gaps,
)


def _clip(size, color, role, start=0, duration=1):
    clip = ColorClip(size=size, color=color, duration=duration).with_start(start)
    clip._mts_video_role = role
    return clip


class VideoCompositionTests(unittest.TestCase):
    def test_recognizes_mts_video_roles(self):
        self.assertEqual(_video_role({'screensharing': {'id': 1}}), 'screensharing')
        self.assertEqual(_video_role({'conference': {'id': 2}}), 'conference')
        self.assertEqual(_video_role(None), 'unknown')

    def test_screen_share_is_last_and_defines_canvas(self):
        screen = _clip((1626, 996), (0, 255, 0), 'screensharing')
        camera = _clip((640, 360), (255, 0, 0), 'conference')
        ordered = _ordered_video_clips([screen, camera])

        self.assertEqual(ordered, [camera, screen])
        self.assertEqual(_video_output_size(ordered), (1626, 996))

    def test_largest_video_wins_when_stream_metadata_is_missing(self):
        small = _clip((640, 360), (255, 0, 0), 'unknown', duration=100)
        large = _clip((1626, 996), (0, 255, 0), 'unknown', duration=90)

        self.assertEqual(_ordered_video_clips([large, small]), [small, large])
        self.assertEqual(_video_output_size([small, large]), (1626, 996))

    def test_screen_share_covers_conference_only_while_active(self):
        camera = _clip((2, 2), (255, 0, 0), 'conference', duration=2)
        screen = _clip((4, 4), (0, 255, 0), 'screensharing', start=0.5, duration=1)
        result = create_video_with_gaps(2, [screen, camera])
        try:
            np.testing.assert_array_equal(result.get_frame(0.25)[0, 0], [255, 0, 0])
            np.testing.assert_array_equal(result.get_frame(0.75)[0, 0], [0, 255, 0])
            np.testing.assert_array_equal(result.get_frame(1.75)[0, 0], [255, 0, 0])
            self.assertEqual(result.duration, 2)
        finally:
            result.close()


if __name__ == '__main__':
    unittest.main()
