import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

import numpy as np

import vanish_cam as app
from naruto_effect import EFFECT_SECONDS, load_log, render_reappearance, substitution_anchor
from test_video_output import arguments


def run_session(mode, gestures, no_mirror=False):
    """Run the camera loop on a fake webcam; `gestures` maps phase -> scripted detections."""
    phases, shown, sent = [], [], []

    class RecordingState(app.AppState):
        def __setattr__(self, key, value):
            if key == "phase" and (not phases or phases[-1] != value):
                phases.append(value)
            super().__setattr__(key, value)

    clock = [0.0]

    def perf_counter():
        clock[0] += 0.02
        return clock[0]

    def seen():
        script = gestures.get(phases[-1], [])
        return script.pop(0) if script else False

    frame = np.zeros((24, 40, 3), np.uint8)
    frame[:, :20] = 200  # Asymmetric, so mirroring is detectable.
    capture = Mock()
    capture.read.side_effect = lambda: (True, frame.copy())
    recognizer = Mock()
    recognizer.recognize_joined_hands.side_effect = lambda *args: seen()
    camera = Mock(width=40, height=24)
    camera.send.side_effect = lambda sent_frame: sent.append(sent_frame.copy())
    sound = Mock()
    mask = np.zeros((24, 40), np.float32)
    mask[4:20, 12:28] = 1.0
    keys = iter(range(3000))

    def wait_key(*_):
        done = next(keys) > 2900 or (phases.count("ready") >= 2 and phases[-1] == "ready")
        return ord("q") if done else 255

    with ExitStack() as mocks:
        mocks.enter_context(patch.object(app, "AppState", RecordingState))
        mocks.enter_context(patch.object(app, "download_models"))
        mocks.enter_context(patch.object(app, "disable_app_nap"))
        mocks.enter_context(patch.object(app, "log"))
        mocks.enter_context(patch.object(app, "open_camera", return_value=capture))
        mocks.enter_context(patch.object(app, "create_tasks", return_value=(Mock(), recognizer)))
        mocks.enter_context(patch.object(app, "open_virtual_camera", return_value=camera))
        mocks.enter_context(patch.object(app, "VanishSound", return_value=sound))
        mocks.enter_context(patch.object(app, "person_mask", return_value=mask))
        mocks.enter_context(patch.object(
            app, "recognize_gesture",
            side_effect=lambda *args: ("Victory", 0.9) if seen() else ("No_Hand", 0.0),
        ))
        mocks.enter_context(patch.object(app.time, "perf_counter", side_effect=perf_counter))
        for name in ("namedWindow", "resizeWindow", "destroyAllWindows"):
            mocks.enter_context(patch.object(app.cv2, name))
        mocks.enter_context(patch.object(app.cv2, "imshow", side_effect=lambda _, f: shown.append(f.copy())))
        mocks.enter_context(patch.object(app.cv2, "waitKey", side_effect=wait_key))
        with ExitStack() as resources:
            app.run_camera(arguments(mode=mode, no_mirror=no_mirror), resources, mode)
    return phases, shown, sent, sound


class GestureToggleTests(unittest.TestCase):
    def test_same_gesture_reappears_only_after_release(self):
        # Still holding the V when vanished must not bring the person back.
        gestures = {"ready": [True], "vanished": [True, True, False, True]}
        phases, *_ = run_session("normal", gestures)
        self.assertEqual(phases, ["calibrating", "ready", "vanishing", "vanished", "reappearing", "ready"])
        self.assertEqual(gestures["vanished"], [])

    def test_naruto_reappearance_plays_sound_again(self):
        phases, _, _, sound = run_session("naruto", {"ready": [True], "vanished": [False, True]})
        self.assertEqual(phases, ["calibrating", "ready", "naruto_vanishing", "vanished", "reappearing", "ready"])
        self.assertEqual(sound.play.call_count, 2)


class MirrorTests(unittest.TestCase):
    def test_meeting_output_is_unmirrored_while_preview_is_mirrored(self):
        _, shown, sent, _ = run_session("normal", {"ready": [True], "vanished": [False, True]})
        self.assertTrue(sent)
        for preview, output in zip(shown, sent):
            np.testing.assert_array_equal(output, preview[:, ::-1])

    def test_no_mirror_sends_the_preview_as_is(self):
        _, shown, sent, _ = run_session("normal", {}, no_mirror=True)
        for preview, output in zip(shown, sent):
            np.testing.assert_array_equal(output, preview)


class ReappearAnimationTests(unittest.TestCase):
    def test_normal_reappearance_builds_up_gradually(self):
        height, width = 180, 320
        person = np.full((height, width, 3), 200, np.uint8)
        background = np.zeros_like(person)
        mask = np.zeros((height, width), np.float32)
        mask[15:, 110:210] = 1.0
        field, sparkle = app.make_dissolve_fields(height, width)
        visible = []
        for t in np.linspace(0, app.REAPPEAR_SECONDS, 8):
            progress = app.REAPPEAR_START_PROGRESS * max(0.0, 1 - t / app.REAPPEAR_SECONDS)
            output = app.apply_dissolve(person, background, mask, progress, field, sparkle)
            visible.append((output[mask > 0.5].mean(axis=1) > 150).mean())
        self.assertLess(visible[1], 0.5)
        self.assertEqual(visible[-1], 1.0)
        self.assertEqual(visible, sorted(visible))

    def test_naruto_reappearance_ends_on_the_live_frame(self):
        frame = np.full((90, 160, 3), 120, np.uint8)
        mask = np.zeros((90, 160), np.float32)
        mask[20:80, 60:100] = 1.0
        anchor = substitution_anchor(mask)
        sprite = load_log()
        for age in np.linspace(0, EFFECT_SECONDS, 12):
            output = render_reappearance(frame, np.zeros_like(frame), mask, sprite, anchor, anchor, age)
            self.assertEqual(output.shape, frame.shape)
        final = render_reappearance(frame, np.zeros_like(frame), mask, sprite, anchor, anchor, EFFECT_SECONDS)
        np.testing.assert_array_equal(final, frame)


if __name__ == "__main__":
    unittest.main()
