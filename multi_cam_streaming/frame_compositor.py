"""Motion-based frame compositor that selects layouts and assigns cameras to slots."""
import logging
import time
from datetime import datetime

import cv2
import numpy as np

log = logging.getLogger(__name__)


_MOTION_THRESHOLD = 15       # pixel diff below this is treated as sensor noise
_DEBUG_BAR_WIDTH = 10        # character width of the score bar in the debug window
_CHANGE_THRESHOLD = 0.05     # min absolute score delta required to accept a layout/assignment change
_MIN_SWITCH_INTERVAL = 5.0   # minimum seconds between accepted layout/assignment changes
_TRANSITION_DURATION = 0.5   # seconds to animate slot geometry and alpha-blend on layout change


def _lerp(a, b, t):
    return a + (b - a) * t


class FrameCompositor:
    """Compose a multi-camera output frame from raw captured frames.

    Responsibilities:
    - Score each camera slot by motion (thresholded grayscale diff)
    - Select the layout whose emphasis best matches the current motion distribution
    - Assign cameras to slots by descending motion score
    - Suppress layout/assignment changes unless scores have changed substantially
      (hysteresis via motion_change_threshold)
    - Animate slot geometry (pos/size) from old to new positions during transitions
    - Alpha-blend the animated composite with the pre-transition snapshot for smoothness
    - Enforce a minimum time between accepted layout/assignment changes
    - Respect per-camera slot bounds and activity correction
    - Optionally display a debug window showing the reduced grayscale diff images

    State (prev_grays, motion_scores, timing, active layout, transition) is managed
    internally so the caller only needs to call process() each loop iteration.
    """

    def __init__(self, layouts, output_dims,
                 cam_attrs=None,
                 motion_log_interval=1.0, motion_threshold=_MOTION_THRESHOLD,
                 motion_change_threshold=_CHANGE_THRESHOLD,
                 min_switch_interval=_MIN_SWITCH_INTERVAL,
                 transition_duration=_TRANSITION_DURATION, show_motion_debug=False):
        """
        Args:
            layouts: List of layout dicts, each with 'name' and 'frames' (list of slot dicts).
                     Slot dicts have 'pos' [x,y] and 'size' (scalar, same fraction for w and h).
            output_dims: (width, height) of the output canvas in pixels.
            cam_attrs: Optional list of per-camera attribute dicts, one per camera in the same
                       order as frames passed to process(). Each dict may contain:
                         min_slot (int, default 0): camera is never placed in a slot with a
                             lower index (slot 0 = largest). Ignored if the constraint cannot
                             be satisfied (slot stays black).
                         max_slot (int|inf, default inf): camera is never placed in a slot
                             with a higher index.
                         activity_multiplier (float, default 1.0): multiplied with the raw
                             motion score before ranking. Values >1 increase perceived activity,
                             <1 reduce it. 1.0 = no change.
            motion_log_interval: Seconds between motion score computations.
            motion_threshold: Pixel diff value below which changes are ignored (noise gate).
            motion_change_threshold: Minimum absolute score change (0.0–1.0) required on at
                                     least one camera before a layout or assignment switch is
                                     accepted. Prevents flickering from minor score fluctuations.
            min_switch_interval: Minimum seconds that must elapse between two accepted
                                 layout or assignment changes. Score-based hysteresis may
                                 qualify a change, but it is suppressed until this interval
                                 has passed since the last accepted switch.
            transition_duration: Seconds to animate slot geometry and alpha-blend when
                                 layout or camera assignment changes.
            show_motion_debug: If True, display a debug window with grayscale diff panels.
        """
        if not layouts:
            raise ValueError("layouts must contain at least one layout")

        self.output_dims = output_dims
        self._cam_attrs = cam_attrs or []
        self.motion_log_interval = motion_log_interval
        self.motion_threshold = motion_threshold
        self.motion_change_threshold = motion_change_threshold
        self.min_switch_interval = min_switch_interval
        self.transition_duration = transition_duration
        self.show_motion_debug = show_motion_debug

        self._layouts = layouts
        self._layout_emphasis = [self._compute_emphasis(l['frames']) for l in layouts]

        self._active_layout_idx = 0
        self._slot_assignment = []   # ordered list of cam_idxs (int), one per slot

        # Scores at the time the last layout/assignment change was accepted.
        # Keyed by cam_idx; used to measure how much scores have moved since then.
        self._accepted_scores = {}

        # Transition state
        self._transition_start = float('-inf')
        self._last_switch_time = float('-inf')
        self._frame_old = None      # alpha-blend source: snapshot at transition start
        self._old_geom = {}         # cam_idx → {'pos': [x,y], 'size': s} at transition start

        self._prev_grays = {}       # cam_idx → grayscale array
        self._last_score_time = 0.0
        self._motion_scores = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frames):
        """Score, select layout, assign cameras, and composite into the output canvas.

        Args:
            frames: List of BGR numpy arrays from read_frames(), one per camera.

        Returns:
            BGR numpy array of shape (height, width, 3).
        """
        now = time.time()

        if now - self._last_score_time >= self.motion_log_interval:
            self._motion_scores, diff_images = self._compute_motion_scores(frames)
            if self.show_motion_debug:
                self._show_motion_debug(diff_images, self._motion_scores)
            self._last_score_time = now

            # Only consider a layout/assignment change if scores moved substantially
            score_by_idx = {cam_idx: score
                            for cam_idx, score in enumerate(self._motion_scores)}
            if self._scores_changed_substantially(score_by_idx):
                new_idx = self._select_layout(self._motion_scores)
                ranked = self._rank_cameras(frames)
                new_assignment = self._build_assignment(ranked, self._layouts[new_idx]['frames'])

                if (new_idx != self._active_layout_idx or new_assignment != self._slot_assignment) \
                        and now - self._last_switch_time >= self.min_switch_interval:
                    old_layout = self._layouts[self._active_layout_idx]['frames']
                    self._frame_old = self._composite(
                        self._rank_cameras(frames), old_layout,
                        self._slot_assignment, t=1.0)
                    self._old_geom = self._geometry_by_camera(old_layout, self._slot_assignment)

                    if log.isEnabledFor(logging.DEBUG):
                        self._log_transition(
                            old_layout, self._slot_assignment, self._accepted_scores or {},
                            new_idx, new_assignment, score_by_idx,
                        )

                    self._active_layout_idx = new_idx
                    self._slot_assignment = new_assignment
                    self._transition_start = now
                    self._last_switch_time = now

                # Update accepted scores regardless of whether the state changed, so that
                # once a change is accepted the bar resets from the new baseline.
                self._accepted_scores = score_by_idx

        elapsed = now - self._transition_start
        t = min(1.0, elapsed / max(self.transition_duration, 1e-6))

        ranked = self._rank_cameras(frames)
        layout_frames = self._layouts[self._active_layout_idx]['frames']
        frame_new = self._composite(ranked, layout_frames, self._slot_assignment,
                                    t=t, old_geom=self._old_geom)

        if t < 1.0 and self._frame_old is not None:
            return cv2.addWeighted(self._frame_old, 1.0 - t, frame_new, t, 0)
        return frame_new

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log_transition(self, old_layout, old_assignment, old_scores,
                        new_idx, new_assignment, new_scores):
        def _fmt(scores, assignment, layout_frames):
            lines = []
            for slot_idx, cam_idx in enumerate(assignment):
                score = scores.get(cam_idx, 0.0)
                slot = layout_frames[slot_idx] if slot_idx < len(layout_frames) else {}
                size = slot.get('size', 0)
                lines.append(
                    f"  slot {slot_idx}: cam {cam_idx}"
                    f"  score={score:.4f}"
                    f"  size={size:.2f}"
                )
            return '\n'.join(lines) or '  (none)'

        old_name = self._layouts[self._active_layout_idx]['name']
        new_name = self._layouts[new_idx]['name']
        log.debug(
            "Layout/assignment change triggered\n"
            "  OLD layout: %s\n%s\n"
            "  NEW layout: %s\n%s",
            old_name, _fmt(old_scores, old_assignment, old_layout),
            new_name, _fmt(new_scores, new_assignment, self._layouts[new_idx]['frames']),
        )

    def _scores_changed_substantially(self, score_by_idx):
        """Return True if any camera's score has changed by more than motion_change_threshold
        since the last accepted state, or if no baseline exists yet.
        """
        if not self._accepted_scores:
            return True
        for cam_idx, score in score_by_idx.items():
            prev = self._accepted_scores.get(cam_idx, 0.0)
            if abs(score - prev) > self.motion_change_threshold:
                return True
        return False

    @staticmethod
    def _compute_emphasis(frames_list):
        """Return emphasis = area(largest slot) / mean(all slot areas).

        A value near 1.0 means all slots are equal size (e.g. quad).
        Higher values indicate one dominant slot (e.g. main-right-strip ≈ 4.0).
        """
        areas = [s['size'] ** 2 for s in frames_list]
        if not areas:
            return 1.0
        mean_area = sum(areas) / len(areas)
        if mean_area == 0:
            return 1.0
        return max(areas) / mean_area

    def _select_layout(self, motion_scores):
        """Return the index of the layout whose emphasis best matches current motion."""
        sorted_scores = sorted(motion_scores, reverse=True)
        if len(sorted_scores) < 2 or sum(sorted_scores) == 0:
            motion_emphasis = 1.0
        else:
            rest_mean = sum(sorted_scores[1:]) / len(sorted_scores[1:])
            if rest_mean == 0:
                motion_emphasis = self._layout_emphasis[-1] if self._layout_emphasis else 1.0
            else:
                motion_emphasis = sorted_scores[0] / rest_mean

        return min(
            range(len(self._layout_emphasis)),
            key=lambda i: abs(self._layout_emphasis[i] - motion_emphasis)
        )

    def _build_assignment(self, ranked_frames, layout_frames):
        """Return a list of cam_idxs ordered by slot index, with largest slot getting top scorer.

        Cameras are assigned in order of how many slots they are eligible for (most constrained
        first). Within the same constraint tightness, motion rank is preserved. Each camera is
        placed in its highest-ranked eligible slot that has not yet been filled. This ensures
        constrained cameras are never crowded out by unconstrained ones.
        """
        cameras = [(cam_idx, frame) for _, (cam_idx, frame) in ranked_frames]

        def _allowed(cam_idx, slot_idx):
            if cam_idx >= len(self._cam_attrs):
                return True
            a = self._cam_attrs[cam_idx]
            return a['min_slot'] <= slot_idx <= a['max_slot']

        slot_rank_order = sorted(
            range(len(layout_frames)),
            key=lambda i: layout_frames[i]['size'] ** 2,
            reverse=True
        )

        # Sort cameras: most constrained (fewest eligible slots) first;
        # break ties by motion rank (already the order in `cameras`).
        n_slots = len(layout_frames)
        cameras_by_constraint = sorted(
            range(len(cameras)),
            key=lambda r: sum(_allowed(cameras[r][0], s) for s in range(n_slots))
        )

        assignment = [None] * n_slots
        used = set()

        for rank in cameras_by_constraint:
            cam_idx, _ = cameras[rank]
            for slot_idx in slot_rank_order:
                if assignment[slot_idx] is not None:
                    continue
                if _allowed(cam_idx, slot_idx):
                    assignment[slot_idx] = cam_idx
                    used.add(cam_idx)
                    break

        return assignment

    @staticmethod
    def _geometry_by_camera(layout_frames, slot_assignment):
        """Return dict mapping cam_idx → {'pos': [x,y], 'size': s} for each assigned slot."""
        geom = {}
        for slot_idx, slot in enumerate(layout_frames):
            if slot_idx >= len(slot_assignment):
                break
            cam_idx = slot_assignment[slot_idx]
            if cam_idx is not None:
                geom[cam_idx] = {'pos': list(slot['pos']), 'size': slot['size']}
        return geom

    def _compute_motion_scores(self, frames):
        """Return (scores, diff_images) — one score and one diff image per camera.

        Pipeline: resize to 1/4 → grayscale → thresholded absdiff vs previous gray.
        Score = fraction of pixels with diff > motion_threshold.
        """
        scores = []
        diff_images = []
        updated_grays = {}

        for cam_idx, frame in enumerate(frames):
            small = cv2.resize(frame, (frame.shape[1] // 4, frame.shape[0] // 4))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if cam_idx in self._prev_grays:
                diff = cv2.absdiff(gray, self._prev_grays[cam_idx])
                raw = float(np.count_nonzero(diff > self.motion_threshold)) / diff.size
            else:
                diff = np.zeros_like(gray)
                raw = 0.0
            multiplier = self._cam_attrs[cam_idx]['activity_multiplier'] \
                if cam_idx < len(self._cam_attrs) else 1.0
            scores.append(raw * multiplier)
            diff_images.append(diff)
            updated_grays[cam_idx] = gray

        self._prev_grays.update(updated_grays)
        return scores, diff_images

    def _show_motion_debug(self, diff_images, scores):
        """Display a debug window with diff panels and score bars side by side."""
        max_score = max(scores, default=1.0) or 1.0
        panels = []
        for diff_img, score in zip(diff_images, scores):
            panel = cv2.cvtColor(diff_img, cv2.COLOR_GRAY2BGR)
            bar = f"{score:.4f} {'*' * int(score / max_score * _DEBUG_BAR_WIDTH):<{_DEBUG_BAR_WIDTH}}"
            h = panel.shape[0]
            cv2.rectangle(panel, (0, h - 14), (panel.shape[1], h), (40, 40, 40), -1)
            cv2.putText(panel, bar, (2, h - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            panels.append(panel)
        if panels:
            cv2.imshow("motion debug", np.hstack(panels))

    def _rank_cameras(self, frames):
        """Return (cam_idx, frame) pairs sorted by descending motion score.

        Guards against length mismatch: falls back to original order when counts differ.
        """
        indexed = list(enumerate(frames))
        scores = self._motion_scores
        if len(scores) != len(indexed):
            return [(0.0, item) for item in indexed]
        return sorted(zip(scores, indexed), key=lambda x: x[0], reverse=True)

    def _composite(self, ranked_frames, layout_frames, slot_assignment, t=1.0, old_geom=None):
        """Blit frames onto a black canvas, animating slot geometry during transitions.

        Args:
            ranked_frames: List of (score, (cam_idx, frame)) sorted by descending score.
            layout_frames: List of target slot dicts with 'pos' and 'size' (scalar).
            slot_assignment: List of cam_idxs, one per slot. None entries leave the slot black.
            t: Transition progress 0.0→1.0. At 1.0 (steady state) slots are at target geometry.
            old_geom: Dict cam_idx → {'pos', 'size'} from before the transition. When provided
                      and t < 1, each camera's geometry is interpolated from old to new.
                      Cameras absent from old_geom start directly at their new geometry.
        """
        W, H = self.output_dims
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        cam_frame = {cam_idx: frame for _, (cam_idx, frame) in ranked_frames}

        for slot_idx, slot in enumerate(layout_frames):
            cam_idx = slot_assignment[slot_idx] if slot_idx < len(slot_assignment) else None
            if cam_idx is None or cam_idx not in cam_frame:
                continue

            target_pos = slot['pos']
            target_size = slot['size']
            if t < 1.0 and old_geom and cam_idx in old_geom:
                src = old_geom[cam_idx]
                pos_x = _lerp(src['pos'][0], target_pos[0], t)
                pos_y = _lerp(src['pos'][1], target_pos[1], t)
                sz    = _lerp(src['size'],   target_size,   t)
            else:
                pos_x, pos_y = target_pos[0], target_pos[1]
                sz = target_size

            px = int(pos_x * W)
            py = int(pos_y * H)
            pw = int(sz * W)
            ph = int(sz * H)

            if pw <= 0 or ph <= 0:
                continue

            frame = cam_frame[cam_idx]
            resized = cv2.resize(frame, (pw, ph))

            x1, y1 = max(px, 0), max(py, 0)
            x2, y2 = min(px + pw, W), min(py + ph, H)
            rx1 = x1 - px
            ry1 = y1 - py
            canvas[y1:y2, x1:x2] = resized[ry1:ry1 + (y2 - y1), rx1:rx1 + (x2 - x1)]

            cv2.putText(canvas, str(cam_idx), (max(px + 5, 5), max(py + 20, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.rectangle(canvas, (0, H - 15), (120, H), (80, 80, 80), -1)
        cv2.putText(canvas, timestamp, (3, H - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))

        return canvas
