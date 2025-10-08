from fish_benchmark.typing.types import SlidingStyle

TEST_FRAMES = SlidingStyle(
    name = "test_frames",
    window_size = 1,
    tolerance_region = 0,
    samples_per_window = 1,
    step_size = 1,
    data_ndim = 3,
    patch_type = "relative",
    patch_h = 1,
    patch_w = 1,
)

TEST_SLIDING_WINDOW = SlidingStyle(
    name = "test_sliding_window",
    window_size = 16,
    tolerance_region = 0,
    samples_per_window = 16,
    step_size = 1,
    data_ndim = 4,
    patch_type = "relative",
    patch_h = 1,
    patch_w = 1,
)

FRAMES = SlidingStyle(
    name = "frames",
    window_size = 1,
    tolerance_region = 0,
    samples_per_window = 1,
    step_size = 1,
    data_ndim = 3,
    patch_type = "relative",
    patch_h = 1,
    patch_w = 1,
)

SLIDING_WINDOW_W_TEMP = SlidingStyle(
    name = "sliding_window_w_temp",
    window_size = 16,
    tolerance_region = 7,
    samples_per_window = 16,
    step_size = 8,
    data_ndim = 4,
    patch_type = "relative",
    patch_h = 1,
    patch_w = 1,
)

SLIDING_STYLES = {
    "test_frames": TEST_FRAMES,
    "test_sliding_window": TEST_SLIDING_WINDOW,
    "frames": FRAMES,
    "sliding_window_w_temp": SLIDING_WINDOW_W_TEMP,
}

