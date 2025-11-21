from vision_bench.typing.types import  LocalDataset, Split
from .sliding_styles import *

FISHFOLLOW = LocalDataset(
    name="fishfollow",
    path='/share/j_sun/jth264/mikev3',
    precomputed_path='/share/j_sun/jth264/precomputed/fishfollow',
    categories=[
        "Other behavior",
        "Medium bites",
        "High bites",
        "Traversing/swimming",
        "Departure",
        "Cleaner wrasse",
        "Low bites",
        "Change in Focal Fish",
        "Solo foraging/swimming",
        "Seafloor bites",
        "Aggressive on focal",
        "Social foraging/swimming",
        "Sand rubbing",
        "Not visible",
        "Foraging",
        "Idle",
        "Coral habitat",
        "Rubble habitat",
        "Aggressive by focal",
        "Sand habitat"
    ],
    label_type="onehot",
    splits=[
        Split(
            name="train",
            sliding_styles=[
                FRAMES_W_TEMP,
                SLIDING_WINDOW_W_TEMP,
            ]
        ), Split(
            name="val",
            sliding_styles=[
                FRAMES_W_TEMP,
                SLIDING_WINDOW_W_TEMP,
            ]
        ), Split(
            name="test",
            sliding_styles=[
                TEST_FRAMES,
                TEST_SLIDING_WINDOW,
            ]
        )
    ]
)

CORALCAM = LocalDataset(
    name="coralcam",
    path='/share/j_sun/jth264/refactor/coralcam',
    precomputed_path='/share/j_sun/jth264/refactor/coralcam/precomputed',
    label_type="onehot",
    categories=[
        "mouth not visible",
        "feeding",
        "charging",
        "being charged",
        "cturn"
    ],
    splits=[
        Split(
            name="train",
            sliding_styles=[
                FRAMES_W_TEMP,
                SLIDING_WINDOW_W_TEMP,
            ]
        ), Split(
            name="val",
            sliding_styles=[
                FRAMES_W_TEMP,
                SLIDING_WINDOW_W_TEMP,
            ]
        ), Split(
            name="test",
            sliding_styles=[
                TEST_FRAMES,
                TEST_SLIDING_WINDOW,
            ]
        )
    ]
)

DATASETS = {
    "fishfollow": FISHFOLLOW,
    "coralcam": CORALCAM,
}
