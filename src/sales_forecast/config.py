"""Project-level configuration constants."""

DEFAULT_SEASONAL_PERIOD = 52
DEFAULT_TEST_HORIZON = 52

# Segment-wise SARIMAX params from the original notebook.
MODEL_CONFIG = {
    "Consumer": {
        "order": (2, 0, 1),
        "seasonal_order": (1, 1, 0, DEFAULT_SEASONAL_PERIOD),
    },
    "Home Office": {
        "order": (1, 0, 0),
        "seasonal_order": (0, 1, 1, DEFAULT_SEASONAL_PERIOD),
    },
    "Corporate": {
        "order": (1, 0, 1),
        "seasonal_order": (2, 1, 1, DEFAULT_SEASONAL_PERIOD),
    },
}

SEGMENT_ALIASES = {
    "consumer": "Consumer",
    "home office": "Home Office",
    "home_office": "Home Office",
    "corporate": "Corporate",
}
