"""features: feature definitions built on bankpulse_core (one module per feature group).

Importing this package imports every feature module, which registers all features, so
callers only need `import features` before resolving the config's feature list.
"""

from features import campaign, client_profile, contact_history, economic  # noqa: F401  (registers features)
