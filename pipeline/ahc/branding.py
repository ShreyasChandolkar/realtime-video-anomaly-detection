"""The name shown on the public leaderboard.

Held fixed on purpose. `model_name` is displayed publicly beside the score, so
changing it announces every architectural move we make to everyone reading the
board. It stays as the string it has always been regardless of what is actually
running underneath.
"""

PUBLIC_MODEL_NAME = "siglip2-onset-cascade"
