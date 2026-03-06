# src/ui/assets.py
import os
import pygame

RANKS = "23456789TJQKA"
SUITS = "shdc"

def load_card_images(cards_dir: str, card_size: tuple[int, int]) -> tuple[dict[str, pygame.Surface], pygame.Surface]:
    """
    Returns:
      fronts: dict like {"As": Surface, "Td": Surface, ...}
      back: Surface
    """
    fronts: dict[str, pygame.Surface] = {}

    def load_png(path: str) -> pygame.Surface:
        img = pygame.image.load(path).convert_alpha()
        return pygame.transform.smoothscale(img, card_size)

    # back
    back_path = os.path.join(cards_dir, "back.png")
    if not os.path.exists(back_path):
        raise FileNotFoundError(f"Missing card back image: {back_path}")
    back = load_png(back_path)

    # fronts
    missing = []
    for r in RANKS:
        for s in SUITS:
            key = f"{r}{s}"
            path = os.path.join(cards_dir, f"{key}.png")
            if not os.path.exists(path):
                missing.append(path)
                continue
            fronts[key] = load_png(path)

    if missing:
        # You can make this a warning instead, but failing fast is better while wiring up assets
        raise FileNotFoundError(
            "Missing some card images. Example missing:\n"
            + "\n".join(missing[:10])
            + ("\n..." if len(missing) > 10 else "")
        )

    return fronts, back