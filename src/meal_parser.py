"""
Optional: turn a plain-English meal description into carbs / fibre / protein / fat.

    "2 rotis, dal and a bowl of curd"  ->  {carbs_g: 62, fiber_g: 9, protein_g: 20, fat_g: 11, ...}

Why this exists: nobody knows the gram values of Indian food. Typing "2 rotis and dal" is something a
patient or a doctor can actually do; typing "57 g net carbs" is not. It replaces the ten-row
approximate menu in data/indian_meals.csv with anything the user can describe.

What it is NOT: it does not touch the digital twin, the LightGBM model, or any reported number. It
only produces the four nutrient values that the user would otherwise type by hand into "Build your
own". The twin does the physiology; this is a text-to-numbers convenience.

It is entirely OPTIONAL. With no API key configured, `available()` returns False and the dashboard
hides the feature - everything else works exactly as before. The key is read from Streamlit secrets
or the environment and is never written to disk by this module.

Estimates are approximate, like the rest of the menu, and the UI says so.
"""
from __future__ import annotations

import json
import os

import config

MODEL = "claude-opus-5-5"
MAX_TOKENS = 1024

SYSTEM = """You estimate the nutrient content of a described meal.

Return ONLY the nutrients for the WHOLE meal as described, summing every item and respecting the
quantities given ("2 rotis" is two, "a small bowl" is smaller than "a bowl").

Use typical Indian household serving sizes when the food is Indian and no size is given. If the
description is not food at all, or is far too vague to estimate (for example "lunch"), set
"understood" to false and explain briefly in "note" what you would need to know.

Be honest about uncertainty in "note" - one short sentence, for example which assumption you made
about portion size. Do not claim precision you do not have.

has_veg_protein is true when the meal contains a distinct vegetable, dal, curd, paneer, egg, fish or
meat portion that a person could physically eat BEFORE the carbohydrate part of the meal."""

SCHEMA = {
    "type": "object",
    "properties": {
        "understood": {"type": "boolean", "description": "false if this is not estimable food"},
        "name": {"type": "string", "description": "short name for the meal, e.g. '2 roti + dal'"},
        "items": {"type": "array", "items": {"type": "string"},
                  "description": "each food item with the quantity you assumed"},
        "carbs_g": {"type": "number", "description": "total carbohydrate in grams, including fibre"},
        "fiber_g": {"type": "number", "description": "total dietary fibre in grams"},
        "protein_g": {"type": "number"},
        "fat_g": {"type": "number"},
        "calories_kcal": {"type": "number"},
        "has_veg_protein": {"type": "boolean"},
        "note": {"type": "string", "description": "one short sentence on the main assumption"},
    },
    "required": ["understood", "name", "items", "carbs_g", "fiber_g", "protein_g", "fat_g",
                 "calories_kcal", "has_veg_protein", "note"],
    "additionalProperties": False,
}


def get_key() -> str | None:
    """Streamlit secrets first, then the environment. Returns None when nothing is configured."""
    try:
        import streamlit as st
        key = st.secrets.get("ANTHROPIC_API_KEY")
        if key:
            return str(key)
    except Exception:
        pass  # no streamlit context, no secrets file, or the key is absent
    return os.environ.get("ANTHROPIC_API_KEY") or None


def available() -> bool:
    """True only if the SDK is installed AND a key is configured. The dashboard gates on this."""
    if not get_key():
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def parse_meal(description: str) -> dict:
    """Estimate one meal's nutrients. Returns {"ok": bool, ...} and never raises.

    On success the dict carries the schema fields above plus net_carbs, computed with the same
    rule the pipeline uses for real meals (carbs - fibre, floored at zero).
    """
    description = (description or "").strip()
    if not description:
        return {"ok": False, "error": "Type what you ate first."}
    key = get_key()
    if not key:
        return {"ok": False, "error": "No API key configured."}

    try:
        import anthropic
    except ImportError:
        return {"ok": False, "error": "The 'anthropic' package is not installed."}

    try:
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": description}],
        )
    except anthropic.AuthenticationError:
        return {"ok": False, "error": "The API key was rejected."}
    except anthropic.RateLimitError:
        return {"ok": False, "error": "Rate limited - wait a moment and try again."}
    except anthropic.APIConnectionError:
        return {"ok": False, "error": "Could not reach the API. Check the network."}
    except Exception as e:  # never let this take the dashboard down
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:150]}"}

    if resp.stop_reason == "refusal":
        return {"ok": False, "error": "The request was declined. Try describing the food plainly."}

    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"ok": False, "error": "Could not read the estimate. Try rephrasing."}

    if not data.get("understood", False):
        return {"ok": False, "error": data.get("note") or "That could not be read as a meal."}

    out = {"ok": True}
    for k in ("name", "items", "note", "has_veg_protein"):
        out[k] = data.get(k)
    for k in ("carbs_g", "fiber_g", "protein_g", "fat_g", "calories_kcal"):
        try:
            out[k] = max(float(data.get(k, 0.0)), 0.0)
        except (TypeError, ValueError):
            out[k] = 0.0
    out["fiber_g"] = min(out["fiber_g"], out["carbs_g"])      # fibre is part of total carbohydrate
    out["net_carbs"] = max(out["carbs_g"] - out["fiber_g"], 0.0)
    out["has_veg_protein"] = bool(out.get("has_veg_protein"))
    out["usage"] = {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}
    return out
