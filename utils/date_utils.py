from datetime import datetime


def ensure_datetime(value):
    if isinstance(value, datetime):
        return value
    elif isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return datetime.now()  # fallback if string is invalid
    else:
        return datetime.now()
