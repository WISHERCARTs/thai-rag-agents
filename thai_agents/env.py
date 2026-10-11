"""Load GEMINI_API_KEY from a .env file before any model client is created."""
from dotenv import find_dotenv, load_dotenv


def load_env() -> None:
    """Read the nearest .env (from the working folder upwards, then from this package upwards).

    Variables that are already set in the environment win over the file.
    """
    path = find_dotenv(usecwd=True) or find_dotenv()
    if path:
        load_dotenv(path, override=False)
