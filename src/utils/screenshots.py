import os
import re
from datetime import datetime

def sanitize_filename(text: str) -> str:
    """Sanitize text to make it safe for filenames."""
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "_", text)

def take_screenshot(page, page_title: str, screenshots_dir: str, label: str = "screenshot") -> str:
    """Take a screenshot with a label and timestamp."""
    safe_title = sanitize_filename(page_title or "page")
    safe_label = sanitize_filename(label)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{safe_title}_{safe_label}_{timestamp}.png"
    screenshot_path = os.path.join(screenshots_dir, filename)
    page.screenshot(path=screenshot_path, full_page=True)
    return screenshot_path
