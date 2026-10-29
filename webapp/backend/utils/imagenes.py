"""Helpers de imagen del backend."""
import base64
import io


def a_base64_png(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
