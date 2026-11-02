"""Build tiny PDFs that pypdf can read back in tests."""


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(page_texts: list[str]) -> bytes:
    """Return a one-or-more-page PDF whose extractable text is `page_texts`."""
    if not page_texts:
        raise ValueError("page_texts must not be empty.")

    objects: dict[int, bytes] = {3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"}
    next_id = 4
    page_ids: list[int] = []
    for text in page_texts:
        content_id = next_id
        page_id = next_id + 1
        next_id += 2
        stream = f"BT /F1 12 Tf 72 720 Td ({_escape(text)}) Tj ET".encode()
        objects[content_id] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
        )
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R /Resources << /Font << /F1 3 0 R >> >> >>"
        ).encode()
        page_ids.append(page_id)

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = f"<< /Type /Pages /Count {len(page_ids)} /Kids [{kids}] >>".encode()

    header = b"%PDF-1.4\n"
    blob = bytearray(header)
    max_id = max(objects)
    offsets = [0] * (max_id + 1)
    for obj_id in range(1, max_id + 1):
        offsets[obj_id] = len(blob)
        blob.extend(f"{obj_id} 0 obj\n".encode())
        blob.extend(objects[obj_id])
        blob.extend(b"\nendobj\n")
    xref_pos = len(blob)
    blob.extend(f"xref\n0 {max_id + 1}\n".encode())
    blob.extend(b"0000000000 65535 f \n")
    for obj_id in range(1, max_id + 1):
        blob.extend(f"{offsets[obj_id]:010d} 00000 n \n".encode())
    blob.extend(
        f"trailer\n<< /Size {max_id + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode()
    )
    return bytes(blob)
