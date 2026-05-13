"""Find events containing web_results and extract their shape."""

import json
import sys
from pathlib import Path


def parse_event(raw_ev: str):
    et = None
    data_lines: list[str] = []
    for line in raw_ev.split("\n"):
        if line.startswith("event:"):
            et = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())
    try:
        return et, json.loads("\n".join(data_lines)) if data_lines else None
    except Exception:
        return et, None


def main(raw_path: Path) -> None:
    raw = raw_path.read_text().replace("\r\n", "\n")
    events = [e for e in raw.split("\n\n") if e.strip()]

    hits_first_seen = None
    hits_per_event: list[tuple[int, int]] = []
    final_hits: list[dict] = []
    for i, raw_ev in enumerate(events):
        _, data = parse_event(raw_ev)
        if not isinstance(data, dict):
            continue
        for block in data.get("blocks") or []:
            wrb = block.get("web_result_block") if isinstance(block, dict) else None
            if not wrb:
                continue
            results = wrb.get("web_results") or []
            if results:
                if hits_first_seen is None:
                    hits_first_seen = i
                hits_per_event.append((i, len(results)))
                final_hits = results

    print(f"events with web_results: {len(hits_per_event)} / {len(events)}")
    print(f"first event with web_results: index {hits_first_seen}")
    print(f"final web_results count: {len(final_hits)}")
    if hits_per_event:
        print(f"progression: first 5 = {hits_per_event[:5]}")
        print(f"             last 3  = {hits_per_event[-3:]}")

    if final_hits:
        print("\n--- first hit (full shape):")
        print(json.dumps(final_hits[0], indent=2)[:2500])
        print("\n--- summary of all hits:")
        for i, h in enumerate(final_hits[:10]):
            url = h.get("url", "")[:80]
            name = (h.get("name") or "")[:60]
            print(f"  {i + 1:2d}. {name}")
            print(f"      {url}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
