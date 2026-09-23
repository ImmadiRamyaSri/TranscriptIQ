import re

TIMESTAMP = re.compile(r"^\d{1,2}:\d{2}$")
SPEAKER   = re.compile(r"^([A-Za-z][A-Za-z.'\- ]*?):\s*(.*)$")
EXPERT    = re.compile(r"^Expert\s*\d*\s*[–—-]\s*(.+)$")
FIELD     = re.compile(r"^(Role|Market)\s*:\s*(.+)$", re.I)
QUESTION  = re.compile(r"^(\d+)\.\s*(.+)$")

def parse_transcript(path):
    """
    Parses the transcripts and extracts the header and list of segments
    """
    header = {"name": None, "role": None,"market" :None}
    segments = []
    current = None
    in_header = True

    with open(path, encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if in_header:
                if TIMESTAMP.match(line):
                    in_header = False
                else:
                    m = EXPERT.match(line)
                    if m:
                        header["name"] = m.group(1).strip()
                    m=FIELD.match(line)
                    if m:
                        header[m.group(1).lower()] = m.group(2).strip()
                    continue

            if TIMESTAMP.match(line):
                if current:
                    segments.append(current)
                current = {"timestamp": line, "speaker": None, "text": []}
                continue

            if current is None:
                 continue

            if current["speaker"] is None:
                m = SPEAKER.match(line)
                if m:
                    current["speaker"] = m.group(1).strip()
                    if m.group(2):
                        current["text"].append(m.group(2).strip())
                        continue

            current["text"].append(line)

    if current:
        segments.append(current)

    for s in segments:
        s["text"] = " ".join(s["text"])

    return header,segments

def parse_questions(path):
    """
    Parses the interview guide and extracts the list of standardized questions.
    """
    questions = []

    with open(path, encoding='utf-8') as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue

                m = QUESTION.match(line)
                if m:
                    questions.append(m.group(2).strip())

    return questions
