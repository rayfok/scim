import json
import os
import sys


def main(start_id, end_id):
    HIGHLIGHTS_DIR = "output/highlights"
    SECTIONS_DIR = "output/sections"
    papers = {}
    for i in range(start_id, end_id + 1):
        paper_id = f"2022.naacl-main.{i}"
        highlights_file = os.path.join(HIGHLIGHTS_DIR, f"{paper_id}.json")
        sections_file = os.path.join(SECTIONS_DIR, f"2022.naacl-main.{i}.json")
        if not os.path.exists(highlights_file) or not os.path.exists(sections_file):
            print(f"=> 2022.naacl-main.{i}.json missing")
            continue
        with open(highlights_file, "r") as f:
            highlights = json.load(f)
        with open(sections_file, "r") as f:
            sections = json.load(f)

        papers[paper_id] = {"highlights": highlights, "sections": sections}
    with open("output/facets.json", "w") as out:
        json.dump(papers, out)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            f"Error: Invalid args.\nRun with python {sys.argv[0]} [START_ID] [END_ID]"
        )
        sys.exit(1)
    start_id, end_id = int(sys.argv[1]), int(sys.argv[2])
    main(start_id, end_id)
