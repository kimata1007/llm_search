# convert jsonl to csv
import csv

import jsonlines


def jsonl2csv(jsonl_path: str, csv_path: str) -> None:
    with jsonlines.open(jsonl_path) as reader:
        data = [row for row in reader]
    with open(csv_path, mode="w") as writer:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(writer, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


if __name__ == "__main__":
    jsonl2csv(
        "../data/tmp.jsonl",
        "../data/tmp.csv",
    )
