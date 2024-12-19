import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import jsonlines
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


@dataclass
class Config:
    model: str = "gpt-4o-mini"
    api_key: str = os.getenv("OPENAI_API_KEY")
    data_path: Path = Path("../data")
    prompt_path: Path = Path("../prompt")
    data: Path = None
    database: Path = None
    prompt: Path = None

    def __post_init__(self):
        self.data = self.data_path / "test_data_filtered.jsonl"
        self.database = self.data_path / "database.jsonl"
        self.prompt = self.prompt_path / "prompt.toml"


def search(domain: str, engine: list, language: list, pageno: int, searxng_url: str) -> list[str]:
    try:
        response = requests.get(
            searxng_url + "/search",
            params={
                "q": f"{domain}の事業内容",
                "engines": engine,
                "language": language,
                "pageno": pageno,
                "max_ban_time_on_fail": 10,
            },
            timeout=60,
        )
    except requests.exceptions.RequestException as e:
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.find_all("article", class_="result result-default category-general")
    urls = [article.find("a")["href"] for article in articles]
    content = []
    for url in urls:
        try:
            response = requests.get(url, timeout=60)
        except requests.exceptions.RequestException as e:
            continue
        response.encoding = "utf-8"
        if domain in url:
            soup = BeautifulSoup(response.text, "html.parser")
            content.append(soup.get_text().replace("\n", ""))
    return content


def main():
    parser = argparse.ArgumentParser(description="Generate text using OpenAI")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    parser.add_argument("--data_path", type=Path, default=Path("../data"), help="Data path")
    parser.add_argument("--prompt_path", type=Path, default=Path("../prompt"), help="Prompt path")
    args = parser.parse_args()

    # initialize config
    config = Config(model=args.model, api_key=args.api_key, data_path=args.data_path, prompt_path=args.prompt_path)

    with jsonlines.open(config.data) as reader:
        data: list[dict] = [row for row in reader]

    engine = ["google"]
    language = ["ja", "en"]
    pageno = 1
    for row in tqdm(data):
        domain: str = urlparse(row["url"]).netloc
        content = search(domain, engine, language, pageno, os.getenv("SEARXNG_URL"))
        row["content"] = content

    # save jsonl
    with open("tmp.jsonl", mode="w") as writer:
        for row in data:
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
