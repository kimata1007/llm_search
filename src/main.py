import argparse
import json
import os
from pathlib import Path

import jsonlines
import openai
import requests
import toml
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


# set args
def set_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    args = parser.parse_args(args=[])
    data_path = Path("../data")
    result_path = Path("../results")
    prompt_path = Path("../prompt/")
    args.api_key = os.getenv("OPENAI_API_KEY")
    args.model = "gpt-4o-mini"
    args.data_path = data_path / "tmp.jsonl"
    args.save_path_jsonl = result_path / "results.jsonl"  # TODO: change
    args.prompt_path = prompt_path / "prompt.toml"
    args.sample_size = 1
    return args


class ChatGPT:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

        self.client = openai.OpenAI(api_key=args.api_key)
        
        # load data
        if args.data_path.suffix == ".jsonl":
            with jsonlines.open(args.data_path) as reader:
                self.data: list[dict] = [row for row in reader]
        else:
            raise ValueError("Invalid data file format")

        if args.prompt_path.suffix == ".toml":
            with open(args.prompt_path, mode="r") as reader:
                self.prompt: dict = toml.load(reader)
        else:
            raise ValueError("Invalid prompt file format")

        # load saved list
        self.save_list: list = []
        self.save_input: list = []

        self.prompt_tokens: int = 0
        self.output_tokens: int = 0
        self.total_tokens: int = 0

    def get_chatgpt_response(self, data: str, **kargs: dict) -> str:
        messages: list[dict] = [
            {"role": self.prompt["messages"][0]["role"], "content": self.prompt["messages"][0]["content"]},
            {
                "role": self.prompt["messages"][1]["role"],
                "content": self.prompt["messages"][1]["content"]
                .replace("<CONTENT>", data["companies"])
                .replace("<COMPANY INFORMATION>", str(data["content"])[0:50000]),
            },
        ]
        response = self.client.chat.completions.create(
            model=self.args.model,
            messages=messages,
            temperature=0,
            max_tokens=16000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            seed=42,
        )
        response = response.choices[0].message.content
        print(response) 
        return response

    def postprocess_response(self, response: str) -> str:
        """
        postprocess response
        """
        # delete line breaks
        response = response.replace("\n\n", "")
        return response

    def main(self) -> None:
        """
        main function to get perplexity response
        """
        for data in tqdm(self.data, dynamic_ncols=True):
            response = self.get_chatgpt_response(data)
            # delete content
            data.pop("content")
            self.save_list.append(data | {"response": response})


if __name__ == "__main__":
    args = set_args()
    perplexity = ChatGPT(args)
    perplexity.main()

    # save list to jsonl
    with jsonlines.open(args.save_path_jsonl, mode="w") as writer:
        for data in perplexity.save_list:
            writer.write(data)
