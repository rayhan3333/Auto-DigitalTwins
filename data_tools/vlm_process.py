import re
import os
import json
import ast
import base64
import argparse
from pathlib import Path
from openai import OpenAI, AzureOpenAI


class GPT:
    def __init__(self, api_key, endpoint=None, base_url=None, api_version="2025-03-01-preview"):
        if endpoint is not None:
            self.client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=api_version,
            )
        else:
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
        self.model_name = "gpt-4o"

    def encode_image_as_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def payload_joint_info(self, image_path_list: list, data='videoartgs', mode='video') -> dict:
        if data == 'v2a' and mode == 'image':
            system_prompt = """
            ```
            ### Task Description ###
            The user will provide a picture of an articulated object. The system will analyze:
            1.identify the articulated object in the picture.
            2.identify the movable parts of the articulated object has been moved in the picture.
            3.identify the joint type of the moved part.(hinge or slider)
            4.output the information in the form of a json file.
            For the static part, output joint type as heavy. For the movable part, output the joint type as hinge (revolute joint) or slider (prismatic joint).
            Example answer: 
            The articulated object is a cabinet.
            The moved part are the top-drawer and right-door,there is still a left door.
            The joint type is a slider, hinge.
            The output json file is:
           [
                {
                    "id": 0,
                    "name": "cabinet_base",
                    "joint": "heavy",
                    "parent": -1
                },
                {
                    "id": 1,
                    "name": "door",
                    "joint": "hinge",
                    "parent": 0
                }
            ]
            ```
            """.strip()
        if data == 'v2a' and mode == 'video':
            system_prompt = """
            ```
            ### Task Description ###
            The user will provide a sequence of pictures of an articulated object. The system will analyze:
            1.identify the articulated object in the picture.
            2.identify the movable parts of the articulated object has been moved in the pictures.
            3.identify the joint type of the moved part.(hinge or slider)
            4.output the information in the form of a json file.
            For the static part, output joint type as heavy. For the movable part, output the joint type as hinge (revolute joint) or slider (prismatic joint).
            Example answer: 
            The articulated object is a cabinet.
            The moved part are the top-drawer and right-door,there is still a left door.
            The joint type is a slider, hinge.
            The output json file is:
           [
                {
                    "id": 0,
                    "name": "cabinet_base",
                    "joint": "heavy",
                    "parent": -1
                },
                {
                    "id": 1,
                    "name": "door",
                    "joint": "hinge",
                    "parent": 0
                }
            ]
            ```
            """.strip()
        if data == 'videoartgs' and mode == 'image':
            system_prompt = """```
            ### Task Description ###
            The user will provide a picture of an articulated object. The system will analyze:
            1.identify the articulated object in the picture.
            2.identify the movable parts of the articulated object has been moved in the picture.
            3.identify the joint type of the moved part.(hinge or slider)
            4.output the information in the form of a json file.
            For the static part, output joint type as heavy. For the movable part, output the joint type as hinge (revolute joint) or slider (prismatic joint).
            Example answer: 
            The articulated object is a cabinet.
            The moved part are the top-drawer and right-door,there is still a left door.
            The joint type is a slider, hinge.
            The output json file is:
           [
                {
                    "id": 0,
                    "name": "cabinet_base",
                    "joint": "heavy",
                    "parent": -1
                },
                {
                    "id": 1,
                    "name": "top_drawer",
                    "joint": "slider",
                    "parent": 0
                },
                {
                    "id": 2,
                    "name": "bottom_drawer",
                    "joint": "slider",
                    "parent": 0
                },
                {
                    "id": 3,
                    "name": "middle_drawer",
                    "joint": "slider",
                    "parent": 0
                }
            ]
            ```
            """.strip()
            
        if data == 'videoartgs' and mode == 'video':
            system_prompt = """```
            ### Task Description ###
            The user will provide a sequence of pictures of articulated objects, some of which shows how human interact with it. The system will analyze:
            1.identify the articulated object in each picture.
            2.identify the static part of this articulated object.
            3.identify all movable parts of the object and output the composed information in the form of one json file.
            For the static part, output joint type as heavy. For the movable part, output the joint type as hinge (revolute joint) or slider (prismatic joint).
            Example answer 1: 
            The articulated object is a cabinet.
            The static part is the cabinet base.
            The movable parts are the top-drawer, bottom-drawer, and middle-drawer.
            The joint type is a slider.
            The output json file is:
            [
                {
                    "id": 0,
                    "name": "cabinet_base",
                    "joint": "heavy",
                    "parent": -1
                },
                {
                    "id": 1,
                    "name": "top_drawer",
                    "joint": "slider",
                    "parent": 0
                },
                {
                    "id": 2,
                    "name": "bottom_drawer",
                    "joint": "slider",
                    "parent": 0
                },
                {
                    "id": 3,
                    "name": "middle_drawer",
                    "joint": "slider",
                    "parent": 0
                }
            ]
            ```
            """.strip()
        user_prompt = "alnalyze the picture of an articulated object and output the information in the form of a json file."
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        if mode == "image":
            image_path = image_path_list[0]
            base64_img = self.encode_image_as_base64(image_path)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}} 
                ]
            })
            return {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 800,
            }

        else:
            user_content = [{"type": "text", "text": user_prompt}]
            for idx, image_path in enumerate(image_path_list):
                base64_img = self.encode_image_as_base64(image_path)
                image_url_var = f"image_url_{idx}"
                globals()[image_url_var] = f"data:image/png;base64,{base64_img}"
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": globals()[image_url_var]}
                })

            messages.append({
                "role": "user",
                "content": user_content
            })
            return {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 800,
            }
        
    def __call__(self, payload: dict) -> str:
        resp = self.client.chat.completions.create(**payload)
        return resp.choices[0].message.content
    
    
def fix_common_json_errors(text: str) -> str:
    """
    Fix simple, common JSON formatting issues.
    You can expand this as needed.
    """
    # Replace image quotes with double quotes
    text = text.replace("'", '"')
    # Remove trailing commas
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return text


def parse_gpt_output(raw: str) -> dict:
    """
    Robustly parse GPT output string into a Python dictionary or list.
    Handles ```json blocks, common format issues, etc.
    """

    # Step 1: Try to extract JSON block from ```json ... ``` code block
    json_match = re.search(r"```json\s*(\[.*?\]|\{.*?\})\s*```", raw, re.DOTALL)
    if json_match:
        candidate = json_match.group(1).strip()
    else:
        # Fallback to original logic
        # Strip Markdown code block markers if present
        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        candidate = code_block_match.group(1) if code_block_match else raw.strip()

        # Remove leftover Markdown backticks if still present
        candidate = re.sub(r"^```(?:json)?", "", candidate.strip())
        candidate = re.sub(r"```$", "", candidate.strip())

    # Step 2: Try parsing as JSON
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Step 3: Try fixing common JSON issues and parse again
    try:
        fixed = fix_common_json_errors(candidate)
        return json.loads(fixed)
    except Exception:
        pass

    # Step 4: Try using ast.literal_eval as a last resort
    try:
        parsed = ast.literal_eval(candidate)
        if not isinstance(parsed, (dict, list)):
            raise ValueError("Parsed object is not a dict or list")
        return parsed
    except Exception as e:
        raise ValueError(f"Unable to parse GPT output as JSON or Python literal: {e}")


def save_as_json(data: dict, filename: str) -> None:
    """
    Pretty‐print the dictionary `data` into `filename` as JSON.
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✅ Written parsed JSON to {filename}")


def select_frames(folder, start_index=0, interval=5):
    """
    Extract images from a directory in encoded order.

    Args:
        folder (str): The folder path where the images are located
        start_index (int): The starting frame index (e.g. 50 means from 000050.png)
        interval (int): The interval between frames to select

    Returns:
        List[str]: The list of selected image paths
    """
    # Get all .png files
    all_images = [f for f in os.listdir(folder) if f.endswith('.png') and len(f) == 10]
    
    # Parse the encoded numbers and sort
    sorted_images = sorted(
        [f for f in all_images if f[:6].isdigit()],
        key=lambda x: int(x[:6])
    )

    selected_paths = []
    for i, fname in enumerate(sorted_images):
        frame_id = int(fname[:6])
        if frame_id >= start_index and (frame_id - start_index) % interval == 0:
            selected_paths.append(os.path.join(folder, fname))

    return selected_paths



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate one image URL with GPT-4o")
    parser.add_argument("--api_key", default='')
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--base_url", default="https://api.poe.com/v1")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default='videoartgs', choices=['videoartgs', 'v2a'])
    parser.add_argument("--subset", type=str, default='realscan', choices=['sapien', 'realscan'])
    parser.add_argument("--mode", type=str, default='video', choices=['image', 'video'])
    parser.add_argument("--video_name", type=str, default='cab1',
                        help="Scene name")
    args = parser.parse_args()
    data_dir = Path(args.data_dir).resolve()
    video_name = args.video_name
    image_path = os.path.join(data_dir, args.dataset, args.subset, video_name, "images")
    output_dir = os.path.join(data_dir, args.dataset, args.subset, video_name)
    gpt = GPT(api_key=args.api_key, endpoint=args.endpoint, base_url=args.base_url)
    if args.dataset == 'v2a':
        n_canonical = 24
    elif args.dataset == 'videoartgs':
        if args.subset == 'realscan':
            n_canonical = 100
        else:   
            n_canonical = 150
    if args.mode =='image':
        imgs = os.listdir(image_path)
        imgs = sorted(
            [f for f in imgs if f.endswith('.png')],
            key=lambda x: int(os.path.splitext(x)[0])
        )
        image_path_list = [os.path.join(image_path, imgs[n_canonical])]
    else:
        image_path_list = select_frames(image_path, start_index=n_canonical, interval=10)
    print(image_path_list)
    payload  = gpt.payload_joint_info(image_path_list, data=args.dataset,  mode=args.mode)
    response = gpt(payload)

    print("=== GPT Response ===")
    print(response)
    dict = parse_gpt_output(response)
    if args.mode =='image':
        save_as_json(dict, os.path.join(output_dir,"joint_infos_vlm_img.json"))
    else:
        save_as_json(dict, os.path.join(output_dir,"joint_infos_vlm.json"))