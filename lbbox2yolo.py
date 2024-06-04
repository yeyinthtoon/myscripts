import json
from io import BytesIO
from pathlib import Path

import requests
import typer
from PIL import Image
from rich import print as rich_print
from collections import defaultdict

def create_folder(folder_dir):
    if not folder_dir.exists():
        folder_dir.mkdir(parents=True)


def load_image(url):
    response = requests.get(url)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    return image


def read_json(json_path):
    with open(json_path, "r", encoding="utf-8") as jsf:
        json_data = json.load(jsf)
    return json_data


def box_lbbox2yolo(
    project_id: str,
    lbbox_annotaion_json_path: Path,
    label_map_path: Path,
    dataset_dir: Path,
    skip_unknown_label: bool = False,
):
    label_dir = dataset_dir / "labels"
    image_dir = dataset_dir / "images"
    label_map = read_json(label_map_path)
    create_folder(label_dir)
    create_folder(image_dir)
    with open(lbbox_annotaion_json_path, "r", encoding="utf-8") as jsonl:
        annotations = list(jsonl.readlines())
    skips_count = defaultdict(int)
    for anno in annotations:
        annotation = json.loads(anno)
        image_name = annotation["data_row"]["external_id"]
        image_path = image_dir / image_name
        if not image_path.exists():
            rich_print(
                "[green]Image file not exist, Trying to download from labelbox[/green]"
            )
        image = load_image(annotation["data_row"]["row_data"])
        image_width, image_height = image.size
        image.save(image_path)

        label_path = (label_dir / image_name).with_suffix(".txt")
        yolo_annotations = []
        assert len(annotation["projects"][project_id]["labels"]) == 1, "case not handle"
        for instance in annotation["projects"][project_id]["labels"]:
            for annotated_object in instance["annotations"]["objects"]:
                x = annotated_object["bounding_box"]["left"]
                y = annotated_object["bounding_box"]["top"]
                width = annotated_object["bounding_box"]["width"]
                height = annotated_object["bounding_box"]["height"]
                xc = (x + width) / 2
                yc = (y + height) / 2
                class_name = annotated_object["name"]
                if class_name not in label_map:
                    if skip_unknown_label:
                        skips_count[class_name] += 1
                        continue
                    else:
                        rich_print(
                            f"[red] {class_name} not presented in label_map, please update label"
                            "map or use skip_unknown_label=True[/red]"
                        )
                        raise typer.Exit()
                class_id = label_map[class_name]
                yolo_annotations.append(
                    f"{class_id} {xc/image_width} {yc/image_height} {width/image_width} {height/image_height}"
                )
        yolo_annotation = "\n".join(yolo_annotations)
        with open(label_path, "w", encoding="utf-8") as labelfile:
            labelfile.write(yolo_annotation)
    if skips_count:
        rich_print(f"skipped instances: {dict(skips_count)}")


if __name__ == "__main__":
    typer.run(box_lbbox2yolo)
