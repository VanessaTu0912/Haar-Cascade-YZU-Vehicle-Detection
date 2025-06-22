import os
import xml.etree.ElementTree as ET

annotations_dir = "annotations"
output_txt = "pos.txt"
image_dir = "pos"

with open(output_txt, 'w') as out:
    for filename in os.listdir(annotations_dir):
        if filename.endswith(".xml"):
            xml_path = os.path.join(annotations_dir, filename)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            img_filename = root.find("filename").text
            objects = root.findall("object")

            # 如果該圖中沒有標註，就跳過
            if not objects:
                continue

            line = f"{image_dir}/{img_filename} {len(objects)}"

            for obj in objects:
                bndbox = obj.find("bndbox")
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)
                w = xmax - xmin
                h = ymax - ymin
                line += f" {xmin} {ymin} {w} {h}"

            out.write(line + "\n")

print("✅ 已成功轉換為 pos.txt！")
