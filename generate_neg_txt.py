import os

neg_dir = "neg"
with open("neg.txt", "w") as f:
    for file in os.listdir(neg_dir):
        if file.lower().endswith(('.jpg', '.png')):
            f.write(f"{neg_dir}/{file}\n")

print("✅ neg.txt 產生完成！")
