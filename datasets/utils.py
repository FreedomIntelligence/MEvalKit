import json

def load_dataset_info(dataset_info_path):
    with open(dataset_info_path, "r") as f:
        dataset_info = json.load(f)
    return dataset_info

if __name__ == "__main__":
    dataset_info = load_dataset_info("/mnt/nvme1n1/yuang_workspace/EvaluatorKit/datasets/dataset_info.json")
    print(dataset_info["MMLU"])