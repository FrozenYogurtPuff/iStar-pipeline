from pathlib import Path

if __name__ == "__main__":
    K = 5
    base_dir = "../pretrained_model/2022_Kfold/actor"
    all_dict: dict[str, dict] = dict()
    for i in range(K):
        file = (
            Path(base_dir)
            .joinpath(str(i))
            .joinpath("output")
            .joinpath("eval_results.txt")
        )
        with open(file, "r") as f:
            all_lines = f.readlines()
            for line in reversed(all_lines):
                if line.strip() == "":
                    continue
                if line.startswith("micro avg") or line.startswith(
                    "macro avg"
                ):
                    continue
                if line.startswith("report"):
                    break
                name, p, r, f1, _ = list(
                    filter(lambda x: x != "", line.split(" "))
                )
                if name not in all_dict:
                    all_dict[name] = dict()
                    all_dict[name]["p"] = list()
                    all_dict[name]["r"] = list()
                    all_dict[name]["f1"] = list()
                all_dict[name]["p"].append(float(p))
                all_dict[name]["r"].append(float(r))
                all_dict[name]["f1"].append(float(f1))
    for name in all_dict:
        assert (
            len(all_dict[name]["p"])
            == len(all_dict[name]["r"])
            == len(all_dict[name]["f1"])
            == K
        )
        p_avg = sum(all_dict[name]["p"]) / len(all_dict[name]["p"])
        r_avg = sum(all_dict[name]["r"]) / len(all_dict[name]["r"])
        f1_avg = sum(all_dict[name]["f1"]) / len(all_dict[name]["f1"])
        print(name, str(p_avg), str(r_avg), str(f1_avg))
