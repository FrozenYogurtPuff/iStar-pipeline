import json

if __name__ == "__main__":
    FILENAME = "goalModel.txt"
    with open(FILENAME, "r") as file:
        result = json.load(file)
    actor = len(result["actors"])
    intention = 0
    relation = 0
    dependency = len(result["dependencies"])
    for act in result["actors"]:
        intention += len(act["nodes"])
    for link in result["links"]:
        if "Dependency" not in link["type"]:
            relation += 1

    print(actor, intention, relation, dependency)
