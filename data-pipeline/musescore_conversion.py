import os, json, subprocess, random



def main():
    os.chdir('dataset/mxl')

    # write json files for every 'step' mxl files
    files = os.listdir()
    random.shuffle(files)
    n = len(files)
    step = n // 15 + 1

    for i in range(n // step + 1):
        l = []
        start = i * step
        end = start + step
        if end > n:
            end = n
        for file in files[start:end]:
            if file.endswith('.mxl'):
                l.append({
                    "in": f"../mxl/{file}",
                    "out": f"../audio/{file.replace('.mxl', '.wav')}"
                })
        with open(f'../json/job{i}.json', 'w') as f:
            json.dump(l, f, indent=2)

    # write batch files for every json file in json folder
    os.chdir('../json')
    for file in os.listdir():
        if file.endswith('.json'):
            with open(f'../bat/{file.replace(".json", ".bat")}', 'w') as f:
                f.write(f"MuseScore4.exe -j ../json/{file}")

    # run batch files in parallel
    os.chdir('../bat')
    for file in os.listdir():
        if file.endswith('.bat'):
            subprocess.Popen(file, shell=True)

if __name__ == "__main__":
    main()