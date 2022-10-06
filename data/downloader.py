"""Downloads dataset CSV files into their respective directories."""

import urllib.request

data_dir = "./data"
datasets = {}
with open(f"{data_dir}/urls.txt", encoding='utf8') as url_file:
    for line in url_file:
        (dataset_name, url) = line.split(": ")
        print(f"Downloading {dataset_name.upper()} dataset from: {url}")
        response = urllib.request.urlopen(url).read().decode('utf-8')
        file_name = f'{data_dir}/{dataset_name}/categorical_data.csv'
        with open(file_name, 'w', encoding='utf8') as csv_file:
            csv_file.write(response)
