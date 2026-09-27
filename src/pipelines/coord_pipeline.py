import pandas as pd 
from src.utils.configs import import_config
import sys


def make_columns_description(columns:list[str]) -> str:
    description:str = "#?"
    for column in columns:
        description+= f" {column}"

    return description+"\n"

def make_file_name(csv_file_path:str) -> str: 
    return (csv_file_path.split("/")[-1].split(".")[0])+".txt"

def make_coord_file(csv_file_path:str, dir_clean_coord_path:str,
                    coords_config_path:str = "configs/coords_configs.toml") -> None:
    df:pd.DataFrame = pd.read_csv(csv_file_path)
    df.columns = df.columns.str.lstrip("#").str.strip()
    coords_configs:dict[str,str] = import_config(coords_config_path)

    

    for key in coords_configs.keys(): 
        df[key] = coords_configs[key]


    file_path:str = dir_clean_coord_path.rstrip("/") + "/" + make_file_name(csv_file_path)
    

    with open(file_path,"w") as f:
        f.write(make_columns_description(df.columns))
        df.to_csv(f, sep=" ", index=False, header=False)


if __name__ == "__main__": 
    csv_file_path , dir_clean_coord_path, coords_config_path = sys.argv[1:]

    if coords_config_path == "-":
        make_coord_file(csv_file_path,dir_clean_coord_path)
    else: 
        make_coord_file(csv_file_path,dir_clean_coord_path,coords_config_path)
    



