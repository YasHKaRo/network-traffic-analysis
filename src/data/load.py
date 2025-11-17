import pandas as pd
from pathlib import Path

def load_unsw_nb15_parquet(data_dir: str = "../data/raw") -> pd.DataFrame:
    """
    Загрузка датасета UNSW-NB15 в формате Parquet
    Args:
        data_dir: Папка с исходными данными
    Returns:
        pd.DataFrame: Загруженный датасет
    """
    data_path = Path(data_dir)

    # Поиск файлов из датасета
    parquet_files = list(data_path.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"Я не нашёл файлы в {data_path}, ты меня обманываешь, да?")

    print(f"Нашёл! {len(parquet_files)} файлов:")
    for file in parquet_files:
        print(f" - {file.name}")

    # Загружаем и объединяем все части
    dfs = []
    for file in parquet_files:
        print(f"Загружаем {file.name}...")
        df_part = pd.read_parquet(file)
        print(f"  Shape: {df_part.shape}")
        print(f"  Columns: {len(df_part.columns)}")
        dfs.append(df_part)

    # Объединяем все данные
    df = pd.concat(dfs, ignore_index=True)

    print(f"Датасет установлен успешно: {df.shape[0]} строки, {df.shape[1]} столбцы")
    return df