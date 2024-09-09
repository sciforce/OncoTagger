import pandas as pd

# Список файлов для объединения с полным путем
files = [
    'D:\\results\\1-1000.xls',
    'D:\\results\\1001-2000.xls',
    'D:\\results\\2001-3000.xls',
    'D:\\results\\3001-4000.xls',
    'D:\\results\\4001-5000.xls',
    'D:\\results\\5001-6000.xls',
    'D:\\results\\6001-6466.xls'
]

# Создаем пустой DataFrame для объединения данных
combined_df = pd.DataFrame()

# Читаем каждый файл и добавляем его в общий DataFrame
for file in files:
    df = pd.read_excel(file, engine='xlrd')
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# Сохраняем объединенный DataFrame в новый Excel файл
combined_df.to_excel('D:\\results\\1-6466.xlsx', index=False)

print("Объединение завершено. Файл сохранен как 'D:\\results\\1-6466.xlsx'.")
